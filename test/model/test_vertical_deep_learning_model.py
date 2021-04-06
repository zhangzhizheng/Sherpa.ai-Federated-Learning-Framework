import numpy as np
from unittest.mock import Mock, patch, call
import pytest
import torch
import torch.nn as nn

from shfl.model.vertical_deep_learning_model import VerticalNeuralNetClient
from shfl.model.vertical_deep_learning_model import VerticalNeuralNetServer


@pytest.fixture
def global_vars():
    global_vars = {"n_features": 12,
                   "n_embeddings": 3,
                   "num_data": 100,
                   "batch_size": 32,
                   "epoch": 2,
                   "metrics": [0, 1, 2, 3],
                   "device": "cpu",
                   "num_data": 100}

    return global_vars


@pytest.fixture
def data_loader(global_vars):
    data = np.array([np.random.rand(global_vars["n_features"])
                     for i in range(global_vars["num_data"])])
    data_loader = []
    split_indices = np.array_split(np.arange(global_vars["num_data"]),
                                   int(round(global_vars["num_data"] /
                                             global_vars["batch_size"])))
    for indices_samples in split_indices:
        x = data[indices_samples]
        x = [x]
        data_loader.append([x, indices_samples])

    return data_loader, data


@pytest.fixture
def model_builder(vertical_node_type, n_input, n_output):
    loss = Mock()
    optimizer = Mock()
    model = Mock()
    model.parameters.return_value = \
        [nn.Parameter(torch.rand((n_output, n_input))),
         nn.Parameter(torch.rand(n_output))]

    batch = 32
    epoch = 2
    metrics = [0, 1, 2, 3]
    device = 'device0'
    model = vertical_node_type(model, loss, optimizer,
                               batch, epoch, metrics, device)

    return model


@pytest.mark.parametrize("vertical_node_type",
                         [VerticalNeuralNetClient,
                          VerticalNeuralNetServer])
@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
def test_vertical_node_model_private_data(mock_get_params, global_vars,
                                          vertical_node_type):

    loss = Mock()
    optimizer = Mock()
    model = Mock()
    mock_get_params.return_value = \
        [np.random.rand(global_vars["n_embeddings"], global_vars["n_features"]),
         np.random.rand(global_vars["n_embeddings"])]

    dpl = vertical_node_type(model, loss, optimizer,
                             global_vars["batch_size"], global_vars["epoch"],
                             global_vars["metrics"], global_vars["device"])

    assert dpl._model.id == model.id
    assert dpl._data_shape == global_vars["n_features"]
    assert dpl._labels_shape == (global_vars["n_embeddings"],)
    assert dpl._loss.id == loss.id
    assert dpl._optimizer.id == optimizer.id
    assert dpl._batch_size == global_vars["batch_size"]
    assert dpl._epochs == global_vars["epoch"]
    assert np.array_equal(dpl._metrics, global_vars["metrics"])
    assert dpl._device == global_vars["device"]


@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
@patch('shfl.model.vertical_deep_learning_model.torch')
@patch('shfl.model.vertical_deep_learning_model.TensorDatasetIndex')
@patch('shfl.model.vertical_deep_learning_model.DataLoader')
def test_vertical_client_model_forward(mock_dl, mock_tdt,
                                       mock_torch, mock_get_params,
                                       global_vars, data_loader):

    batch_loader, data = data_loader
    mock_dl.return_value = batch_loader

    optimizer = Mock()
    model = Mock()
    embeddings = np.random.rand(global_vars["n_embeddings"])
    model.return_value = embeddings
    mock_get_params.return_value = \
        [np.random.rand(global_vars["n_embeddings"], global_vars["n_features"]),
         np.random.rand(global_vars["n_embeddings"])]

    loss = None
    vert_nn = VerticalNeuralNetClient(model, loss, optimizer,
                                      global_vars["batch_size"],
                                      global_vars["epoch"],
                                      global_vars["metrics"],
                                      global_vars["device"])

    for _ in batch_loader:
        vert_nn.train(data, labels=None)
        vert_nn._batch_counter += 1

    model_calls = []
    for batch in batch_loader:
        inputs, indices_samples = batch
        model_calls.extend([call(inputs[0])])

    vert_nn._model.assert_has_calls(model_calls)


@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
@patch('shfl.model.vertical_deep_learning_model.torch')
@patch('shfl.model.vertical_deep_learning_model.TensorDatasetIndex')
@patch('shfl.model.vertical_deep_learning_model.DataLoader')
def test_vertical_client_model_backpropagation(mock_dl, mock_tdt,
                                               mock_torch, mock_get_params,
                                               global_vars, data_loader):

    batch_loader, data = data_loader
    optimizer = Mock()
    model = Mock()
    mock_get_params.return_value = \
        [np.random.rand(global_vars["n_embeddings"], global_vars["n_features"]),
         np.random.rand(global_vars["n_embeddings"])]

    loss = None
    vert_nn = VerticalNeuralNetClient(model, loss, optimizer,
                                      global_vars["batch_size"],
                                      global_vars["epoch"],
                                      global_vars["metrics"],
                                      global_vars["device"])

    vert_nn._embeddings = Mock()
    vert_nn._data_loader = batch_loader

    embeddings_grads = []
    for batch in batch_loader:
        _, indices_samples = batch
        grads = np.random.rand(len(indices_samples))
        meta_params = (grads, indices_samples)
        vert_nn.train(data, labels=None, meta_params=meta_params)
        embeddings_grads.append(grads)

    optimizer_calls = []
    backpropagation_calls = []
    for grads in embeddings_grads:
        optimizer_calls.extend([call.zero_grad(), call.step()])
        backpropagation_calls.extend([call.backward(gradient=grads)])

    vert_nn._optimizer.assert_has_calls(optimizer_calls)
    vert_nn._embeddings.assert_has_calls(backpropagation_calls)


@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
@patch('shfl.model.vertical_deep_learning_model.torch')
@patch('shfl.model.vertical_deep_learning_model.TensorDatasetIndex')
@patch('shfl.model.vertical_deep_learning_model.DataLoader')
def test_vertical_client_get_meta_params(mock_dl, mock_tdt,
                                         mock_torch, mock_get_params,
                                         global_vars, data_loader):

    batch_loader, data = data_loader
    mock_dl.return_value = batch_loader

    optimizer = Mock()
    model = Mock()
    embeddings = np.random.rand(global_vars["batch_size"],
                                global_vars["n_embeddings"])
    model.return_value = embeddings
    mock_get_params.return_value = \
        [np.random.rand(global_vars["n_embeddings"], global_vars["n_features"]),
         np.random.rand(global_vars["n_embeddings"])]

    loss = None
    vert_nn = VerticalNeuralNetClient(model, loss, optimizer,
                                      global_vars["batch_size"],
                                      global_vars["epoch"],
                                      global_vars["metrics"],
                                      global_vars["device"])

    for batch in batch_loader:
        vert_nn.train(data, labels=None)
        vert_nn._batch_counter += 1

        _, indices_samples = batch
        meta_params = vert_nn.get_meta_params()
        np.testing.assert_array_equal(embeddings, meta_params[0])
        np.testing.assert_array_equal(indices_samples, meta_params[1])
