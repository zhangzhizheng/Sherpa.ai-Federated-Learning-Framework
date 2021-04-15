import numpy as np
from unittest.mock import Mock, patch, call
import pytest

from shfl.model.vertical_deep_learning_model import VerticalNeuralNetClient
from shfl.model.vertical_deep_learning_model import VerticalNeuralNetServer


@pytest.fixture
def global_vars():
    global_vars = {"n_features": 23,
                   "n_classes": 2,
                   "n_embeddings": 3,
                   "num_data": 100,
                   "batch_size": 32,
                   "epoch": 2,
                   "metrics": [0, 1, 2, 3],
                   "device": "cpu"}

    return global_vars


@pytest.fixture
def data_loader(global_vars):
    data = np.random.rand(global_vars["num_data"],
                          global_vars["n_features"])
    data_loader = []
    indices_batch_split = np.arange(len(data))[global_vars["batch_size"]::
                                               global_vars["batch_size"]]
    batch_indices = np.array_split(np.arange(len(data)), indices_batch_split)
    for indices_samples in batch_indices:
        x = data[indices_samples]
        data_loader.append([x, indices_samples])

    return data_loader, data


def mock_torch_as_tensor(value, **kwargs):
    return value


def mock_torch_autograd_variable(value, **kwargs):
    return value


@pytest.mark.parametrize("vertical_model_type",
                         [VerticalNeuralNetClient,
                          VerticalNeuralNetServer])
@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
def test_vertical_model_private_data(mock_get_params, global_vars,
                                     vertical_model_type):
    loss = Mock()
    optimizer = Mock()
    model = Mock()
    mock_get_params.return_value = \
        [np.random.rand(global_vars["n_embeddings"], global_vars["n_features"]),
         np.random.rand(global_vars["n_embeddings"])]

    vertical_model = vertical_model_type(
        model, loss, optimizer,
        global_vars["batch_size"], global_vars["epoch"],
        global_vars["metrics"], global_vars["device"])

    assert vertical_model._model.id == model.id
    assert vertical_model._data_shape == global_vars["n_features"]
    assert vertical_model._labels_shape == (global_vars["n_embeddings"],)
    assert vertical_model._loss.id == loss.id
    assert vertical_model._optimizer.id == optimizer.id
    assert vertical_model._batch_size == global_vars["batch_size"]
    assert vertical_model._epochs == global_vars["epoch"]
    assert np.array_equal(vertical_model._metrics, global_vars["metrics"])
    assert vertical_model._device == global_vars["device"]


@pytest.fixture
@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
@patch('shfl.model.vertical_deep_learning_model.torch')
def vertical_client_model(mock_torch, mock_get_params, global_vars):
    optimizer = Mock()
    model = Mock()
    loss = Mock()
    embeddings = Mock()
    val_embeddings = np.random.rand(global_vars["n_embeddings"])
    embeddings.detach().cpu().numpy.return_value = val_embeddings

    model.return_value = embeddings
    mock_get_params.return_value = \
        [np.random.rand(global_vars["n_embeddings"], global_vars["n_features"]),
         np.random.rand(global_vars["n_embeddings"])]

    mock_torch.as_tensor = mock_torch_as_tensor

    vert_nn = VerticalNeuralNetClient(model, loss, optimizer,
                                      global_vars["batch_size"],
                                      global_vars["epoch"],
                                      global_vars["metrics"],
                                      global_vars["device"])

    return vert_nn, embeddings


def test_vertical_client_model_train_forward(vertical_client_model,
                                             global_vars,
                                             data_loader):
    vert_nn, embeddings = vertical_client_model
    batch_loader, data = data_loader

    call_args = []
    for _ in batch_loader:
        vert_nn.train(data, labels=None)
        vert_nn._batch_counter += 1
        call_args.append(vert_nn._model.call_args)

    for i, batch in enumerate(batch_loader):
        input_data, indices_samples = batch
        np.testing.assert_array_equal(call_args[i][0][0], input_data)


@patch('shfl.model.vertical_deep_learning_model.torch')
def test_vertical_client_model_train_backpropagation(mock_torch,
                                                     vertical_client_model,
                                                     global_vars,
                                                     data_loader):
    vert_nn, embeddings = vertical_client_model
    batch_loader, data = data_loader

    vert_nn._embeddings = Mock()

    embeddings_grads = []
    backpropagation_calls_args = []
    for batch in batch_loader:
        _, indices_samples = batch
        grads = np.random.rand(len(indices_samples),
                               global_vars["n_embeddings"])
        meta_params = (grads, indices_samples)
        mock_torch.as_tensor = mock_torch_as_tensor
        vert_nn._embeddings_indices = indices_samples
        vert_nn.train(data, labels=None, meta_params=meta_params)
        embeddings_grads.append(grads)
        backpropagation_calls_args.append(
            vert_nn._embeddings.backward.call_args)

    optimizer_calls = []
    for _ in embeddings_grads:
        optimizer_calls.extend([call.zero_grad(), call.step()])
    vert_nn._optimizer.assert_has_calls(optimizer_calls)

    # Backpropagation calls: need to check each array separately
    for i, grads in enumerate(embeddings_grads):
        np.testing.assert_array_equal(
            backpropagation_calls_args[i][1]["gradient"], grads)


def test_vertical_client_model_get_meta_params(vertical_client_model,
                                               global_vars,
                                               data_loader):
    vert_nn, embeddings = vertical_client_model
    batch_loader, data = data_loader

    meta_params_list = []
    val_embeddings_list = []
    for _ in batch_loader:
        embeddings = Mock()
        val_embeddings = np.random.rand(global_vars["batch_size"],
                                        global_vars["n_embeddings"])
        embeddings.detach().cpu().numpy.return_value = val_embeddings
        vert_nn._model.return_value = embeddings
        vert_nn.train(data, labels=None)
        vert_nn._batch_counter += 1

        meta_params = vert_nn.get_meta_params()

        meta_params_list.append(meta_params)
        val_embeddings_list.append(val_embeddings)

    for i, meta_params in enumerate(meta_params_list):
        np.testing.assert_array_equal(meta_params[0], val_embeddings_list[i])
        np.testing.assert_array_equal(meta_params[1], batch_loader[i][1])


@pytest.fixture
@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
@patch('shfl.model.vertical_deep_learning_model.torch')
def vertical_server_model_train(mock_torch, mock_get_params,
                                global_vars):
    optimizer = Mock()
    model = Mock()
    loss = Mock()
    val_prediction = np.random.rand(global_vars["batch_size"],
                                    global_vars["n_classes"])
    model.return_value = val_prediction
    mock_get_params.return_value = \
        [np.random.rand(global_vars["n_classes"], global_vars["n_embeddings"]),
         np.random.rand(global_vars["n_classes"])]

    vert_nn = VerticalNeuralNetServer(model, loss, optimizer,
                                      global_vars["batch_size"],
                                      global_vars["epoch"],
                                      global_vars["metrics"],
                                      global_vars["device"])

    val_grad_embeddings = np.random.rand(global_vars["batch_size"],
                                         global_vars["n_embeddings"])
    embeddings = Mock()
    embeddings.grad.detach().cpu().numpy.return_value = val_grad_embeddings

    embeddings_indices = np.random.choice(a=global_vars["num_data"],
                                          size=global_vars["batch_size"],
                                          replace=False)

    val_labels = np.random.randint(low=0,
                                   high=global_vars["n_classes"],
                                   size=(global_vars["num_data"], 1))

    meta_params = (embeddings, embeddings_indices)
    mock_torch.as_tensor = mock_torch_as_tensor
    mock_torch.autograd.Variable = mock_torch_autograd_variable

    vert_nn.train(data=None, labels=val_labels, meta_params=meta_params)

    return vert_nn, embeddings, val_grad_embeddings, \
        embeddings_indices, val_labels, val_prediction


def test_vertical_server_model_train(vertical_server_model_train):
    vert_nn, embeddings, _, embeddings_indices, \
        val_labels, val_prediction = vertical_server_model_train

    optimizer_calls = [call.zero_grad(), call.step()]
    model_calls = [call(embeddings)]

    vert_nn._optimizer.assert_has_calls(optimizer_calls)
    vert_nn._model.assert_has_calls(model_calls)

    vert_nn._loss.assert_has_calls([call().backward()])
    # Loss calls: need to check each array argument separately
    np.testing.assert_array_equal(vert_nn._loss.call_args[0][0],
                                  val_prediction)
    np.testing.assert_array_equal(vert_nn._loss.call_args[0][1],
                                  val_labels[embeddings_indices])


def test_vertical_server_model_get_meta_params(vertical_server_model_train):
    vert_nn, _, val_grad_embeddings, \
        embeddings_indices, _, _ = vertical_server_model_train

    meta_params = vert_nn.get_meta_params()

    np.testing.assert_array_equal(meta_params[0],
                                  val_grad_embeddings)
    np.testing.assert_array_equal(meta_params[1],
                                  embeddings_indices)
