import numpy as np
from unittest.mock import Mock, patch, call
import pytest

from shfl.model.vertical_deep_learning_model import VerticalNeuralNetClient
from shfl.model.vertical_deep_learning_model import VerticalNeuralNetServer


@pytest.fixture
def global_vars():
    global_vars = {"n_features": 12,
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


@pytest.fixture
@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
@patch('shfl.model.vertical_deep_learning_model.torch')
@patch('shfl.model.vertical_deep_learning_model.TensorDatasetIndex')
@patch('shfl.model.vertical_deep_learning_model.DataLoader')
def fixture_vertical_client_model(mock_dl, mock_tdt,
                                  mock_torch, mock_get_params,
                                  global_vars, data_loader):
    batch_loader, data = data_loader
    mock_dl.return_value = batch_loader

    optimizer = Mock()
    model = Mock()
    embeddings = Mock()
    val_embeddings = np.random.rand(global_vars["n_embeddings"])
    embeddings.detach().cpu().numpy.return_value = val_embeddings

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

    return vert_nn, embeddings, batch_loader, data


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
        mock_torch.from_numpy.return_value = grads
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
    # Mock first batch:
    val_indices_batch = batch_loader[0][1]
    mock_indices_batch = Mock()
    mock_indices_batch.detach().cpu().numpy.return_value = val_indices_batch
    batch_loader[0][1] = mock_indices_batch

    mock_dl.return_value = batch_loader

    optimizer = Mock()
    model = Mock()
    embeddings = Mock()
    val_embeddings = np.random.rand(global_vars["n_embeddings"])
    embeddings.detach().cpu().numpy.return_value = val_embeddings
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

    vert_nn.train(data, labels=None)
    meta_params = vert_nn.get_meta_params()

    np.testing.assert_array_equal(meta_params[0], val_embeddings)
    np.testing.assert_array_equal(meta_params[1], val_indices_batch)


def mock_torch_from_numpy(value):
    return value


def mock_torch_autograd_variable(value, **kwargs):
    return value


@pytest.fixture
@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
@patch('shfl.model.vertical_deep_learning_model.torch')
@patch('shfl.model.vertical_deep_learning_model.TensorDatasetIndex')
@patch('shfl.model.vertical_deep_learning_model.DataLoader')
def fixture_vertical_server_model_train(mock_dl, mock_tdt,
                                        mock_torch, mock_get_params,
                                        global_vars, data_loader):

    batch_loader, data = data_loader
    mock_dl.return_value = batch_loader

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
    labels = Mock()
    labels.return_value = val_labels
    labels.float.return_value = val_labels

    meta_params = (embeddings, embeddings_indices)
    mock_torch.from_numpy = mock_torch_from_numpy
    mock_torch.autograd.Variable = mock_torch_autograd_variable

    vert_nn.train(data=None, labels=labels, meta_params=meta_params)

    return vert_nn, embeddings, val_grad_embeddings, \
        embeddings_indices, val_labels, val_prediction


def test_vertical_server_model_train(fixture_vertical_server_model_train):

    vert_nn, embeddings, _, embeddings_indices, \
        val_labels, val_prediction = fixture_vertical_server_model_train

    optimizer_calls = [call.zero_grad(), call.step()]
    model_calls = [call(embeddings)]

    vert_nn._optimizer.assert_has_calls(optimizer_calls)
    vert_nn._model.assert_has_calls(model_calls)

    # Loss calls: need to check each array argument separately
    np.testing.assert_array_equal(vert_nn._loss.call_args[0][0],
                                  val_prediction)
    np.testing.assert_array_equal(vert_nn._loss.call_args[0][1],
                                  val_labels[embeddings_indices])
    vert_nn._loss.assert_has_calls([call().backward()])


def test_vertical_server_model_get_meta_params(fixture_vertical_server_model_train):

    vert_nn, _, val_grad_embeddings, \
        embeddings_indices, _, _ = fixture_vertical_server_model_train

    meta_params = vert_nn.get_meta_params()

    np.testing.assert_array_equal(meta_params[0],
                                  val_grad_embeddings)
    np.testing.assert_array_equal(meta_params[1],
                                  embeddings_indices)
