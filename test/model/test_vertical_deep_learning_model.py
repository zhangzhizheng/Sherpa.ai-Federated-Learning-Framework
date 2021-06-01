from unittest.mock import Mock, patch, call
import pytest
import numpy as np

from shfl.model.vertical_deep_learning_model import VerticalNeuralNetClient
from shfl.model.vertical_deep_learning_model import VerticalNeuralNetServer


class VerticalNeuralNetClientTest(VerticalNeuralNetClient):
    """Creates a test class for the vertical client model.

    It allows to access some private attributes during testing."""

    @property
    def embeddings(self):
        """Returns the embeddings."""
        return self._embeddings

    @embeddings.setter
    def embeddings(self, embeddings):
        """Sets the embeddings"""
        self._embeddings = embeddings

    @property
    def embeddings_indices(self):
        """Returns the embeddings indices for a data chunk."""
        return self._embeddings_indices

    @embeddings_indices.setter
    def embeddings_indices(self, indices):
        """Sets the embeddings indices for a data chunk."""
        self._embeddings_indices = indices

    @property
    def batch_counter(self):
        """Returns the mini-batch counter."""
        return self._batch_counter

    @batch_counter.setter
    def batch_counter(self, counter):
        """Sets the mini-batch counter."""
        self._batch_counter = counter


@pytest.fixture(name="global_vars")
def fixture_global_vars():
    """Returns the global variables to set up the tests."""
    global_vars = {"n_features": 23,
                   "n_classes": 2,
                   "n_embeddings": 3,
                   "num_data": 100,
                   "batch_size": 32,
                   "epoch": 2,
                   "metrics": [0, 1, 2, 3],
                   "device": "cpu"}

    return global_vars


@pytest.fixture(name="data_loader")
def fixture_data_loader(global_vars):
    """Returns a data loader with random data.

    It is an iterable whose items contain the data mini-batches."""
    data = np.random.rand(global_vars["num_data"],
                          global_vars["n_features"])
    data_loader = []
    indices_batch_split = np.arange(len(data))[global_vars["batch_size"]::
                                               global_vars["batch_size"]]
    batch_indices = np.array_split(np.arange(len(data)), indices_batch_split)
    for indices_samples in batch_indices:
        data_chunk = data[indices_samples]
        data_loader.append([data_chunk, indices_samples])

    return data_loader, data


def side_effect_torch_as_tensor(value, device):
    """Returns the side effect of torch's "as_tensor" method."""
    del device
    return value


def side_effect_torch_autograd_variable(value, requires_grad):
    """Returns the side effect of torch's "autograd variable" method."""
    del requires_grad
    return value


@pytest.fixture(name="vertical_client_model")
@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
@patch('shfl.model.vertical_deep_learning_model.torch')
def fixture_vertical_client_model(mock_torch, mock_get_params, global_vars):
    """Returns the mocked vertical client model and random output embeddings.

    The associated (mocked) pytorch model, loss, and optimizer are also returned."""
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

    mock_torch.as_tensor.side_effect = side_effect_torch_as_tensor

    wrapped_model = VerticalNeuralNetClientTest(model, loss, optimizer,
                                                global_vars["batch_size"],
                                                global_vars["epoch"],
                                                global_vars["metrics"],
                                                global_vars["device"])

    return wrapped_model, embeddings, model, loss, optimizer


@pytest.mark.parametrize("wrapped_model_type",
                         [VerticalNeuralNetClient,
                          VerticalNeuralNetServer])
@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
def test_initialization(mock_get_params, wrapped_model_type, global_vars):
    """Checks that the vertical deep learning model initializes correctly.

    Both Client and Server's models are tested."""
    loss = Mock()
    optimizer = Mock()
    model = Mock()
    mock_get_params.return_value = \
        [np.random.rand(global_vars["n_embeddings"], global_vars["n_features"]),
         np.random.rand(global_vars["n_embeddings"])]

    wrapped_model = wrapped_model_type(
        model, loss, optimizer,
        global_vars["batch_size"], global_vars["epoch"],
        global_vars["metrics"], global_vars["device"])

    assert hasattr(wrapped_model, "_embeddings_indices")
    if isinstance(wrapped_model, VerticalNeuralNetClient):
        assert hasattr(wrapped_model, "_embeddings")
        assert hasattr(wrapped_model, "_batch_counter")
        assert hasattr(wrapped_model, "_epoch_counter")
    if isinstance(wrapped_model, VerticalNeuralNetServer):
        assert hasattr(wrapped_model, "_grad_embeddings")


def test_vertical_client_model_train_forward(vertical_client_model,
                                             data_loader):
    """Checks that the vertical Client model trains correctly.

    The forward step is tested."""
    wrapped_model, _, model, _, _ = vertical_client_model
    batch_loader, data = data_loader

    call_args = []
    for _ in batch_loader:
        wrapped_model.train(data, labels=None)
        wrapped_model.batch_counter += 1
        call_args.append(model.call_args)

    for i, batch in enumerate(batch_loader):
        input_data = batch[0]
        np.testing.assert_array_equal(call_args[i][0][0], input_data)


@patch('shfl.model.vertical_deep_learning_model.torch')
def test_vertical_client_model_train_backpropagation(mock_torch,
                                                     vertical_client_model,
                                                     global_vars,
                                                     data_loader):
    """Checks that the vertical Client model trains correctly.

    The backward step is tested."""
    wrapped_model, _, _, _, optimizer = vertical_client_model
    wrapped_model.embeddings = Mock()
    mock_torch.as_tensor = side_effect_torch_as_tensor
    embeddings_grads = []
    backpropagation_calls_args = []
    for batch in data_loader[0]:
        _, indices_samples = batch
        grads = np.random.rand(len(indices_samples),
                               global_vars["n_embeddings"])
        meta_params = (grads, indices_samples)
        wrapped_model.embeddings_indices = indices_samples
        wrapped_model.train(data_loader[1], labels=None, meta_params=meta_params)
        embeddings_grads.append(grads)
        backpropagation_calls_args.append(
            wrapped_model.embeddings.backward.call_args)

    optimizer_calls = []
    for _ in embeddings_grads:
        optimizer_calls.extend([call.zero_grad(), call.step()])
    optimizer.assert_has_calls(optimizer_calls)

    # Backpropagation calls: need to check each array separately
    for i, grads in enumerate(embeddings_grads):
        np.testing.assert_array_equal(
            backpropagation_calls_args[i][1]["gradient"], grads)


def test_vertical_client_model_get_meta_params(vertical_client_model,
                                               global_vars,
                                               data_loader):
    """Checks that the vertical Client's model gets the Meta parameters correctly."""
    wrapped_model, embeddings, model, _, _ = vertical_client_model
    batch_loader, data = data_loader

    meta_params_list = []
    val_embeddings_list = []
    for _ in batch_loader:
        embeddings = Mock()
        val_embeddings = np.random.rand(global_vars["batch_size"],
                                        global_vars["n_embeddings"])
        embeddings.detach().cpu().numpy.return_value = val_embeddings
        model.return_value = embeddings
        wrapped_model.train(data, labels=None)
        wrapped_model.batch_counter += 1

        meta_params = wrapped_model.get_meta_params()

        meta_params_list.append(meta_params)
        val_embeddings_list.append(val_embeddings)

    for i, meta_params in enumerate(meta_params_list):
        np.testing.assert_array_equal(meta_params[0], val_embeddings_list[i])
        np.testing.assert_array_equal(meta_params[1], batch_loader[i][1])


@pytest.fixture(name="vertical_server_model_train")
@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
@patch('shfl.model.vertical_deep_learning_model.torch')
def fixture_vertical_server_model_train(mock_torch, mock_get_params, global_vars):
    """Returns a trained vertical Server's model used in tests.

    Also, for the considered mini-batch, it also returns:
    the (mocked) embeddings; the value of their gradient;
    the indices; the labels; the prediction;
    the (mocked) model, loss and optimizer."""
    optimizer = Mock()
    model = Mock()
    loss = Mock()
    val_prediction = np.random.rand(global_vars["batch_size"],
                                    global_vars["n_classes"])
    model.return_value = val_prediction
    mock_get_params.return_value = \
        [np.random.rand(global_vars["n_classes"], global_vars["n_embeddings"]),
         np.random.rand(global_vars["n_classes"])]

    wrapped_model = VerticalNeuralNetServer(model, loss, optimizer,
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
    mock_torch.as_tensor = side_effect_torch_as_tensor
    mock_torch.autograd.Variable.side_effect = side_effect_torch_autograd_variable

    wrapped_model.train(data=None, labels=val_labels, meta_params=meta_params)

    return wrapped_model, embeddings, val_grad_embeddings, \
        embeddings_indices, val_labels, val_prediction, \
        model, loss, optimizer


def test_vertical_server_model_train(vertical_server_model_train):
    """Checks that the vertical Server's model trains correctly."""
    _, embeddings, _, embeddings_indices, \
        val_labels, val_prediction, model, loss, optimizer = vertical_server_model_train

    optimizer_calls = [call.zero_grad(), call.step()]
    model_calls = [call(embeddings)]

    optimizer.assert_has_calls(optimizer_calls)
    model.assert_has_calls(model_calls)

    loss.assert_has_calls([call().backward()])
    # Loss calls: need to check each array argument separately
    np.testing.assert_array_equal(loss.call_args[0][0],
                                  val_prediction)
    np.testing.assert_array_equal(loss.call_args[0][1],
                                  val_labels[embeddings_indices])


def test_vertical_server_model_get_meta_params(vertical_server_model_train):
    """Checks that the vertical Server model gets the Meta parameters correctly."""
    wrapped_model, _, val_grad_embeddings, \
        embeddings_indices, _, _, _, _, _ = vertical_server_model_train

    meta_params = wrapped_model.get_meta_params()

    np.testing.assert_array_equal(meta_params[0], val_grad_embeddings)
    np.testing.assert_array_equal(meta_params[1], embeddings_indices)
