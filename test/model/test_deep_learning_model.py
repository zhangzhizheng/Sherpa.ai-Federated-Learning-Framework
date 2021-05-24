from unittest.mock import Mock
import pytest
import numpy as np

from shfl.model.deep_learning_model import DeepLearningModel


@pytest.fixture(name="layers_shapes")
def fixture_layers_shapes():
    """Returns the shapes of the layers of the test neural network model."""
    shapes = [(30, 64, 64), (64, 10)]

    return shapes


@pytest.fixture(name="data_and_labels")
def fixture_data_and_labels(layers_shapes):
    """Returns a random data set with labels."""
    num_data = 40
    input_shape = layers_shapes[0]
    output_shape = layers_shapes[-1]
    data = np.random.rand(num_data, *input_shape[1:])
    labels = np.zeros(shape=(num_data, output_shape[-1]))
    for label_value in labels:
        label_value[np.random.randint(0, len(label_value))] = 1

    return data, labels


@pytest.fixture(name="wrapper_arguments")
def fixture_wrapper_arguments(layers_shapes):
    """Returns the component necessary for wrapping a deep learning model."""
    model = Mock()
    loss = Mock()
    optimizer = Mock()
    metrics = Mock()

    input_layer = Mock()
    input_layer.get_input_shape_at.return_value = layers_shapes[0]
    output_layer = Mock()
    output_layer.get_output_shape_at.return_value = layers_shapes[1]
    model.layers = [input_layer, output_layer]

    batch = 32
    epoch = 2

    return model, loss, optimizer, batch, epoch, metrics


def test_initialization(wrapper_arguments):
    """Checks that the deep learning model is correctly initialized."""
    model, loss, optimizer, \
        batch, epoch, metrics = wrapper_arguments

    wrapped_model = DeepLearningModel(model, loss, optimizer,
                                          batch, epoch, metrics)

    model.compile.assert_called_once_with(optimizer=optimizer,
                                          loss=loss,
                                          metrics=metrics)
    assert hasattr(wrapped_model, "_model")
    assert hasattr(wrapped_model, "_data_shape")
    assert hasattr(wrapped_model, "_labels_shape")
    assert hasattr(wrapped_model, "_batch_size")
    assert hasattr(wrapped_model, "_epochs")
    assert hasattr(wrapped_model, "_loss")
    assert hasattr(wrapped_model, "_optimizer")
    assert hasattr(wrapped_model, "_metrics")


def test_train_wrong_data(wrapper_arguments, data_and_labels, layers_shapes):
    """Checks that an error is raised if wrong shaped data are used as input."""

    wrapped_model = DeepLearningModel(*wrapper_arguments)
    _, labels = data_and_labels
    wrong_input_shape = np.array(layers_shapes[0])
    wrong_input_shape[-1] += 1
    wrong_data = np.random.rand(len(labels), *wrong_input_shape[1:])

    with pytest.raises(AssertionError):
        wrapped_model.train(wrong_data, labels)


def test_train_wrong_labels(wrapper_arguments, data_and_labels, layers_shapes):
    """Checks that an error is raised if wrong shaped labels are used as input."""

    wrapped_model = DeepLearningModel(*wrapper_arguments)
    data, _ = data_and_labels
    wrong_output_shape = np.array(layers_shapes[1])
    wrong_output_shape[-1] += 1
    wrong_labels = np.zeros(shape=(len(data), wrong_output_shape[-1]))
    for label_value in wrong_labels:
        label_value[np.random.randint(0, len(label_value))] = 1

    with pytest.raises(AssertionError):
        wrapped_model.train(data, wrong_labels)


def test_train(wrapper_arguments, data_and_labels):
    """Checks that the model trains correctly."""
    model, criterion, optimizer, batch, epoch, metrics = wrapper_arguments
    wrapped_model = DeepLearningModel(model, criterion, optimizer,
                                      batch, epoch, metrics)
    data, labels = data_and_labels

    wrapped_model.train(data, labels)

    model.fit.assert_called_once()
    params = model.fit.call_args_list[0][1]
    assert np.array_equal(params['x'], data)
    assert np.array_equal(params['y'], labels)
    assert params['batch_size'] == batch
    assert params['epochs'] == epoch


def test_evaluate(wrapper_arguments, data_and_labels):
    """Checks that the model evaluates correctly."""
    model, criterion, optimizer, batch, epoch, metrics = wrapper_arguments
    wrapped_model = DeepLearningModel(model, criterion, optimizer,
                                      batch, epoch, metrics)
    data, labels = data_and_labels

    wrapped_model.evaluate(data, labels)

    model.evaluate.assert_called_once_with(data, labels, verbose=0)


def test_predict(wrapper_arguments, data_and_labels):
    """Checks that the model predict correctly."""
    model, criterion, optimizer, batch, epoch, metrics = wrapper_arguments
    wrapped_model = DeepLearningModel(model, criterion, optimizer,
                                      batch, epoch, metrics)
    data, _ = data_and_labels

    wrapped_model.predict(data)

    model.predict.assert_called_once_with(data, batch_size=batch)


def test_predict_wrong_data(wrapper_arguments, data_and_labels, layers_shapes):
    """Checks that an error is raised if wrong shaped data are used as input."""

    wrapped_model = DeepLearningModel(*wrapper_arguments)
    _, labels = data_and_labels
    wrong_input_shape = np.array(layers_shapes[0])
    wrong_input_shape[-1] += 1
    wrong_data = np.random.rand(len(labels), *wrong_input_shape[1:])

    with pytest.raises(AssertionError):
        wrapped_model.predict(wrong_data)


def test_get_model_params(wrapper_arguments):
    """Checks that the models' parameters are correctly returned."""
    model, criterion, optimizer, batch, epoch, metrics = wrapper_arguments
    params = np.random.rand(30, 24, 24)
    model.get_weights.return_value = params
    wrapped_model = DeepLearningModel(model, criterion, optimizer,
                                      batch, epoch, metrics)

    returned_params = wrapped_model.get_model_params()

    assert np.array_equal(params, returned_params)
    model.get_weights.assert_called_once()


def test_set_model_params(wrapper_arguments):
    """Checks that the models' parameters are correctly set."""
    model, criterion, optimizer, batch, epoch, metrics = wrapper_arguments
    params = np.random.rand(30, 24, 24)
    wrapped_model = DeepLearningModel(model, criterion, optimizer,
                                      batch, epoch, metrics)

    wrapped_model.set_model_params(params)

    model.set_weights.assert_called_once_with(params)


def test_performance(wrapper_arguments, data_and_labels):
    """Checks that the model evaluates performance correctly."""
    model, criterion, optimizer, batch, epoch, metrics = wrapper_arguments
    performance = [0.8, 0.9, 0.5]
    model.evaluate.return_value = performance
    wrapped_model = DeepLearningModel(model, criterion, optimizer,
                                      batch, epoch, metrics)
    data, labels = data_and_labels

    returned_performance = wrapped_model.performance(data, labels)

    model.evaluate.assert_called_once_with(data, labels, verbose=0)
    assert returned_performance == performance[0]
