from unittest.mock import Mock, patch, call
import pytest
import numpy as np

from shfl.model.deep_learning_model_pt import DeepLearningModelPyTorch


@pytest.fixture(name="layers_shapes")
def fixture_layers_shapes():
    """Returns the shapes of the layers of the test neural network model."""
    num_data = 5
    shapes = [(num_data, 1, 24, 64),
              (64, 128, 64),
              (num_data, 10)]

    return shapes


@pytest.fixture(name="layers_weights")
def fixture_layers_weights(layers_shapes):
    """Returns random weights for the neural network."""
    array_params = [np.random.rand(*shapes) for shapes in layers_shapes]
    mock_params = []
    for layer in array_params:
        weights_values = Mock()
        weights_values.cpu().data.numpy.return_value = layer
        mock_params.append(weights_values)

    return array_params, mock_params


@pytest.fixture(name="input_dataset")
def fixture_input_dataset(layers_shapes):
    """Returns a random data base for training.

    Both the iterable data loader and the arrays are returned."""
    data = np.random.rand(*layers_shapes[0])
    labels = np.zeros(shape=layers_shapes[-1])
    for label_value in labels:
        label_value[np.random.randint(0, len(label_value))] = 1

    iterable = []
    for data_item, label_item in zip(data, labels):
        features = Mock()
        features.float().to.return_value = data_item[np.newaxis]
        target = Mock()
        target.float().to.return_value = label_item[np.newaxis]

        iterable.append([features, target])

    return iterable, data, labels[np.newaxis]


@pytest.fixture(name="wrapper_arguments")
def fixture_wrapper_arguments(layers_weights):
    """Returns the component necessary for wrapping a deep learning model."""
    model = Mock()
    model.parameters.return_value = layers_weights[1]
    loss = Mock()
    optimizer = Mock()
    batch = 32
    epoch = 2
    metrics = [0, 1, 2, 3]
    device = "cpu"

    return model, loss, optimizer, batch, epoch, metrics, device


def side_effect_from_numpy(value):
    """Returns the side effect of pytorch function from numpy."""
    numpy_array = Mock()
    numpy_array.float.return_value = value

    return numpy_array


def side_effect_argmax(value, _):
    """Returns the side effect of argmax function."""
    return value


def test_initialization(wrapper_arguments):
    """Checks that the pytorch deep learning model is correctly initialized."""
    wrapped_model = DeepLearningModelPyTorch(*wrapper_arguments)

    assert hasattr(wrapped_model, "_model")
    assert hasattr(wrapped_model, "_in_out_sizes")
    assert hasattr(wrapped_model, "_loss")
    assert hasattr(wrapped_model, "_optimizer")
    assert hasattr(wrapped_model, "_batch_size")
    assert hasattr(wrapped_model, "_epochs")
    assert hasattr(wrapped_model, "_metrics")
    assert hasattr(wrapped_model, "_device")


@patch('shfl.model.deep_learning_model_pt.torch')
@patch('shfl.model.deep_learning_model_pt.DataLoader')
def test_train(mock_data_loader, mock_torch,
                             wrapper_arguments, input_dataset):
    """Checks that the pytorch deep learning model trains correctly."""
    model, loss, optimizer = wrapper_arguments[:3]
    model.return_value = [1, 2, 3, 4, 5]
    wrapped_model = DeepLearningModelPyTorch(model, *wrapper_arguments[1:])
    mock_data_loader.return_value = input_dataset[0]

    wrapped_model.train(data=input_dataset[1], labels=input_dataset[2])

    optimizer_calls = []
    model_calls = []
    loss_calls = []
    for _ in range(2):
        for item in input_dataset[0]:
            optimizer_calls.extend([call.zero_grad(), call.step()])
            model_calls.extend([call(item[0].float().to()), call.zero_grad()])
            loss_calls.extend([call([1, 2, 3, 4, 5],
                                    mock_torch.argmax(item[1].float().to(), -1)),
                               call().backward()])

    optimizer.assert_has_calls(optimizer_calls)
    model.assert_has_calls(model_calls)
    loss.assert_has_calls(loss_calls)


@patch('shfl.model.deep_learning_model_pt.DataLoader')
def test_predict(mock_data_loader, wrapper_arguments, input_dataset):
    """Checks that the pytorch deep learning model predicts correctly."""
    model = wrapper_arguments[0]
    model_return = Mock()
    model_return.cpu().numpy.return_value = [1, 2, 3, 4]
    model.return_value = model_return
    wrapped_model = DeepLearningModelPyTorch(model, *wrapper_arguments[1:])
    mock_data_loader.return_value = input_dataset[0]

    prediction = wrapped_model.predict(input_dataset[1])

    model_calls = []
    output = []
    for elem in input_dataset[0]:
        inputs = elem[0].float().to()
        model_calls.extend([call(inputs),
                            call(inputs).cpu(),
                            call(inputs).cpu().numpy()])
        output.extend(model_return.cpu().numpy.return_value)

    model.assert_has_calls(model_calls)
    assert np.array_equal(output, prediction)


@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.predict')
@patch('shfl.model.deep_learning_model_pt.torch')
def test_evaluate(mock_torch, mock_predict,
                  wrapper_arguments, layers_shapes, input_dataset):
    """Checks that the pytorch deep learning model evaluates correctly."""
    loss = wrapper_arguments[1]
    loss.return_value = np.float64(0.0)
    mock_torch.argmax.side_effect = side_effect_argmax
    mock_torch.from_numpy.side_effect = side_effect_from_numpy
    predict_return = Mock()
    predict_return.cpu().numpy.return_value = np.random.rand(*layers_shapes[-1])
    mock_predict.return_value = predict_return
    wrapped_model = DeepLearningModelPyTorch(model=wrapper_arguments[0],
                                             loss=loss,
                                             optimizer=wrapper_arguments[2],
                                             batch_size=wrapper_arguments[3],
                                             epochs=wrapper_arguments[4],
                                             metrics={'aux': lambda x, y: -1},
                                             device=wrapper_arguments[6])

    output_metrics = wrapped_model.evaluate(*input_dataset[1:3])

    mock_predict.assert_called_once_with(input_dataset[1])
    loss.assert_called_once_with(mock_predict.return_value, input_dataset[2])
    assert np.array_equal([0, -1], output_metrics)


@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.evaluate')
def test_performance(mock_evaluate,
                     wrapper_arguments, input_dataset):
    """Checks that the pytorch deep learning model calls the performance correctly."""
    mock_evaluate.return_value = [0, 1, 2, 3, 4]
    wrapped_model = DeepLearningModelPyTorch(*wrapper_arguments)

    output_performance = wrapped_model.performance(*input_dataset[1:3])

    mock_evaluate.assert_called_once_with(*input_dataset[1:3])
    assert output_performance == mock_evaluate.return_value[0]


def test_get_model_params(wrapper_arguments, layers_weights):
    """Checks that the pytorch deep learning model gets the parameters correctly."""
    model = wrapper_arguments[0]
    wrapped_model = DeepLearningModelPyTorch(*wrapper_arguments)

    output_params = wrapped_model.get_model_params()

    # two calls in constructor and one call in get_model_params method
    model.parameters.assert_has_calls([call() for _ in range(3)])
    for true_values, output_values in zip(layers_weights[0], output_params):
        assert np.array_equal(true_values, output_values)


@patch('shfl.model.deep_learning_model_pt.torch')
@patch('shfl.model.deep_learning_model_pt.DeepLearningModelPyTorch.get_model_params')
def test_set_weights(mock_get_params, mock_torch,
                     wrapper_arguments, layers_shapes, layers_weights):
    """Checks that the pytorch deep learning model sets the parameters correctly."""
    model = wrapper_arguments[0]
    mock_get_params.return_value = [np.random.rand(*layers_shapes[0]),
                                    np.random.rand(layers_shapes[-1][-1])]
    old_model_params = [np.random.rand(*layer.shape) for layer in layers_weights[0]]
    mock_params = []
    for weights_values in old_model_params:
        layer_weights = Mock()
        layer_weights.data = weights_values
        mock_params.append(layer_weights)
    model.parameters.return_value = mock_params
    mock_torch.from_numpy.side_effect = side_effect_from_numpy
    wrapped_model = DeepLearningModelPyTorch(model, *wrapper_arguments[1:])

    wrapped_model.set_model_params(layers_weights[0])

    assigned_params = [layer.data for layer in model.parameters()]
    for true_values, assigned_values in zip(layers_weights[0], assigned_params):
        assert np.array_equal(true_values, assigned_values)


def test_wrong_data_input(wrapper_arguments, input_dataset, helpers):
    """Checks that the pytorch deep learning model raises an error if wrong shape
    input data is used."""
    wrapped_model = DeepLearningModelPyTorch(*wrapper_arguments)
    _, data, labels = input_dataset

    helpers.check_wrong_data(wrapped_model, data, labels)


def test_wrong_label_input(wrapper_arguments, input_dataset):
    """Checks that the pytorch deep learning model raises an error if wrong shape
    input label is used."""
    wrapped_model = DeepLearningModelPyTorch(*wrapper_arguments)
    _, data, labels = input_dataset
    wrong_labels_shape = np.array(labels.shape)
    wrong_labels_shape[-1] += 1
    wrong_labels = np.zeros(shape=wrong_labels_shape)
    for label_value in labels:
        label_value[np.random.randint(0, len(label_value))] = 1

    with pytest.raises(AssertionError):
        wrapped_model.train(data, wrong_labels)
