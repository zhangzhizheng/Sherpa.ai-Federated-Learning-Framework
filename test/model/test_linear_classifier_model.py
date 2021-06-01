from unittest.mock import Mock, patch
import pytest
import numpy as np

from shfl.model.linear_classifier_model import LinearClassifierModel


@pytest.fixture(name="wrapper_arguments")
def fixture_wrapper_arguments():
    """Returns the component necessary for wrapping a k-means clustering model."""
    n_features = 9
    classes = ["a", "b", "c"]

    return n_features, classes


@pytest.fixture(name="input_data")
def fixture_input_data(wrapper_arguments):
    """Returns a random labeled dataset."""
    n_features, classes = wrapper_arguments
    num_data = 50
    data = np.random.rand(num_data, n_features)
    labels = np.random.choice(classes, size=num_data)

    return data, labels


@pytest.mark.parametrize("classes, n_classes", [(["a", "b"], 1),
                                                (["a", "b", "c"], 3)])
@patch('shfl.model.linear_classifier_model.LogisticRegression')
def test_initialization_binary_classes(mock_classifier, classes, n_classes,
                                       wrapper_arguments):
    """Checks that the linear classifier correctly initializes.

    For a binary classification, the number of classes
    "n_classes" is equal to one. Instead, for a multi-class case,
    the number of classes "n_classes" is equal
    to the actual number of classes."""
    model = Mock()
    mock_classifier.return_value = model

    wrapped_model = LinearClassifierModel(n_features=wrapper_arguments[0],
                                          classes=classes)

    assert hasattr(wrapped_model, "_model")
    assert hasattr(wrapped_model, "_n_features")
    assert model.intercept_.shape[0] == n_classes
    assert model.coef_.shape == (n_classes, wrapper_arguments[0])
    assert np.array_equal(classes, model.classes_)


@pytest.mark.parametrize("n_features, classes", [(9.5, ['a', 'b', 'c']),
                                                 (-1, ['a', 'b', 'c']),
                                                 (9, ['b']),
                                                 (9, ['a', 'b', 'a'])])
@patch('shfl.model.linear_classifier_model.LogisticRegression')
def test_model_wrong_initialization(mock_classifier, n_features, classes):
    """Checks that the linear classification model throws an error if
    not initialized correctly.

    Namely, the number of features must be: integer, non-negative.
    The classes must be: more than one, not repeating."""
    mock_classifier.return_value = Mock()

    with pytest.raises(AssertionError):
        LinearClassifierModel(n_features, classes)


@patch('shfl.model.linear_classifier_model.LogisticRegression')
def test_train(mock_classifier, wrapper_arguments, input_data):
    """Checks that the linear classifier model trains correctly."""
    model = Mock()
    mock_classifier.return_value = model
    wrapped_model = LinearClassifierModel(*wrapper_arguments)

    wrapped_model.train(*input_data)

    model.fit.assert_called_once_with(*input_data)


@patch('shfl.model.linear_classifier_model.LogisticRegression')
def test_train_wrong_data(mock_classifier, wrapper_arguments, input_data, helpers):
    """Checks that the linear classifier model throws an error if wrong
    data are used as input."""
    mock_classifier.return_value = Mock()
    wrapped_model = LinearClassifierModel(*wrapper_arguments)

    helpers.check_wrong_data(wrapped_model, *input_data)


@patch('shfl.model.linear_classifier_model.LogisticRegression')
def test_train_wrong_data_single_feature(mock_classifier, wrapper_arguments, input_data):
    """Checks that the linear classifier model throws an error if wrong
    data are used as input.

    If data contains only one column, then the number of features must be 1."""
    mock_classifier.return_value = Mock()
    wrapped_model = LinearClassifierModel(*wrapper_arguments)
    data, labels = input_data
    wrong_data = np.random.rand(len(data))

    with pytest.raises(AssertionError):
        wrapped_model.train(wrong_data, labels)


@patch('shfl.model.linear_classifier_model.LogisticRegression')
def test_train_wrong_labels(mock_classifier, wrapper_arguments, input_data):
    """Checks that the linear classifier model throws an error if wrong
    labels are used as input."""
    mock_classifier.return_value = Mock()
    wrapped_model = LinearClassifierModel(*wrapper_arguments)
    data, labels = input_data
    wrong_labels = labels
    wrong_labels[0] = "not_initialized_class"

    with pytest.raises(AssertionError):
        wrapped_model.train(data, wrong_labels)


@patch('shfl.model.linear_classifier_model.LogisticRegression')
def test_predict(mock_classifier, wrapper_arguments, input_data):
    """Checks that the linear classifier model predicts correctly."""
    data, labels = input_data
    model = Mock()
    true_prediction = np.random.choice(labels, size=len(data))
    model.predict.return_value = true_prediction
    mock_classifier.return_value = model
    wrapped_model = LinearClassifierModel(*wrapper_arguments)

    output_prediction = wrapped_model.predict(data)

    model.predict.assert_called_once_with(data)
    np.testing.assert_array_equal(output_prediction, true_prediction)


@patch('shfl.model.linear_classifier_model.metrics')
@patch('shfl.model.linear_classifier_model.LogisticRegression')
def test_evaluate(mock_classifier, mock_metrics, wrapper_arguments, input_data):
    """Checks that the linear classifier model evaluates correctly."""
    data, labels = input_data
    model = Mock()
    mock_classifier.return_value = model
    wrapped_model = LinearClassifierModel(*wrapper_arguments)
    wrapped_model.predict = Mock()
    true_prediction = np.random.choice(labels, size=len(data))
    wrapped_model.predict.return_value = true_prediction
    mock_metrics.balanced_accuracy_score.return_value = 0.5
    mock_metrics.cohen_kappa_score.return_value = 0.7

    balanced_accuracy_score, cohen_kappa_score = wrapped_model.evaluate(data, labels)

    wrapped_model.predict.assert_called_once_with(data)
    mock_metrics.balanced_accuracy_score.assert_called_once_with(labels, true_prediction)
    mock_metrics.cohen_kappa_score.assert_called_once_with(labels, true_prediction)
    assert balanced_accuracy_score == 0.5
    assert cohen_kappa_score == 0.7


@patch('shfl.model.linear_classifier_model.LogisticRegression')
def test_evaluate_wrong_labels(mock_classifier, wrapper_arguments, input_data):
    """Checks that the linear classifier model throws an error if wrong
    labels are used as input."""
    mock_classifier.return_value = Mock()
    wrapped_model = LinearClassifierModel(*wrapper_arguments)
    data, labels = input_data
    wrong_labels = labels
    wrong_labels[0] = "not_initialized_class"

    with pytest.raises(AssertionError):
        wrapped_model.evaluate(data, wrong_labels)


@patch('shfl.model.linear_classifier_model.metrics')
@patch('shfl.model.linear_classifier_model.LogisticRegression')
def test_performance(mock_classifier, mock_metrics, wrapper_arguments, input_data):
    """Checks that the linear classifier model calls performance correctly."""
    data, labels = input_data
    model = Mock()
    mock_classifier.return_value = model
    wrapped_model = LinearClassifierModel(*wrapper_arguments)
    wrapped_model.predict = Mock()
    true_prediction = np.random.choice(labels, size=len(data))
    wrapped_model.predict.return_value = true_prediction
    mock_metrics.balanced_accuracy_score.return_value = 0.5

    balanced_accuracy_score = wrapped_model.performance(data, labels)

    wrapped_model.predict.assert_called_once_with(data)
    assert balanced_accuracy_score == 0.5


@patch('shfl.model.linear_classifier_model.LogisticRegression')
def test_get_model_params(mock_classifier, wrapper_arguments):
    """Checks that the linear classifier gets the model's parameters correctly."""
    model = Mock()
    mock_classifier.return_value = model
    wrapped_model = LinearClassifierModel(*wrapper_arguments)

    output_params = wrapped_model.get_model_params()

    np.testing.assert_array_equal(model.intercept_, output_params[0])
    np.testing.assert_array_equal(model.coef_, output_params[1])


@patch('shfl.model.linear_classifier_model.LogisticRegression')
def test_set_model_params(mock_classifier, wrapper_arguments):
    """Checks that the linear classifier sets the model's parameters correctly."""
    n_features, classes = wrapper_arguments
    model = Mock()
    mock_classifier.return_value = model
    wrapped_model = LinearClassifierModel(n_features, classes)
    n_classes = len(classes)
    input_params = (np.random.rand(n_classes),
                    np.random.rand(n_classes, n_features))

    wrapped_model.set_model_params(input_params)

    np.testing.assert_array_equal(model.intercept_, input_params[0])
    np.testing.assert_array_equal(model.coef_, input_params[1])
