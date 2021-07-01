from unittest.mock import Mock, patch
import pytest
import numpy as np

from shfl.model.linear_regression_model import LinearRegressionModel


@pytest.fixture(name="wrapper_arguments")
def fixture_wrapper_arguments():
    """Returns the component necessary for wrapping a k-means clustering model."""
    n_features = 9
    n_targets = 1

    return n_features, n_targets


@pytest.fixture(name="input_data")
def fixture_input_data(wrapper_arguments):
    """Returns a random data set with targets."""
    n_features, n_targets = wrapper_arguments
    num_data = 50
    data = np.random.rand(num_data, n_features)
    labels = np.random.rand(num_data, n_targets)

    return data, labels


@pytest.mark.parametrize("n_targets", [1, 2, 3])
@patch('shfl.model.linear_regression_model.LinearRegression')
def test_initialization(mock_regression, n_targets, wrapper_arguments):
    """Checks that the linear regression model initializes correctly.

    The case of single and multiple targets are tested."""
    model = Mock()
    mock_regression.return_value = model
    LinearRegressionModel(n_features=wrapper_arguments[0], n_targets=n_targets)

    assert model.intercept_.shape == (n_targets,)
    assert model.coef_.shape == (n_targets, wrapper_arguments[0])


@pytest.mark.parametrize("n_features, n_targets", [(9.5, 1),
                                                   (-1, 1),
                                                   (0, 1),
                                                   (1, 0),
                                                   (9, -1),
                                                   (9, 1.5)])
@patch('shfl.model.linear_regression_model.LinearRegression')
def test_wrong_initialization(mock_regression, n_features, n_targets):
    """Checks that the linear regression model throws an error when initialized
    with wrong inputs.

    Namely, the number of features and targets must be integer
    and greater or equal to one."""
    mock_regression.return_value = Mock()
    with pytest.raises(AssertionError):
        LinearRegressionModel(n_features, n_targets)


@patch('shfl.model.linear_regression_model.LinearRegression')
def test_train(mock_regression, wrapper_arguments, input_data):
    """Checks that the linear regression model trains properly."""
    model = Mock()
    mock_regression.return_value = model
    wrapped_model = LinearRegressionModel(*wrapper_arguments)

    wrapped_model.train(*input_data)

    model.fit.assert_called_once_with(*input_data)


@patch('shfl.model.linear_regression_model.LinearRegression')
def test_train_wrong_input_data_shape(mock_regression, wrapper_arguments, input_data):
    """Checks that the linear regression model throws an error if a wrong shape
    input data are used."""
    n_features, n_targets = wrapper_arguments
    data, labels = input_data
    mock_regression.return_value = Mock()
    wrapped_model = LinearRegressionModel(n_features=n_features, n_targets=n_targets)

    wrong_data_single_feature = np.random.rand(len(data))
    with pytest.raises(AssertionError):
        wrapped_model.train(wrong_data_single_feature, labels)

    wrong_data_multiple_features = np.random.rand(len(data), n_features + 1)
    with pytest.raises(AssertionError):
        wrapped_model.train(wrong_data_multiple_features, labels)


@patch('shfl.model.linear_regression_model.LinearRegression')
def test_train_wrong_labels_shape(mock_regression, wrapper_arguments, input_data):
    """Checks that the linear regression model throws an error if a wrong shape
    input labels are used."""
    n_features, _ = wrapper_arguments
    n_targets = 2
    data, _ = input_data
    mock_regression.return_value = Mock()
    wrapped_model = LinearRegressionModel(n_features=n_features, n_targets=n_targets)

    wrong_label_single_target = np.random.rand(len(data))
    with pytest.raises(AssertionError):
        wrapped_model.train(data, wrong_label_single_target)

    wrong_label_multiple_targets = np.random.rand(len(data), n_targets + 1)
    with pytest.raises(AssertionError):
        wrapped_model.train(data, wrong_label_multiple_targets)


@patch('shfl.model.linear_regression_model.LinearRegression')
def test_predict(mock_classifier, wrapper_arguments, input_data):
    """Checks that the linear regression model predicts correctly."""
    data, _ = input_data
    model = Mock()
    true_prediction = np.random.rand(len(data), wrapper_arguments[1])
    model.predict.return_value = true_prediction
    mock_classifier.return_value = model
    wrapped_model = LinearRegressionModel(*wrapper_arguments)

    output_prediction = wrapped_model.predict(data)

    model.predict.assert_called_once_with(data)
    np.testing.assert_array_equal(output_prediction, true_prediction)


@patch('shfl.model.linear_regression_model.metrics')
@patch('shfl.model.linear_regression_model.LinearRegression')
def test_evaluate(mock_classifier, mock_metrics, wrapper_arguments, input_data):
    """Checks that the linear regression model predicts correctly."""
    data, _ = input_data
    mock_metrics.mean_squared_error.return_value = 0.8
    mock_metrics.r2_score.return_value = 0.7
    mock_classifier.return_value = Mock()
    wrapped_model = LinearRegressionModel(*wrapper_arguments)
    wrapped_model.predict = Mock()
    true_prediction = np.random.rand(len(data), wrapper_arguments[1])
    wrapped_model.predict.return_value = true_prediction

    root_mean_squared_error, r2_score = wrapped_model.evaluate(*input_data)

    wrapped_model.predict.assert_called_once_with(data)
    mock_metrics.mean_squared_error.assert_called_once_with(input_data[1], true_prediction)
    mock_metrics.r2_score.assert_called_once_with(input_data[1], true_prediction)
    assert root_mean_squared_error == np.sqrt(0.8)
    assert r2_score == 0.7


@patch('shfl.model.linear_regression_model.metrics')
@patch('shfl.model.linear_regression_model.LinearRegression')
def test_performance(mock_classifier, mock_metrics, wrapper_arguments, input_data):
    """Checks that the linear regression model calls performance correctly."""
    data, _ = input_data
    mock_metrics.mean_squared_error.return_value = 0.8
    mock_classifier.return_value = Mock()
    wrapped_model = LinearRegressionModel(*wrapper_arguments)
    wrapped_model.predict = Mock()
    true_prediction = np.random.rand(len(data), wrapper_arguments[1])
    wrapped_model.predict.return_value = true_prediction

    negative_root_mean_squared_error = wrapped_model.performance(*input_data)

    wrapped_model.predict.assert_called_once_with(data)
    mock_metrics.mean_squared_error.assert_called_once_with(input_data[1], true_prediction)
    assert negative_root_mean_squared_error == -np.sqrt(0.8)


@patch('shfl.model.linear_regression_model.LinearRegression')
def test_get_model_params(mock_classifier, wrapper_arguments):
    """Checks that the linear regression gets the model's parameters correctly."""
    model = Mock()
    mock_classifier.return_value = model
    wrapped_model = LinearRegressionModel(*wrapper_arguments)

    output_params = wrapped_model.get_model_params()

    np.testing.assert_array_equal(model.intercept_, output_params[0])
    np.testing.assert_array_equal(model.coef_, output_params[1])


@patch('shfl.model.linear_regression_model.LinearRegression')
def test_set_model_params(mock_classifier, wrapper_arguments):
    """Checks that the linear regression sets the model's parameters correctly."""
    n_features, n_targets = wrapper_arguments
    model = Mock()
    mock_classifier.return_value = model
    wrapped_model = LinearRegressionModel(n_features, n_targets)
    input_params = (np.random.rand(n_targets),
                    np.random.rand(n_targets, n_features))

    wrapped_model.set_model_params(input_params)

    np.testing.assert_array_equal(model.intercept_, input_params[0])
    np.testing.assert_array_equal(model.coef_, input_params[1])
