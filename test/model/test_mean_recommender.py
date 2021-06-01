from unittest.mock import Mock, patch
import pytest
import numpy as np

from shfl.model.mean_recommender import MeanRecommender


@pytest.fixture(name="data_and_labels")
def fixture_data_and_labels():
    """Returns random data features and rating labels."""
    data = np.array([[2, 3, 51],
                     [2, 34, 6],
                     [2, 33, 7],
                     [2, 13, 65],
                     [2, 3, 15]])
    labels = np.array([3, 2, 5, 6, 7])

    return data, labels


def test_initialization():
    """Checks that the mean recommender initializes correctly."""
    mean_recommender = MeanRecommender()

    assert hasattr(mean_recommender, "_client_identifier")


@patch('shfl.model.mean_recommender.np')
def test_train(mock_numpy, data_and_labels):
    """Checks that the mean recommender trains correctly."""
    mean_recommender = MeanRecommender()
    mock_numpy.mean = Mock()
    mock_numpy.mean.return_value = 5

    mean_recommender.train(*data_and_labels)

    mock_numpy.mean.assert_called_once_with(data_and_labels[1])


def test_train_wrong_data(data_and_labels):
    """Checks that the mean recommender raises an error if wrong data are used as input.

    In this case, the data contains one rating from a different user."""
    data, labels = data_and_labels
    wrong_data = data
    wrong_data[0, 0] += 1
    mean_recommender = MeanRecommender()

    with pytest.raises(AssertionError):
        mean_recommender.train(wrong_data, labels)


def test_train_wrong_data_labels_lengths(data_and_labels):
    """Checks that the mean recommender raises an error if input data and labels
    have different lengths."""
    data, labels = data_and_labels
    wrong_labels = labels[:-1]
    mean_recommender = MeanRecommender()

    with pytest.raises(AssertionError):
        mean_recommender.train(data, wrong_labels)


@patch('shfl.model.mean_recommender.np')
def test_predict(mock_numpy, data_and_labels):
    """Checks that the mean recommender predicts correctly."""
    mean_recommender = MeanRecommender()
    true_prediction = np.ones(shape=data_and_labels[1].shape) * 2.5
    mock_numpy.full = Mock()
    mock_numpy.full.return_value = true_prediction

    output_predictions = mean_recommender.predict(data_and_labels[0])

    np.testing.assert_array_equal(output_predictions, true_prediction)
    mock_numpy.full.assert_called_once()


def test_evaluate(data_and_labels):
    """Checks that the mean recommender evaluates correctly."""
    mean_recommender = MeanRecommender()
    mean_recommender.predict = Mock()
    true_prediction = np.ones(shape=data_and_labels[1].shape) * 2.5
    mean_recommender.predict.return_value = true_prediction

    root_mean_square = mean_recommender.evaluate(*data_and_labels)

    assert root_mean_square == np.sqrt(np.mean(
        (true_prediction - data_and_labels[1]) ** 2))


def test_evaluate_no_data():
    """Checks that the mean recommender evaluates correctly when no data is available."""
    data = np.empty((0, 3))
    labels = np.empty(0)

    mean_recommender = MeanRecommender()
    mean_recommender.predict = Mock()
    true_prediction = np.ones(shape=labels.shape) * 2.5
    mean_recommender.predict.return_value = true_prediction

    root_mean_square = mean_recommender.evaluate(data, labels)

    assert root_mean_square == 0


def test_performance(data_and_labels):
    """Checks that the mean recommender correctly calls the performance."""
    mean_recommender = MeanRecommender()
    mean_recommender.evaluate = Mock()
    mean_recommender.evaluate.return_value = 0.7

    root_mean_square = mean_recommender.performance(*data_and_labels)

    mean_recommender.evaluate.assert_called_once_with(*data_and_labels)
    assert root_mean_square == 0.7


def test_set_model_params():
    """Checks that the mean recommender model correctly sets the model's parameters."""
    mean_recommender = MeanRecommender()

    mean_recommender.set_model_params(3.4)

    assert mean_recommender.get_model_params() == 3.4


def test_get_model_params():
    """Checks that the mean recommender model correctly gets the model's parameters."""
    mean_recommender = MeanRecommender()

    params = mean_recommender.get_model_params()

    assert params is None
