"""
This file is part of Sherpa Federated Learning Framework.

Sherpa Federated Learning Framework is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3.

Sherpa Federated Learning Framework is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
"""

from unittest.mock import Mock, patch
import pytest
import numpy as np

from shfl.model.kmeans_model import KMeansModel


@pytest.fixture(name="wrapper_arguments")
def fixture_wrapper_arguments():
    """Returns the component necessary for wrapping a k-means clustering model."""
    n_clusters = 5
    n_features = 5
    init = "k_highest-means++"
    n_init = 10

    return n_clusters, n_features, init, n_init


@pytest.fixture(name="input_data")
def fixture_input_data(wrapper_arguments):
    """Returns a random data set."""
    n_features = wrapper_arguments[1]
    num_data = 50
    data = np.random.rand(num_data, n_features)

    return data


@pytest.mark.parametrize("init_type", ["k_highest-means++", np.array([0, 0])])
@patch('shfl.model.kmeans_model.KMeans')
def test_initialization(mock_kmeans, init_type, wrapper_arguments):
    """Checks that the k-means model is correctly initialized.

    Both default initialization with random ("k_highest-means++") and
    input centroids (a numpy array) are tested."""
    mock_kmeans.return_value = Mock()
    n_clusters, n_features, _, n_init = wrapper_arguments
    init = init_type
    wrapped_model = KMeansModel(n_clusters, n_features, init, n_init)

    mock_kmeans.assert_called_once_with(n_clusters=n_clusters, init=init, n_init=n_init)
    assert hasattr(wrapped_model, "_model")
    assert hasattr(wrapped_model, "_init")
    assert hasattr(wrapped_model, "_n_features")
    assert hasattr(wrapped_model, "_n_init")


@patch('shfl.model.kmeans_model.KMeans')
def test_train(mock_kmeans, wrapper_arguments, input_data):
    """Checks that the k-means model trains correctly."""
    model = Mock()
    mock_kmeans.return_value = model
    wrapped_model = KMeansModel(*wrapper_arguments)

    wrapped_model.train(input_data)

    model.fit.assert_called_once_with(input_data)


@patch('shfl.model.kmeans_model.KMeans')
def test_predict(mock_kmeans, wrapper_arguments, input_data):
    """Checks that the k-means model predicts correctly."""
    model = Mock()
    true_prediction = np.random.rand(10)
    model.predict.return_value = true_prediction
    mock_kmeans.return_value = model
    wrapped_model = KMeansModel(*wrapper_arguments)

    output_prediction = wrapped_model.predict(input_data)

    model.predict.assert_called_once_with(input_data)
    assert np.array_equal(true_prediction, output_prediction)


@patch('shfl.model.kmeans_model.KMeans')
@patch('shfl.model.kmeans_model.metrics')
def test_evaluate(mock_metrics, mock_kmeans,
                  wrapper_arguments, input_data):
    """Checks that the k-means model evaluates correctly."""
    mock_metrics.adjusted_rand_score.return_value = 3
    mock_metrics.v_measure_score.return_value = 2
    mock_metrics.completeness_score.return_value = 1
    mock_metrics.homogeneity_score.return_value = 0
    model = Mock()
    model.predict.return_value = np.random.rand(10)
    mock_kmeans.return_value = model
    wrapped_model = KMeansModel(*wrapper_arguments)
    wrapped_model.predict = Mock()
    true_prediction = np.random.randint(low=0, high=2, size=len(input_data))
    wrapped_model.predict.return_value = true_prediction

    labels = np.random.randint(low=0, high=2, size=len(input_data))
    homogeneity_score, completeness_score, \
        v_measure_score, adjusted_rand_score = wrapped_model.evaluate(input_data, labels)

    wrapped_model.predict.assert_called_once_with(input_data)
    mock_metrics.homogeneity_score.assert_called_once_with(labels, true_prediction)
    mock_metrics.completeness_score.assert_called_once_with(labels, true_prediction)
    mock_metrics.v_measure_score.assert_called_once_with(labels, true_prediction)
    mock_metrics.adjusted_rand_score.assert_called_once_with(labels, true_prediction)
    assert homogeneity_score == 0
    assert completeness_score == 1
    assert v_measure_score == 2
    assert adjusted_rand_score == 3


@patch('shfl.model.kmeans_model.KMeans')
def test_get_model_params(mock_kmeans, wrapper_arguments):
    """Checks that the k-means model properly gets the model's parameters."""
    model = Mock()
    mock_kmeans.return_value = model
    wrapped_model = KMeansModel(*wrapper_arguments)

    output_params = wrapped_model.get_model_params()

    assert np.array_equal(model.cluster_centers_, output_params)


@pytest.mark.parametrize("input_params", [np.random.rand(5, 5), np.zeros(shape=(5, 5))])
@patch('shfl.model.kmeans_model.KMeans')
def test_set_model_params(mock_kmeans, input_params, wrapper_arguments):
    """Checks that the k-means model properly sets the model's parameters.

    Both initialization (array with zeroes) and new centroids
    (array with random numbers) are tested."""
    model = Mock()
    mock_kmeans.return_value = model
    wrapped_model = KMeansModel(*wrapper_arguments)

    wrapped_model.set_model_params(input_params)

    assert np.array_equal(model.cluster_centers_, input_params)


@patch('shfl.model.kmeans_model.metrics.v_measure_score')
@patch('shfl.model.kmeans_model.KMeans')
def test_performance(mock_kmeans, mock_v_measure_score, wrapper_arguments, input_data):
    """Checks that the k-means model correctly calls the performance."""
    model = Mock()
    mock_kmeans.return_value = model
    mock_v_measure_score.return_value = 0
    wrapped_model = KMeansModel(*wrapper_arguments)
    labels = np.random.randint(low=0, high=2, size=len(input_data))
    wrapped_model.predict = Mock()
    prediction = ~labels
    wrapped_model.predict.return_value = prediction

    output_performance = wrapped_model.performance(input_data, labels)

    assert output_performance == 0
    mock_v_measure_score.assert_called_once_with(labels, prediction)
    wrapped_model.predict.assert_called_once_with(input_data)
