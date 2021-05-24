from unittest.mock import Mock, patch
import numpy as np
from shfl.model.kmeans_model import KMeansModel


@patch('shfl.model.kmeans_model.KMeans')
def test_kmeans_model(mock_kmeans):
    model = Mock()
    mock_kmeans.return_value = model

    n_clusters = 5
    n_features = 5
    init = 'k_highest-means++'
    n_init = 10
    wrapped_model = KMeansModel(n_clusters, n_features, init, n_init)

    mock_kmeans.assert_called_once_with(n_clusters=n_clusters, init=init, n_init=n_init)
    assert model == wrapped_model._model
    assert init == wrapped_model._init
    assert n_features == wrapped_model._n_features
    assert n_init == wrapped_model._n_init
    assert np.array_equal(np.zeros((n_clusters, n_features)), wrapped_model._model.cluster_centers_)


@patch('shfl.model.kmeans_model.KMeans')
def test_kmeans_model_ndarray(mock_kmeans):
    model = Mock()
    mock_kmeans.return_value = model

    n_clusters = 5
    n_features = 5
    init = np.array([0, 0])
    n_init = 10
    wrapped_model = KMeansModel(n_clusters, n_features, init, n_init)

    mock_kmeans.assert_called_once_with(n_clusters=n_clusters, init=init, n_init=n_init)
    assert model == wrapped_model._model
    assert np.array_equal(init, wrapped_model._init)
    assert n_init == wrapped_model._n_init
    assert n_features == wrapped_model._n_features
    assert np.array_equal(init, wrapped_model._model.cluster_centers_)


@patch('shfl.model.kmeans_model.KMeans')
def test_train(mock_kmeans):
    model = Mock()
    mock_kmeans.return_value = model

    n_clusters = 5
    n_features = 5
    init = 'k_highest-means++'
    n_init = 10
    wrapped_model = KMeansModel(n_clusters, n_features, init, n_init)

    data = np.random.rand(10)
    wrapped_model.train(data)

    model.fit.assert_called_once_with(data)


@patch('shfl.model.kmeans_model.KMeans')
def test_predict(mock_kmeans):
    model = Mock()
    y_pred_real = np.random.rand(10)
    model.predict.return_value = y_pred_real
    mock_kmeans.return_value = model

    n_clusters = 5
    n_features = 5
    init = 'k_highest-means++'
    n_init = 10
    wrapped_model = KMeansModel(n_clusters, n_features, init, n_init)

    data = np.random.rand(10)
    y_pred = wrapped_model.predict(data)

    model.predict.assert_called_once_with(data)
    assert np.array_equal(y_pred_real, y_pred)


@patch('shfl.model.kmeans_model.KMeans')
@patch('shfl.model.kmeans_model.metrics.homogeneity_score')
@patch('shfl.model.kmeans_model.metrics.completeness_score')
@patch('shfl.model.kmeans_model.metrics.v_measure_score')
@patch('shfl.model.kmeans_model.metrics.adjusted_rand_score')
def test_evaluate(mock_adjusted_rand_score,
                  mock_v_measure_score,
                  mock_completeness_score,
                  mock_homogeneity_score, mock_kmeans):

    mock_adjusted_rand_score.return_value = 3
    mock_v_measure_score.return_value = 2
    mock_completeness_score.return_value = 1
    mock_homogeneity_score.return_value = 0

    model = Mock()
    y_pred_real = np.random.rand(10)
    model.predict.return_value = y_pred_real
    mock_kmeans.return_value = model

    n_clusters = 5
    n_features = 5
    init = 'k_highest-means++'
    n_init = 10
    wrapped_model = KMeansModel(n_clusters, n_features, init, n_init)

    wrapped_model.predict = Mock()
    y_pred = np.random.randint(0, 2, 10)
    wrapped_model.predict.return_value = y_pred

    data = np.random.rand(10)
    labels = np.random.randint(0, 2, 10)
    homo, compl, v_meas, rai = wrapped_model.evaluate(data, labels)

    wrapped_model.predict.assert_called_once_with(data)
    mock_homogeneity_score.assert_called_once_with(labels, y_pred)
    mock_completeness_score.assert_called_once_with(labels, y_pred)
    mock_v_measure_score.assert_called_once_with(labels, y_pred)
    mock_adjusted_rand_score.assert_called_once_with(labels, y_pred)
    assert homo == 0
    assert compl == 1
    assert v_meas == 2
    assert rai == 3


@patch('shfl.model.kmeans_model.KMeans')
def test_get_model_params(mock_kmeans):
    model = Mock()
    mock_kmeans.return_value = model

    n_clusters = 5
    n_features = 5
    init = 'k_highest-means++'
    n_init = 10
    wrapped_model = KMeansModel(n_clusters, n_features, init, n_init)

    model_params = wrapped_model.get_model_params()

    assert np.array_equal(wrapped_model._model.cluster_centers_, model_params)


@patch('shfl.model.kmeans_model.KMeans')
def test_set_model_params(mock_kmeans):
    model = Mock()
    mock_kmeans.return_value = model

    n_clusters = 5
    n_features = 5
    init = 'k_highest-means++'
    n_init = 1  # Explicit initial center position passed: performing only one init
    wrapped_model = KMeansModel(n_clusters, n_features, init, n_init)

    params = np.random.rand(10).reshape((5, 2))

    wrapped_model.set_model_params(params)

    assert model == wrapped_model._model
    assert np.array_equal(params, wrapped_model._init)
    assert n_init == wrapped_model._n_init
    assert n_features == wrapped_model._n_features
    assert np.array_equal(params, wrapped_model._model.cluster_centers_)


@patch('shfl.model.kmeans_model.KMeans')
def test_set_model_params_zeros_array(mock_kmeans):
    model = Mock()
    mock_kmeans.return_value = model

    n_clusters = 5
    n_features = 5
    init = 'k_highest-means++'
    n_init = 10
    wrapped_model = KMeansModel(n_clusters, n_features, init, n_init)

    params = np.zeros(10).reshape((5, 2))

    wrapped_model.set_model_params(params)

    assert model == wrapped_model._model
    assert init == wrapped_model._init
    assert n_init == wrapped_model._n_init
    assert n_features == wrapped_model._n_features
    assert np.array_equal(np.zeros((params.shape[0], n_features)), wrapped_model._model.cluster_centers_)


@patch('shfl.model.kmeans_model.metrics.v_measure_score')
@patch('shfl.model.kmeans_model.KMeans')
def test_performance(mock_kmeans, mock_v_measure_score):
    model = Mock()
    mock_kmeans.return_value = model
    mock_v_measure_score.return_value = 0

    n_clusters = 5
    n_features = 5
    init = 'k_highest-means++'
    n_init = 10
    wrapped_model = KMeansModel(n_clusters, n_features, init, n_init)

    data = np.random.rand(25).reshape((5, 5))
    labels = np.random.randint(0, 2, 5)
    wrapped_model.predict = Mock()
    prediction = ~labels
    wrapped_model.predict.return_value = prediction

    res = wrapped_model.performance(data, labels)

    assert res == 0
    mock_v_measure_score.assert_called_once_with(labels, prediction)
    wrapped_model.predict.assert_called_once_with(data)
