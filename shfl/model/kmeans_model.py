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

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

from shfl.model.model import TrainableModel


class KMeansModel(TrainableModel):
    """Wraps the scikit-learn K-means model.

    Implements the class [TrainableModel](../#trainablemodel-class).

    # Arguments:
        n_clusters: Number of clusters.
        init: Method of initialization. Options are
            {‘k-means++’, ‘random’, array}, the default is ’k-means++’.
            If passing an array-like object containing the initial centers,
            this should be of shape (n_clusters, n_features).
            If set to ‘random’, `n_clusters` observations (rows) are randomly
            chosen from data for the initial centroids.
        n_init: Number of time the K-means algorithm will be run
            with different centroid seeds (default is 10).

    # References:
        [sklearn.cluster.KMeans](https://scikit-learn.org/stable/
        modules/generated/sklearn.cluster.KMeans.html)
    """

    def __init__(self, n_clusters, n_features, init='k-means++', n_init=10):
        self._model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init)
        self._init = init
        self._n_features = n_features
        self._n_init = n_init
        self._model._n_threads = None  # compatibility: should be removed from scikit-learn

        if isinstance(init, np.ndarray):
            self._model.cluster_centers_ = init
        else:
            self._model.cluster_centers_ = np.zeros((n_clusters, n_features))

    def train(self, data, labels=None, **kwargs):
        """Trains the model.

        # Arguments
            data: Array-like object of shape (n_samples, n_features)
                containing the data to train the model.
            labels: None.
            **kwargs: Optional named parameters.
        """
        self._model.fit(data)

    def predict(self, data):
        """Makes a prediction on input data.

        # Arguments:
            data: Array-like object of shape (n_samples, n_features)
                containing the input data on which to make the prediction.

        # Returns:
            prediction: Model's prediction using the input data.
        """
        predicted_labels = self._model.predict(data)
        return predicted_labels

    def evaluate(self, data, labels):
        """Evaluates the performance of the model.

        # Arguments:
            data: Array-like object of shape (n_samples, n_features)
                containing the input data on which to make the evaluation.
            labels: Array-like object of shape (n_samples,) or
                (n_samples, n_targets) containing the target labels.

        # Returns:
            homogeneity_score: [Homogeneity score](https://scikit-learn.org/
            stable/modules/generated/sklearn.metrics.homogeneity_score.html).

            completeness_score: [Completeness score](https://scikit-learn.org/
            stable/modules/generated/sklearn.metrics.completeness_score.html).

            v_measure_score: [V-measure score](https://scikit-learn.org/
            stable/modules/generated/sklearn.metrics.v_measure_score.html).

            adjusted_rand_score: [Adjusted rand score](https://scikit-learn.org/
            stable/modules/generated/sklearn.metrics.adjusted_rand_score.html).
        """
        prediction = self.predict(data)

        homogeneity_score = metrics.homogeneity_score(labels, prediction)
        completeness_score = metrics.completeness_score(labels, prediction)
        v_measure_score = metrics.v_measure_score(labels, prediction)
        adjusted_rand_score = metrics.adjusted_rand_score(labels, prediction)

        return homogeneity_score, completeness_score, \
            v_measure_score, adjusted_rand_score

    def performance(self, data, labels):
        """Evaluates the performance of the model using
            the most representative metrics.

        # Arguments:
            data: Array-like object of shape (n_samples, n_features)
                containing the input data on which to make the evaluation.
            labels: Array-like object of shape (n_samples,) or
                (n_samples, n_targets) containing the target labels.

        # Returns:
            v_measure_score: [V-measure score](https://scikit-learn.org/
            stable/modules/generated/sklearn.metrics.v_measure_score.html).
        """
        prediction = self.predict(data)
        v_measure_score = metrics.v_measure_score(labels, prediction)

        return v_measure_score

    def get_model_params(self):
        """See base class."""
        return self._model.cluster_centers_

    def set_model_params(self, params):
        """Sets the model's parameters.

        If an array of non-zeroes is given, this is assumed to represent
        the new centroids, and thus the number of runs "n_init" is set to 1.
        """
        if np.array_equal(params,
                          np.zeros((params.shape[0], params.shape[1]))):
            self.__init__(n_clusters=params.shape[0],
                          n_features=self._n_features,
                          init=self._init,
                          n_init=self._n_init)
        else:
            self.__init__(n_clusters=params.shape[0],
                          n_features=self._n_features,
                          init=params,
                          n_init=1)
