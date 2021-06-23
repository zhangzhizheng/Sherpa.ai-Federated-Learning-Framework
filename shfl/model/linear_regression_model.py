from abc import ABC

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from shfl.model.model import TrainableModel
from .utils import get_model_params
from .utils import set_model_params
from .utils import _check_data


class LinearRegressionModel(TrainableModel, ABC):
    """Wraps the scikit-learn linear regression model.

    Implements the class [TrainableModel](../#trainablemodel-class).

    # Arguments:
        n_features: Number of features.
        n_targets: Optional; Number of targets to predict (default is 1).

    # References:
        [sklearn.linear_model.LinearRegression](https://scikit-learn.org/
        stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    """

    get_model_params = get_model_params
    set_model_params = set_model_params
    _check_data = _check_data

    def __init__(self, n_features, n_targets=1):
        self._check_initialization(n_features)
        self._check_initialization(n_targets)
        self._model = LinearRegression()
        self._n_features = n_features
        self._n_targets = n_targets
        self.set_model_params([np.zeros(n_targets),
                               np.zeros((n_targets, n_features))])

    def train(self, data, labels, **kwargs):
        """Trains the model.

        # Arguments:
            data: Array-like object of shape (n_samples, n_features)
                containing the data to train the model.
            labels: Array-like object of shape (n_samples,)
                or (n_samples, n_targets) containing the target labels.
            **kwargs: Optional named parameters.
        """
        self._check_data(data)
        self._check_labels(labels)

        self._model.fit(data, labels)

    def predict(self, data):
        """Makes a prediction on input data.

        # Arguments:
            data: Array-like object of shape (n_samples, n_features)
                containing the input data on which to make the prediction.

        # Returns:
            prediction: Model's prediction using the input data.
        """
        self._check_data(data)

        prediction = self._model.predict(data)

        return prediction

    def evaluate(self, data, labels):
        """Evaluates the performance of the model.

        # Arguments:
            data: Array-like object of shape (n_samples, n_features)
                containing the input data on which to make the evaluation.
            labels: Array-like object of shape (n_samples,) or
                (n_samples, n_targets) containing the target labels.

        # Returns:
            rmse: [Root mean square error](https://scikit-learn.org/
            stable/modules/generated/sklearn.metrics.mean_squared_error.html).

            r2: [R2 score](https://scikit-learn.org/
            stable/modules/generated/sklearn.metrics.r2_score.html).
        """

        self._check_labels(labels)

        prediction = self.predict(data)
        root_mean_squared_error = np.sqrt(
            metrics.mean_squared_error(labels, prediction))
        r2_score = metrics.r2_score(labels, prediction)

        return root_mean_squared_error, r2_score

    def performance(self, data, labels):
        """Evaluates the performance of the model using
        the most representative metrics.

        # Arguments:
            data: Array-like object of shape (n_samples, n_features)
                containing the input data on which to make the evaluation.
            labels: Array-like object of shape (n_samples,) or
                (n_samples, n_targets) containing the target labels.

        # Returns:
            negative_rmse: Negative root mean square error.
        """
        self._check_data(data)
        self._check_labels(labels)

        prediction = self.predict(data)
        negative_root_mean_squared_error = - np.sqrt(
            metrics.mean_squared_error(labels, prediction))

        return negative_root_mean_squared_error

    def _check_labels(self, labels):
        if labels.ndim == 1:
            if self._n_targets != 1:
                raise AssertionError(
                    "Labels need to have the same number of targets "
                    "described by the model " + str(self._n_targets) +
                    ". Current labels have only 1 target.")
        elif labels.shape[1] != self._n_targets:
            raise AssertionError(
                "Labels need to have the same number of targets "
                "described by the model " + str(self._n_targets) +
                ". Current labels have " + str(labels.shape[1]) +
                " targets.")

    @staticmethod
    def _check_initialization(n_dimensions):
        """Checks whether the model's initialization is correct.

        The number of features and targets must be an integer
        equal or greater to one.

        # Arguments:
            n_rounds: Number of features or targets.
        """
        if not isinstance(n_dimensions, int):
            raise AssertionError(
                "n_features and n_targets must be a positive integer number. "
                "Provided value " + str(n_dimensions) + ".")
        if n_dimensions < 1:
            raise AssertionError(
                "n_features and n_targets must be equal or greater that 1. "
                "Provided value " + str(n_dimensions) + ".")
