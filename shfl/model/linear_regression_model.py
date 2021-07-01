from abc import ABC

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from shfl.model.model import TrainableModel
from .utils import check_initialization_regression
from .utils import check_data_features
from .utils import check_target_size


# Similar with other linear models in sklearn, it can be repeated:
# pylint: disable-msg=R0801
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

    def __init__(self, n_features, n_targets=1):
        check_initialization_regression(n_features)
        check_initialization_regression(n_targets)
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
        check_data_features(self._n_features, data)
        check_target_size(self._n_targets, labels)

        self._model.fit(data, labels)

    def predict(self, data):
        """Makes a prediction on input data.

        # Arguments:
            data: Array-like object of shape (n_samples, n_features)
                containing the input data on which to make the prediction.

        # Returns:
            prediction: Model's prediction using the input data.
        """
        check_data_features(self._n_features, data)

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

        check_target_size(self._n_targets, labels)

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
        check_data_features(self._n_features, data)
        check_target_size(self._n_targets, labels)

        prediction = self.predict(data)
        negative_root_mean_squared_error = - np.sqrt(
            metrics.mean_squared_error(labels, prediction))

        return negative_root_mean_squared_error

    def get_model_params(self):
        """Gets the linear model's parameters."""
        return self._model.intercept_, self._model.coef_

    def set_model_params(self, params):
        """Sets the linear model's parameters."""
        self._model.intercept_ = params[0]
        self._model.coef_ = params[1]
