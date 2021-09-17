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

import abc
import numpy as np

from shfl.model.model import TrainableModel
from .utils import check_data_recommender
from .utils import check_data_labels_recommender


class Recommender(TrainableModel):
    """Wraps a recommender model.

    Implements the class [TrainableModel](../#trainablemodel-class).

    The input data to this model should be an array-like object where
    the first column specifies the client ID.
    In particular, both the training and testing data in each client
    should be such that the value across the first column is constant.
    We do not want the data of a client to go to a different client.
    """

    def __init__(self):
        self._client_identifier = None

    def train(self, data, labels, **kwargs):
        """Trains the model.

        # Arguments:
            data: The data belonging to only one client
                on which to train the model.
            labels: The target labels.
            **kwargs: Optional named parameters.
        """
        check_data_recommender(data)
        check_data_labels_recommender(data, labels)
        self._client_identifier = data[0, 0]
        self.train_recommender(data, labels, **kwargs)

    @abc.abstractmethod
    def train_recommender(self, data, labels, **kwargs):
        """Trains the model (abstract method).
        """

    def predict(self, data):
        """Makes a prediction on input data.

        # Arguments:
            data: Data belonging to only one client
                on which to make the prediction.

        # Returns:
            predictions: Model's prediction using the input data.
        """
        check_data_recommender(data)
        return self.predict_recommender(data)

    @abc.abstractmethod
    def predict_recommender(self, data):
        """Makes a prediction on input data (abstract method).
        """

    def evaluate(self, data, labels):
        """Evaluates the performance of the model.

        # Arguments:
            data: Data belonging to only one client
                on which to make the evaluation.
            labels: The true labels.
        # Returns:
            metrics: Metrics for the evaluation.
        """
        check_data_labels_recommender(data, labels)

        predictions = self.predict(data)
        if predictions.size == 0:
            root_mean_square = 0
        else:
            root_mean_square = np.sqrt(np.mean((predictions - labels) ** 2))

        return root_mean_square

    @abc.abstractmethod
    def get_model_params(self):
        """See base class."""

    @abc.abstractmethod
    def set_model_params(self, params):
        """See base class."""

    def performance(self, data, labels):
        """Evaluates the performance of the model using
        the most representative metrics.

        # Arguments:
            data: Data belonging to only one client
                on which to make the evaluation.
            labels: The true labels.

        # Returns:
            metrics: Most representative metrics for the evaluation.
        """

        return self.evaluate(data, labels)
