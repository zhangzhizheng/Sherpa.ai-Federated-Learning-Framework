import abc
import numpy as np

from shfl.model.model import TrainableModel


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
        self._check_data(data)
        self._check_data_labels(data, labels)
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
        self._check_data(data)
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
        self._check_data(data)
        self._check_data_labels(data, labels)

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

    @staticmethod
    def _check_data(data):
        """Checks whether the data belongs to a single user.

        # Arguments:
            data: Array-like object containing the data.
        """
        number_of_clients = len(np.unique(data[:, 0]))

        if number_of_clients > 1:
            raise AssertionError(
                "Data need to correspond to a single user. "
                "Current data includes "
                "{} clients.".format(number_of_clients))

    @staticmethod
    def _check_data_labels(data, labels):
        """Checks whether the data and the labels
        have matching dimensions.

        # Arguments:
            data: Data to train the model.
            labels: Target labels.
        """
        rows_in_data = data.shape[0]
        number_of_labels = len(labels)

        if rows_in_data != number_of_labels:
            raise AssertionError(
                "Data and labels do not have matching dimensions. "
                "Current data has {} rows and there are "
                "{} labels".format(rows_in_data, number_of_labels))
