import numpy as np

from shfl.model.recommender import Recommender


class MeanRecommender(Recommender):
    """Mean recommender model.

    Implements the class [Recommender](./#recommender).

    Given a set of labels in the training set of each client,
    computes the mean value to make predictions.
    """

    def __init__(self):
        super().__init__()
        self._mu = None

    def train_recommender(self, data, labels, **kwargs):
        """See base class."""
        self._mu = np.mean(labels)

    def predict_recommender(self, data):
        """See base class."""
        predictions = np.full(len(data), self._mu)
        return predictions

    def evaluate_recommender(self, data, labels):
        """Evaluates the performance of the model.

        ## Arguments:
            data: Array-like object containing data on which
                to make the prediction. The data must belong to
                only one client.
            labels: Array-like object containing true target labels.
        # Returns:
            rmse: Root mean square error.
        """
        predictions = self.predict(data)
        if predictions.size == 0:
            rmse = 0
        else:
            rmse = np.sqrt(np.mean((predictions - labels) ** 2))
        return rmse

    def get_model_params(self):
        """See base class."""
        return self._mu

    def set_model_params(self, params):
        """See base class."""
        self._mu = params

    def performance_recommender(self, data, labels):
        """See base class."""
        predictions = self.predict(data)
        if predictions.size == 0:
            rmse = 0
        else:
            rmse = np.sqrt(np.mean((predictions - labels) ** 2))
        return rmse
