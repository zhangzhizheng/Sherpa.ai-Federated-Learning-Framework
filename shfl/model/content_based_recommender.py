import numpy as np
import pandas as pd

from shfl.model.recommender import Recommender


def _check_two_columns(data):
    """
    Checks that the array has two columns
    """
    number_of_columns = data.shape[1]
    if number_of_columns != 2:
        raise AssertionError("Data does not have the correct number of columns."
                             "Current data has {} columns".format(number_of_columns))


def _check_is_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df_items should be a dataframe.")


def _check_no_new_items(data, df_items):
    items_in_data = set(np.unique(data[:, 1]))
    items_in_catalog = set(df_items.index)
    if not items_in_data.issubset(items_in_catalog):
        raise AssertionError("Data has items that are not in the catalog.")


class ContentBasedRecommender(Recommender):
    """
    Implementation of a content-based recommender using \
        [Recommender](../model/#recommender-class)

    This class needs the object df_items that must be a pandas dataframe that contains the numeric
    features of the items. The data that is used to train the model and to make predictions is a numpy array in
    which the first column specifies the client and the second the item. There can be no items in the data that
    no not appear in the catalog df_item. Therefore, the index of df_items must contain every value in the second
    column of data.
    """

    def __init__(self, df_items):
        super().__init__()
        _check_is_dataframe(df_items)
        df_items.index.name = "itemid"
        self._df_items = df_items
        self._mu = None
        self._profile = None

    def _join_dataframe_with_items_features(self, data):
        _check_two_columns(data)
        _check_no_new_items(data, self._df_items)
        df_data = pd.DataFrame(data, columns=['userid', "itemid"])
        df = df_data.join(self._df_items, on="itemid").drop(["userid", "itemid"], axis=1)
        return df

    def train_recommender(self, data, labels):
        """
        Method that trains the model

        # Arguments:
            data: Data to train the model .Only includes the data of this client and every item must be in the catalog.
            labels: Label for each train element
        """
        self._mu = np.mean(labels)
        df = self._join_dataframe_with_items_features(data)
        self._profile = df.multiply(labels - self._mu, axis=0).mean().values

    def predict_recommender(self, data):
        """
        Predict labels for data. Only includes the data of this client and every item must be in the catalog.

        # Arguments:
            data: Data for predictions. Only includes the data of this client

        # Returns:
            predictions: Array with predictions for data
        """
        df = self._join_dataframe_with_items_features(data)
        predictions = self._mu + df.values.dot(self._profile)
        return predictions

    def evaluate_recommender(self, data, labels):
        """
        This method must returns the root mean square error

        # Arguments:
            data: Data to be evaluated. Only includes the data of this client and every item must be in the catalog.
            labels: True values of data of this client
        """
        predictions = self.predict(data)
        if predictions.size == 0:
            rmse = 0
        else:
            rmse = np.sqrt(np.mean((predictions - labels) ** 2))
        return rmse

    def get_model_params(self):
        """
        Gets the params that define the model

        # Returns:
            params: Mean rating
        """
        return self._mu, self._profile

    def set_model_params(self, params):
        """
        Update the params that define the model

        # Arguments:
            params: Parameter defining the model
        """
        self._mu, self._profile = params

    def performance_recommender(self, data, labels):
        """
        This method returns the root mean square error of the recommender.

        # Arguments:
            data: Data to be evaluated. Only includes the data of this client and every item must be in the catalog.
            labels: True values of data of this client
        """
        predictions = self.predict(data)
        if predictions.size == 0:
            rmse = 0
        else:
            rmse = np.sqrt(np.mean((predictions - labels) ** 2))
        return rmse
