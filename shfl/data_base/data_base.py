import abc
import numpy as np
import pandas as pd


def shuffle_rows(data, labels):
    """
    Shuffles rows in two data structures simultaneously.
    It supports either pd.DataFrame/pd.Series or numpy arrays.
    """
    randomize = np.arange(len(labels))
    np.random.shuffle(randomize)

    if isinstance(data, (pd.DataFrame, pd.Series)) and \
            isinstance(labels, (pd.DataFrame, pd.Series)):
        data = data.iloc[randomize]
        labels = labels.iloc[randomize]

    elif isinstance(data, np.ndarray) and \
            isinstance(labels, np.ndarray):
        data = data[randomize, ]
        labels = labels[randomize]

    else:
        raise TypeError("Data and labels must be either "
                        "pd.DataFrame/pd.Series or numpy arrays.")

    return data, labels


def split_train_test(data, labels, dim):
    """
    Method that randomly chooses the train and test sets
    from data and labels.

    # Arguments:
        data: Data for extracting the validation data
        labels: Array with labels
        dim: Size for validation data

    # Returns:
        new_data: Data, labels, validation data and validation labels
    """

    data, labels = shuffle_rows(data, labels)

    test_data = data[0:dim]
    test_labels = labels[0:dim]

    rest_data = data[dim:]
    rest_labels = labels[dim:]

    return rest_data, rest_labels, test_data, test_labels


def vertical_split(data, labels, n_chunks,
                   train_percentage=0.8, shuffle_data=True):
    """
    Splits a 2-D dataset vertically (i.e. along columns).

    # Arguments:
        data: dataframe or numpy array (2-D)
        labels: array containing the target labels
        n_chunks: integer denoting the desired number of vertical splits
        train_percentage: float between 0 and 1 to indicate how much data
            is dedicated to train (if 1 is provided, data is unchanged and
            assigned entirely to train, while test data is set to None)
        shuffle_data: Boolean for shuffling rows in the train and test
            data after the split (default True)
        seed: integer, set for reproducible results (optional)
    # Returns:
        train_data: dictionary whose items contain the train data
            of each chunk
        train_labels: train labels (it is the same for all train chunks)
        test_data: dictionary whose items contain the test data
            of one single chunk
        test_labels: test labels (it is the same for all test chunks)
    """

    if isinstance(data, np.ndarray):
        def get_slice(dataset, col_index):
            return dataset[:, col_index]
    elif isinstance(data, pd.DataFrame):
        def get_slice(dataset, col_index):
            return dataset.iloc[:, col_index]
    else:
        raise TypeError("Data must be either a pd.DataFrame or "
                        "a numpy array.")

    n_features = data.shape[1]
    if n_chunks > n_features:
        raise AssertionError("Too many vertical divisions: " +
                             str(n_chunks) + " requested, but data has " +
                             str(n_features) + " columns.")

    # Split train/test samples (horizontally on rows):
    if train_percentage < 1.:
        test_size = round(len(data) * (1 - train_percentage))
        train_data, train_labels, test_data, test_labels = \
            split_train_test(data, labels, test_size)
    else:
        train_data, train_labels, test_data, test_labels = \
            data, labels, None, None

    if shuffle_data:
        train_data, train_labels = shuffle_rows(train_data, train_labels)
        if test_data is not None:
            test_data, test_labels = shuffle_rows(test_data, test_labels)

    # Split features (vertically on columns):
    features = np.arange(n_features)
    np.random.shuffle(features)
    split_feature_index = np.sort(np.random.choice(
        np.arange(1, n_features), n_chunks - 1, replace=False))
    chunk_features = np.split(features, split_feature_index)

    train_data = {i: get_slice(train_data, chunk_features[i])
                  for i in range(n_chunks)}
    if test_data is not None:
        test_data = {i: get_slice(test_data, chunk_features[i])
                     for i in range(n_chunks)}

    return train_data, train_labels, test_data, test_labels


class DataBase(abc.ABC):
    """
    Abstract class for data base.

    Load method must be implemented in order to create a database able to \
    interact with the system, in concrete with data distribution methods \
    (see: [Data Distribution](../data_distribution)).

    Load method should save data in the protected Attributes:

    # Attributes:
        * **train_data, train_labels, test_data, test_labels**

    # Properties:
        train: Returns train data and labels
        test: Returns test data and labels
        data: Returns train data, train labels, validation data,
            validation labels, test data and test labels
    """

    def __init__(self):
        self._train_data = []
        self._test_data = []
        self._train_labels = []
        self._test_labels = []

    @property
    def train(self):
        return self._train_data, self._train_labels

    @property
    def test(self):
        return self._test_data, self._test_labels

    @property
    def data(self):
        return self._train_data, self._train_labels, \
               self._test_data, self._test_labels

    @abc.abstractmethod
    def load_data(self):
        """
        Abstract method that loads the data
        """

    def shuffle(self):
        """
        Shuffles all data
        """
        self._train_data, self._train_labels = \
            shuffle_rows(self._train_data, self._train_labels)
        self._test_data, self._test_labels = \
            shuffle_rows(self._test_data, self._test_labels)


class LabeledDatabase(DataBase):
    """
    Class to create generic labeled database from data and labels vectors.
    By default, the data is shuffled and split into train and test.
    After, additional shuffling is performed by default separately on train
    and test data.

    # Arguments
        data: Data features to load
        labels: Labels for this features
        train_percentage: float between 0 and 1 to indicate how much data
            is dedicated to train (if 1 is provided, data is unchanged and
            assigned entirely to train, while test data is set to None)
        shuffle_data: Boolean for shuffling rows in the train and test data
            after the split (default True)
    """

    def __init__(self, data, labels, train_percentage=0.8, shuffle_data=True):
        super().__init__()
        self._data = data
        self._labels = labels

        if train_percentage < 1.:
            test_size = round(len(self._data) * (1 - train_percentage))
            self._train_data, self._train_labels, \
                self._test_data, self._test_labels = \
                split_train_test(self._data, self._labels, test_size)
        else:
            self._train_data, self._train_labels, \
                self._test_data, self._test_labels = \
                self._data, self._labels, None, None

        if shuffle_data:
            self.shuffle()

    def load_data(self):
        """
        Load data

        # Returns
            all_data : train data, train labels, test data and test labels
        """

        return self.data
