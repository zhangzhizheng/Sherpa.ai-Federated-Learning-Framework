import abc
import numpy as np
import pandas as pd


class DataBase(abc.ABC):
    """Represents a generic data base.

    This interface allows for a dataset to interact with the Framework's methods.
    In particular, with data distribution methods
    (see class [Data Distribution](../data_distribution)).
    The loaded data should be saved in the protected attributes.

    # Attributes:
        train_data: Array-like object containing the train data.
        train_labels: Array-like object containing the train target labels.
        test_data: Array-like object containing the test data.
        test_labels: Array-like object containing the test target labels.

    # Properties:
        train: 2-Tuple as (train data, train labels).
        test: 2-Tuple as (test data, test labels).
        data: 4-Tuple as (train data, train labels, test data, test labels).
    """

    def __init__(self):
        self._train_data = []
        self._test_data = []
        self._train_labels = []
        self._test_labels = []

    @property
    def train(self):
        """Returns train data and associated target labels."""
        return self._train_data, self._train_labels

    @property
    def test(self):
        """Returns test data and associated target labels."""
        return self._test_data, self._test_labels

    @property
    def data(self):
        """Returns all data as train data, train labels,
        test data, test labels.
        """
        return self._train_data, self._train_labels, \
            self._test_data, self._test_labels

    @abc.abstractmethod
    def load_data(self):
        """Specifies data location and operations at loading.

        Abstract method.
        """


class LabeledDatabase(DataBase):
    """Creates a generic labeled database from input data and labels.

    Implements the class [DataBase](./#database-class).

    # Arguments:
        data: Array-like object containing the features.
        labels: Array-like object containing the target labels.
        train_proportion: Optional; Float between 0 and 1 proportional to the
            amount of data to dedicate to train. If 1 is provided, all data is
            assigned to train (default is 0.8).
        shuffle: Optional; Boolean for shuffling rows before the
            train/test split (default is True).
    """

    def __init__(self, data, labels, train_proportion=0.8, shuffle=True):
        super().__init__()
        self._data_and_labels = (data, labels)
        self._train_proportion = train_proportion
        self._shuffle = shuffle

    def load_data(self):
        """Loads the data (once) and returns the train/test partitions.

        # Returns
            data: 4-Tuple as (train data, train labels, test data, test labels).
        """

        if not self._train_data:
            self._load_data()

        return self.data

    def _load_data(self):
        """Populates private attributes train data and labels,
            and test data and labels.
        """
        if self._shuffle:
            self._data_and_labels = shuffle_rows(*self._data_and_labels)

        self._train_data, self._train_labels, \
            self._test_data, self._test_labels = \
            split_train_test(*self._data_and_labels,
                             train_proportion=self._train_proportion)


def shuffle_rows(data, labels):
    """Shuffles rows on inputs simultaneously.

    It supports either Pandas DataFrame/Series or Numpy arrays.

    # Arguments:
        data: Array-like object containing data.
        labels: Array-like object containing target labels.
    """

    check_array_like_data_type(data)
    check_array_like_data_type(labels)

    randomize = np.arange(len(labels))
    np.random.shuffle(randomize)
    shuffled_data = get_rows(data, randomize)
    shuffled_labels = get_rows(labels, randomize)

    return shuffled_data, shuffled_labels


def split_train_test(data, labels, train_proportion=0.8):
    """Splits data and labels into train and test sets.

    # Arguments:
        data: Array-like object containing the data to split.
        labels: Array-like object containing target labels.
        train_proportion: Optional; Float between 0 and 1 proportional to the
            amount of data to dedicate to train. If 1 is provided, all data is
            assigned to train (default is 0.8).

    # Returns:
        train_data: Array-like object.
        train_labels: Array-like object.
        test_data: Array-like object.
        test_labels: Array-like object.
    """
    train_size = round(len(data) * train_proportion)

    train_data = data[:train_size]
    train_labels = labels[:train_size]

    test_data = data[train_size:]
    test_labels = labels[train_size:]

    return train_data, train_labels, test_data, test_labels


def vertical_split(data, labels, indices_or_sections=2,
                   train_proportion=0.8, v_shuffle=True):
    """Splits a dataset along columns.

    # Arguments:
        data: Dataframe or numpy array.
        labels: Series or array containing the target labels.
        indices_or_sections: Optional; Int or 1-D array containing
            column indices (integers) at which to split
            (see [`numpy.array_split`](https://numpy.org/doc/stable/\
reference/generated/numpy.array_split.html)).
        train_proportion: Optional; Float between 0 and 1 indicating how much data
            is dedicated to train (if 1 is provided, data is
            assigned entirely to train).
        v_shuffle: Boolean for shuffling columns before the vertical split
            (default True).

    # Returns:
        train_data: List whose items contain the train data
            of each chunk.
        train_labels: Train labels (it is the same for all train chunks)
        test_data: List whose items contain the test data
            of one single chunk.
        test_labels: Test labels (it is the same for all test chunks).
    """

    check_array_like_data_type(data)

    n_features = data.shape[1]
    features = np.arange(n_features)
    if v_shuffle:
        np.random.shuffle(features)

    # Split train/test samples (horizontally on rows):
    train_data, train_labels, test_data, test_labels = \
        split_train_test(data, labels, train_proportion)

    # Split features (vertically on columns):
    chunks_indices = np.array_split(features, indices_or_sections)
    train_data = [get_columns(train_data, indices)
                  for indices in chunks_indices]
    if len(test_data) > 0:
        test_data = [get_columns(test_data, indices)
                     for indices in chunks_indices]

    return train_data, train_labels, test_data, test_labels


def get_rows(data, rows_indices):
    """Gets the desired columns in data.

    # Arguments:
        data: Input data, an array-like object.
        rows_indices: Array, the desired rows' indices.

    # Return:
        selected_rows: The desired columns of data.
    """
    selected_rows = None
    if isinstance(data, np.ndarray):
        selected_rows = data[rows_indices, ]
    if isinstance(data, (pd.DataFrame, pd.Series)):
        selected_rows = data.iloc[rows_indices]

    return selected_rows


def get_columns(data, column_indices):
    """Gets the desired columns in data.

    # Arguments:
        data: Input data, an array-like object.
        column_indices: Array, the desired columns' indices.

    # Return:
        selected_columns: The desired columns of data.
    """
    selected_columns = None
    if isinstance(data, np.ndarray):
        selected_columns = data[:, column_indices]
    if isinstance(data, (pd.DataFrame, pd.Series)):
        selected_columns = data.iloc[:, column_indices]

    return selected_columns


def check_array_like_data_type(data):
    """Checks that the data is in one of the allowed types.

    At the present, only array-like types are allowed.

    # Arguments:
        data: Data to be checked.
    """

    allowed_types = (np.ndarray, pd.DataFrame, pd.Series)
    if type(data) not in allowed_types:
        raise TypeError("Data must be either a pd.DataFrame or "
                        "a numpy array.")
