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

    # pylint: disable=too-many-instance-attributes
    # Eight is reasonable in this case.

    def __init__(self, data, labels, train_proportion=0.8, shuffle=True):
        super().__init__()
        self._data = data
        self._labels = labels
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
            self._data, self._labels = \
                shuffle_rows(self._data, self._labels)

        self._train_data, self._train_labels, \
            self._test_data, self._test_labels = \
            split_train_test(self._data, self._labels,
                             self._train_proportion)


def shuffle_rows(data, labels):
    """Shuffles rows on inputs simultaneously.

    It supports either Pandas DataFrame/Series or Numpy arrays.

    # Arguments:
        data: Array-like object containing data.
        labels: Array-like object containing target labels.
    """
    randomize = np.arange(len(labels))
    np.random.shuffle(randomize)

    print()

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
                   equal_size=False, train_proportion=0.8,
                   v_shuffle=True, h_shuffle=True):
    """Splits a dataset vertically.

    Splits a an array-like object along columns.

    # Arguments:
        data: dataframe or numpy array
        labels: series or array containing the target labels
        n_chunks: integer denoting the desired number of vertical splits
        indices_or_sections: int or 1-D array containing
            column indices (integers) at which to split
            (see [numpy split](https://numpy.org/doc/stable/reference/generated/numpy.split.html)).
            If requested split is not possible, an error is raised.
        equal_size: Boolean for splitting in equal number of columns.
            Only used if "indices_or_sections" is int.
        train_percentage: float between 0 and 1 to indicate how much data
            is dedicated to train (if 1 is provided, data is
            assigned entirely to train)
        v_shuffle: Boolean for shuffling columns before the vertical split
            (default True)
        h_shuffle: Boolean for shuffling rows before the train/test split
            (default True)

    # Returns:
        train_data: list whose items contain the train data
            of each chunk
        train_labels: train labels (it is the same for all train chunks)
        test_data: list whose items contain the test data
            of one single chunk
        test_labels: test labels (it is the same for all test chunks)
    """

    # pylint: disable=too-many-arguments
    # Seven is reasonable, although we might consider future refactoring.

    if isinstance(data, np.ndarray):
        def get_slice(dataset, col_index):
            return dataset[:, col_index]
    elif isinstance(data, pd.DataFrame):
        def get_slice(dataset, col_index):
            return dataset.iloc[:, col_index]
    else:
        raise TypeError("Data must be either a pd.DataFrame or "
                        "a numpy array.")

    # Get column indices for split:
    n_features = data.shape[1]
    features = np.arange(n_features)
    if v_shuffle:
        np.random.shuffle(features)
    if not hasattr(indices_or_sections, "__len__"):
        if not equal_size:
            indices_or_sections = np.sort(np.random.choice(
                np.arange(1, n_features), indices_or_sections - 1,
                replace=False))

    if h_shuffle:
        data, labels = shuffle_rows(data, labels)

    # Split train/test samples (horizontally on rows):
    train_data, train_labels, test_data, test_labels = \
        split_train_test(data, labels, train_proportion)

    # Split features (vertically on columns):
    chunks_indices = np.split(features, indices_or_sections)
    train_data = [get_slice(train_data, indices)
                  for indices in chunks_indices]
    if len(test_data) > 0:
        test_data = [get_slice(test_data, indices)
                     for indices in chunks_indices]

    return train_data, train_labels, test_data, test_labels
