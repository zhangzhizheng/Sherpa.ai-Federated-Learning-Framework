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


def split_train_test(data, labels, train_percentage=0.8, shuffle=True):
    """
    Method that randomly chooses the train and test sets
    from data and labels.

    # Arguments:
        data: Data for extracting the validation data
        labels: Array with labels
        train_percentage: float between 0 and 1 to indicate how much data
            is dedicated to train
        shuffle: Boolean for shuffling rows before the train/test split
            (default True)

    # Returns:
        train_data, train_labels, test_data, test_labels: the data after
            the split
    """

    if shuffle:
        data, labels = shuffle_rows(data, labels)

    test_size = round(len(data) * (1 - train_percentage))

    train_data = data[:-test_size]
    train_labels = labels[:-test_size]

    test_data = data[-test_size:]
    test_labels = labels[-test_size:]

    return train_data, train_labels, test_data, test_labels


def vertical_split(data, labels, indices_or_sections=2,
                   equal_size=False, train_percentage=0.8,
                   v_shuffle=True, h_shuffle=True):
    """
    Splits a 2-D dataset vertically (i.e. along columns).

    # Arguments:
        data: dataframe or numpy array (2-D)
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

    # Split train/test samples (horizontally on rows):
    if train_percentage < 1.:
        train_data, train_labels, test_data, test_labels = \
            split_train_test(data, labels, train_percentage, h_shuffle)
    else:
        train_data, train_labels, test_data, test_labels = \
            data, labels, None, None

    # Split features (vertically on columns):
    chunks_indices = np.split(features, indices_or_sections)
    train_data = [get_slice(train_data, indices)
                  for indices in chunks_indices]
    if test_data is not None:
        test_data = [get_slice(test_data, indices)
                     for indices in chunks_indices]

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

    # Arguments
        data: Data features to load
        labels: Labels for this features
        train_percentage: float between 0 and 1 to indicate how much data
            is dedicated to train (if 1 is provided, data is
            assigned entirely to train)
        shuffle: Boolean for shuffling rows before the train/test split
            (default True)
    """

    def __init__(self, data, labels, train_percentage=0.8, shuffle=True):
        super(LabeledDatabase, self).__init__()
        self._data = data
        self._labels = labels
        self._train_percentage = train_percentage
        self._shuffle = shuffle

    def load_data(self):
        """
        Returns all data. If not loaded, loads the data.

        # Returns
            all_data : train data, train labels, test data and test labels
        """

        if not self._train_data:
            self._load_data()

        return self.data

    def _load_data(self):
        """
        Populates private attributes train data and labels,
        and test data and labels.
        """

        if self._train_percentage < 1.:
            self._train_data, self._train_labels, \
                self._test_data, self._test_labels = \
                split_train_test(self._data, self._labels,
                                 self._train_percentage, self._shuffle)
        else:
            self._train_data, self._train_labels = self._data, self._labels
