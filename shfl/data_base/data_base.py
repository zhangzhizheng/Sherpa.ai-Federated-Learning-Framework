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
import pandas as pd


class DataBase(abc.ABC):
    """Represents a generic data base.

    This interface allows for a dataset to interact with the Framework's methods.
    In particular, with data distribution methods
    (see class [Data Distribution](../data_distribution)).
    The loaded data should be saved in the protected attributes.

    # Attributes:
        train_data: Array-like object containing the train data.
        test_data: Array-like object containing the test data.

    # Properties:
        train: The train data.
        test: The test data.
        data: 2-Tuple as (train data, test data).
    """

    def __init__(self):
        self._train_data = []
        self._test_data = []
        self._data = None

    @property
    def train(self):
        """Returns the train data."""
        return self._train_data

    @property
    def test(self):
        """Returns the test data."""
        return self._test_data

    @property
    def data(self):
        """Returns all data.
        """
        return self._train_data, self._test_data

    @abc.abstractmethod
    def load_data(self, **kwargs):
        """Loads the train and test data.

        Abstract method.

        # Returns:
            data: 2-Tuple as (train data, test data).
        """

    def split_data(self, train_proportion=0.8, shuffle=True):
        """Splits the data."""

        if shuffle:
            self._data, = shuffle_rows(self._data)

        self._train_data, self._test_data = \
            split_train_test(self._data, train_proportion=train_proportion)


class LabeledDatabase(DataBase):
    """Represents a generic labeled data base.

    Implements the class [DataBase](./#database-class).

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
        super().__init__()
        self._train_labels = []
        self._test_labels = []
        self._labels = None

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
    def load_data(self, **kwargs):
        """Loads the train and test data.

        Abstract method.

        # Returns:
            data: 4-Tuple as (train data, train labels, test data, test labels).
        """

    def split_data(self, train_proportion=0.8, shuffle=True):
        """Splits the data.

        # Arguments:
            train_proportion: Optional; Float between 0 and 1 proportional to the
                amount of data to dedicate to train. If 1 is provided, all data is
                assigned to train (default is 0.8).
            shuffle: Optional; Boolean for shuffling rows before the
                train/test split (default is True).
        """

        if shuffle:
            self._data, self._labels = shuffle_rows(self._data, self._labels)

        self._train_data, self._train_labels, \
            self._test_data, self._test_labels = \
            split_train_test(self._data, self._labels,
                             train_proportion=train_proportion)


class WrapLabeledDatabase(LabeledDatabase):
    """Wraps labeled data in a database.

    Implements the class [LabeledDatabase](./#labeleddatabase-class).

    # Arguments:
        data: Array-like object containing the data.
        labels: Array-like object containing the target labels.
    """

    def __init__(self, data, labels):
        super().__init__()
        self._data = data
        self._labels = labels

    # False positive since using **kwargs
    # pylint: disable=arguments-differ
    def load_data(self, train_proportion=0.8, shuffle=True):
        """Loads the train and test data.

        # Arguments:
            train_proportion: Optional; Float between 0 and 1 proportional to the
                amount of data to dedicate to train. If 1 is provided, all data is
                assigned to train (default is 0.8).
            shuffle: Optional; Boolean for shuffling rows before the
                train/test split (default is True).

        # Returns:
            data: 4-Tuple as (train data, train labels, test data, test labels).
        """

        self.split_data(train_proportion, shuffle)

        return self.data


def shuffle_rows(*array_like_objects):
    """Shuffles the rows on an arbitrary number of inputs array-like objects simultaneously.

    It supports either Pandas DataFrame/Series or Numpy arrays.
    The inputs must all have the same length (i.e. number of rows).

    # Arguments:
        *array_like_objects: Array-like objects containing data.

    # Returns:
        shuffled_data: A tuple with the shuffled objects.
    """

    for item in array_like_objects:
        check_array_like_data_type(item)

    check_all_same_length(array_like_objects)

    randomize = np.arange(len(array_like_objects[0]))
    np.random.shuffle(randomize)
    shuffled_data = tuple(get_rows(item, randomize)
                          for item in array_like_objects)

    return shuffled_data


def split_train_test(*array_like_objects, train_proportion=0.8):
    """Splits an arbitrary number of inputs array-like objects into train and test sets.

    # Arguments:
        array_like_objects: Array-like objects containing the data to split.
        train_proportion: Optional; Float between 0 and 1 proportional to the
            amount of data to dedicate to train. If 1 is provided, all data is
            assigned to train (default is 0.8).

    # Returns:
        train_data: A tuple with the train objects.
        test_data: A tuple with the test objects.

    # Example:

    If used on two array-like objects `data` and `labels`, the output order is
    as follows:

    ```python
    train_data, train_labels, test_data, test_labels = split_train_test(data, labels)
    ```
    """
    check_all_same_length(array_like_objects)

    train_size = round(len(array_like_objects[0]) * train_proportion)

    train_data = [item[:train_size] for item in array_like_objects]
    test_data = [item[train_size:] for item in array_like_objects]

    return (*train_data, *test_data)


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
        split_train_test(data, labels, train_proportion=train_proportion)

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


def check_all_same_length(objects):
    """Checks that the objects have the same length."""
    if not all(len(item) == len(objects[0])
               for item in objects):
        raise AssertionError("Lengths of objects do not match.")
