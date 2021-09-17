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

import numpy as np
import pandas as pd
import pytest

from shfl.data_base.data_base import DataBase
from shfl.data_base.data_base import LabeledDatabase
from shfl.data_base.data_base import WrapLabeledDatabase
from shfl.data_base.data_base import shuffle_rows
from shfl.data_base.data_base import split_train_test
from shfl.data_base.data_base import vertical_split


class DataBaseTest(DataBase):
    """Test class for DataBase."""
    # False positive since using **kwargs
    # pylint: disable=arguments-differ
    def load_data(self, train_proportion=0.8, shuffle=True):
        """See base class."""
        self._data = np.random.rand(60, 12)

        self.split_data(train_proportion, shuffle)

        return self.data


class LabeledDatabaseTest(LabeledDatabase):
    """Test class for LabeledDatabase."""

    # False positive since using **kwargs
    # pylint: disable=arguments-differ
    def load_data(self, train_proportion=0.8, shuffle=True):
        """See base class."""
        self._data = np.random.rand(60, 12)
        self._labels = np.random.randint(0, 2, size=(60,))

        self.split_data(train_proportion, shuffle)

        return self.data


@pytest.fixture(name="data_and_labels_arrays")
def fixture_data_and_labels_arrays():
    """Returns data and labels arrays containing random values."""
    data = np.random.rand(60, 12)
    labels = np.random.randint(0, 2, size=(60,))

    return data, labels


@pytest.fixture(name="data_and_labels_dataframes")
def fixture_data_and_labels_dataframes(data_and_labels_arrays):
    """Returns data and labels dataframes containing random values."""
    data, labels = data_and_labels_arrays
    data = pd.DataFrame(data)
    labels = pd.Series(labels)

    return data, labels


def test_data_base_load_data():
    """Checks that the DataBase class loads data correctly."""
    database = DataBaseTest()

    train_data, test_data = database.load_data()

    assert database.train is not None
    assert database.test is not None
    np.testing.assert_array_equal(train_data, database.train)
    np.testing.assert_array_equal(test_data, database.test)


def test_labeled_data_base_load_data():
    """Checks that the LabeledDatabase class loads data correctly."""
    database = LabeledDatabaseTest()

    train_data, train_labels, test_data, test_labels = database.load_data()

    assert database.train[0] is not None
    assert database.train[1] is not None
    assert database.test[0] is not None
    assert database.test[1] is not None
    np.testing.assert_array_equal(train_data, database.train[0])
    np.testing.assert_array_equal(train_labels, database.train[1])
    np.testing.assert_array_equal(test_data, database.test[0])
    np.testing.assert_array_equal(test_labels, database.test[1])


@pytest.mark.parametrize("data_labels",
                         ["data_and_labels_arrays",
                          "data_and_labels_dataframes"])
def test_wrap_labeled_database(data_labels, request):
    """Checks the creation of a labeled database."""
    data, labels = request.getfixturevalue(data_labels)
    database = WrapLabeledDatabase(data, labels)
    train_data, train_labels, test_data, test_labels = database.load_data()

    assert train_data is not None
    assert train_labels is not None
    assert test_data is not None
    assert test_labels is not None

    assert len(train_data) + len(test_data) == \
           len(train_labels) + len(test_labels) == len(data)
    # Disable false positive: both array and dataframe have "shape" member
    # pylint: disable=maybe-no-member
    assert train_data.shape[1] == test_data.shape[1] == data.shape[1]


@pytest.mark.parametrize("data_labels",
                         ["data_and_labels_arrays",
                          "data_and_labels_dataframes"])
def test_wrap_labeled_database_no_shuffle(data_labels, request):
    """Checks the creation of a labeled database with no shuffle."""
    data, labels = request.getfixturevalue(data_labels)
    database = WrapLabeledDatabase(data, labels)
    train_data, train_labels, test_data, test_labels = database.load_data(shuffle=False)

    np.testing.assert_array_equal(np.concatenate((train_data, test_data)), data)
    np.testing.assert_array_equal(np.concatenate((train_labels, test_labels)), labels)


@pytest.mark.parametrize("data_labels",
                         ["data_and_labels_arrays",
                          "data_and_labels_dataframes"])
def test_split_train_test(data_labels, request):
    """ Splits the dataset into train and test sets.

    The same test is run both on arrays and dataframes.
    """
    data, labels = request.getfixturevalue(data_labels)
    train_proportion = 0.8
    train_size = round(len(data) * train_proportion)

    train_data, train_labels, \
        test_data, test_labels = \
        split_train_test(data, labels, train_proportion=train_proportion)

    if isinstance(data, np.ndarray):
        assert np.array_equal(train_data, data[:train_size])
        assert np.array_equal(train_labels, labels[:train_size])
        assert np.array_equal(test_data, data[train_size:])
        assert np.array_equal(test_labels, labels[train_size:])
        assert np.array_equal(np.concatenate((train_data, test_data)), data)
        assert np.array_equal(np.concatenate((train_labels, test_labels)), labels)

    if isinstance(data, (pd.DataFrame, pd.Series)):
        assert train_data.equals(data.iloc[:train_size])
        assert train_labels.equals(labels.iloc[:train_size])
        assert test_data.equals(data.iloc[train_size:])
        assert test_labels.equals(labels.iloc[train_size:])
        assert data.equals(pd.concat((train_data, test_data)))
        assert labels.equals(pd.concat((train_labels, test_labels)))

    # Test boundaries: All data assigned to train
    train_proportion = 1
    train_data, train_labels, test_data, test_labels = \
        split_train_test(data, labels, train_proportion=train_proportion)
    np.testing.assert_array_equal(train_data, data)
    np.testing.assert_array_equal(train_labels, labels)
    assert len(test_data) == 0
    assert len(test_labels) == 0

    # Test boundaries: All data assigned to test
    train_proportion = 0
    train_data, train_labels, test_data, test_labels = \
        split_train_test(data, labels, train_proportion=train_proportion)
    np.testing.assert_array_equal(test_data, data)
    np.testing.assert_array_equal(test_labels, labels)
    assert len(train_data) == 0
    assert len(train_labels) == 0


def test_shuffle_rows_numpy_array(data_and_labels_arrays):
    """Checks that the rows in an array-like object are randomly shuffled."""
    data, labels = data_and_labels_arrays
    shuffled_data, shuffled_labels = shuffle_rows(data, labels)

    np.testing.assert_array_equal(np.sort(data.ravel()),
                                  np.sort(shuffled_data.ravel()))
    np.testing.assert_array_equal(np.sort(labels.ravel()),
                                  np.sort(shuffled_labels.ravel()))

    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                             data, shuffled_data)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                             labels, shuffled_labels)


def test_shuffle_rows_pandas_dataframe(data_and_labels_dataframes):
    """Checks that the rows in dataframe are randomly shuffled."""
    data, labels = data_and_labels_dataframes
    shuffled_data, shuffled_labels = shuffle_rows(data, labels)

    np.testing.assert_array_equal(data.sort_index(),
                                  shuffled_data.sort_index())
    np.testing.assert_array_equal(labels.sort_index(),
                                  shuffled_labels.sort_index())

    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                             data, shuffled_data)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                             labels, shuffled_labels)


def test_shuffle_rows_wrong_inputs(data_and_labels_arrays):
    """Raises an exception if the data input are not in one of the allowed formats.

    Input data must be either an array-like object or a dataframe.
    """
    data, labels = data_and_labels_arrays
    labels = list(labels)

    with pytest.raises(TypeError):
        shuffle_rows(data, labels)


def test_arrays_wrong_length(data_and_labels_arrays):
    """Raises an exception if the input array-like objects are not of the same length."""
    data, labels = data_and_labels_arrays

    with pytest.raises(AssertionError):
        shuffle_rows(data, labels[0:-1])


@pytest.mark.parametrize("data_labels",
                         ["data_and_labels_arrays",
                          "data_and_labels_dataframes"])
def test_vertical_split_default_values(data_labels, request):
    """Checks the vertical split of a centralized database.

    The default values are used. It checks that if concatenating the split
    chunks, the original centralized dataset is recovered exactly.
    Also checks that different number of columns are assigned to the chunks.
    """
    data, labels = request.getfixturevalue(data_labels)
    n_samples, n_features = data.shape
    train_proportion = 0.8
    test_size = round(len(data) * (1 - train_proportion))

    # Default values:
    train_data, train_labels, test_data, test_labels = \
        vertical_split(data, labels)

    assert np.concatenate(train_data, axis=1).shape[1] == n_features
    assert np.concatenate(
        [train_data[0], test_data[0]], axis=0).shape[0] == n_samples
    for i, data_chunk in enumerate(train_data):
        assert data_chunk.shape[0] == len(train_labels) == len(data) - test_size
        assert test_data[i].shape[0] == len(test_labels) == test_size


@pytest.mark.parametrize("data_labels",
                         ["data_and_labels_arrays",
                          "data_and_labels_dataframes"])
def test_vertical_split_equal_number_columns(data_labels, request):
    """Checks the vertical split of a centralized database.

    It checks that, if requested by the user, the same number of columns
    are assigned to each chunk.
    """
    data, labels = request.getfixturevalue(data_labels)

    n_runs = 5
    shapes_equal_train = []
    shapes_equal_test = []
    for _ in range(n_runs):
        train_data, _, test_data, _ = \
            vertical_split(
                data, labels, indices_or_sections=3)
        shapes_chunks = np.array([chunk.shape == train_data[0].shape
                                  for chunk in train_data])
        shapes_equal_train.append(shapes_chunks.all())
        shapes_chunks = np.array([chunk.shape == test_data[0].shape
                                  for chunk in test_data])
        shapes_equal_test.append(shapes_chunks.all())
    assert np.array(shapes_equal_train).all()
    assert np.array(shapes_equal_test).all()


@pytest.mark.parametrize("data_labels",
                         ["data_and_labels_arrays",
                          "data_and_labels_dataframes"])
def test_vertical_split_no_vertical_shuffle(data_labels, request):
    """Checks the vertical split of a centralized database.

    Checks that, if requested by the user, the columns and rows of the
    original centralized dataset are not randomly shuffled before the split.
    """
    data, labels = request.getfixturevalue(data_labels)
    train_proportion = 0.8
    test_size = round(len(data) * (1 - train_proportion))

    # No vertical shuffle:
    train_data, train_labels, test_data, test_labels = \
        vertical_split(data, labels, v_shuffle=False)
    if isinstance(data, np.ndarray):
        assert np.array_equal(np.concatenate(train_data, axis=1), data[:-test_size, :])
        assert np.array_equal(np.concatenate(test_data, axis=1), data[-test_size:, :])
        assert np.array_equal(np.concatenate([train_labels, test_labels]), labels)
    elif isinstance(data, pd.DataFrame):
        assert pd.concat(train_data, axis=1).equals(data.iloc[:-test_size, :])
        assert pd.concat(test_data, axis=1).equals(data.iloc[-test_size:, :])
        assert pd.concat([train_labels, test_labels]).equals(labels)


def test_vertical_split_wrong_input_type():
    """Raises an exception if the data input are not in one of the allowed formats."""
    data = list(np.random.rand(60).reshape([10, -1]))
    labels = np.random.rand(10)

    with pytest.raises(TypeError):
        vertical_split(data, labels)
