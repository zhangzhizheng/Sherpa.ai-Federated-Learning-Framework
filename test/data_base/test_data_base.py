import numpy as np
import pandas as pd
import pytest

import shfl.data_base.data_base
from shfl.data_base.data_base import DataBase
from shfl.data_base.data_base import LabeledDatabase
from shfl.data_base.data_base import shuffle_rows
from shfl.data_base.data_base import split_train_test
from shfl.data_base.data_base import vertical_split


@pytest.fixture
def data_and_labels_arrays():
    data = np.random.rand(60, 12)
    labels = np.random.randint(0, 2, size=(60,))

    return data, labels


@pytest.fixture
def data_and_labels_dataframes(data_and_labels_arrays):
    data, labels = data_and_labels_arrays
    data = pd.DataFrame(data)
    labels = pd.Series(labels)

    return data, labels


class TestDataBase(DataBase):
    def __init__(self):
        super(TestDataBase, self).__init__()

    def load_data(self):
        self._train_data = np.random.rand(50).reshape([10, 5])
        self._test_data = np.random.rand(50).reshape([10, 5])
        self._train_labels = np.random.randint(10)
        self._test_labels = np.random.randint(10)


class TestDataBasePandas(DataBase):
    def __init__(self):
        super(TestDataBasePandas, self).__init__()

    def load_data(self):
        self._train_data = pd.DataFrame(np.random.rand(50).reshape([10, 5]))
        self._test_data = pd.DataFrame(np.random.rand(50).reshape([10, 5]))
        self._train_labels = pd.Series(np.random.randint(10))
        self._test_labels = pd.Series(np.random.randint(10))


@pytest.mark.parametrize("data_labels",
                         ["data_and_labels_arrays",
                          "data_and_labels_dataframes"])
def test_split_train_test(data_labels, request):
    data, labels = request.getfixturevalue(data_labels)
    train_proportion = 0.8
    train_size = round(len(data) * train_proportion)

    train_data, train_labels, \
        test_data, test_labels = \
        shfl.data_base.data_base.split_train_test(data, labels, train_proportion)

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
    data, labels = data_and_labels_arrays
    labels = pd.Series(labels)

    with pytest.raises(TypeError):
        shuffle_rows(data, labels)


@pytest.mark.parametrize("data_labels",
                         ["data_and_labels_arrays",
                          "data_and_labels_dataframes"])
def test_labeled_database(data_labels, request):
    data, labels = request.getfixturevalue(data_labels)
    database = LabeledDatabase(data, labels)
    train_data, train_labels, test_data, test_labels = database.load_data()

    assert train_data is not None
    assert train_labels is not None
    assert test_data is not None
    assert test_labels is not None

    assert len(train_data) + len(test_data) == \
           len(train_labels) + len(test_labels) == len(data)
    assert train_data.shape[1] == test_data.shape[1] == data.shape[1]


@pytest.mark.parametrize("data_labels",
                         ["data_and_labels_arrays",
                          "data_and_labels_dataframes"])
def test_vertical_split(data_labels, request):
    data, labels = request.getfixturevalue(data_labels)
    n_samples, n_features = data.shape
    train_percentage = 0.8
    dim = round(len(data) * (1 - train_percentage))

    # Default values:
    train_data, train_labels, test_data, test_labels = \
        vertical_split(data, labels)

    assert np.concatenate(train_data, axis=1).shape[1] == n_features
    assert np.concatenate(
        [train_data[0], test_data[0]], axis=0).shape[0] == n_samples
    for i in range(len(train_data)):
        assert train_data[i].shape[0] == len(train_labels) == len(data) - dim
        assert test_data[i].shape[0] == len(test_labels) == dim

    # Random split: different number of columns in different chunks
    n_runs = 5
    shapes_equal_train = []
    shapes_equal_test = []
    for i_run in range(n_runs):
        train_data, _, test_data, _ = \
            vertical_split(data, labels)
        shapes_equal_train.append(train_data[0].shape == train_data[1].shape)
        shapes_equal_test.append(test_data[0].shape == test_data[1].shape)
    assert not np.array(shapes_equal_train).all()
    assert not np.array(shapes_equal_test).all()

    # Equal size split: same number of columns in chunks
    n_runs = 5
    shapes_equal_train = []
    shapes_equal_test = []
    for i_run in range(n_runs):
        train_data, _, test_data, _ = \
            vertical_split(
                data, labels, indices_or_sections=3, equal_size=True)
        shapes_chunks = np.array([chunk.shape == train_data[0].shape
                                  for chunk in train_data])
        shapes_equal_train.append(shapes_chunks.all())
        shapes_chunks = np.array([chunk.shape == test_data[0].shape
                                  for chunk in test_data])
        shapes_equal_test.append(shapes_chunks.all())
    assert np.array(shapes_equal_train).all()
    assert np.array(shapes_equal_test).all()

    # Rise Value error if unable to split in same number of columns in chunks
    with pytest.raises(ValueError):
        vertical_split(data, labels, indices_or_sections=5, equal_size=True)

    # No vertical/horizontal shuffle:
    train_data, train_labels, test_data, test_labels = \
        vertical_split(data, labels, v_shuffle=False, h_shuffle=False)
    if isinstance(data, np.ndarray):
        assert np.array_equal(np.concatenate(train_data, axis=1), data[:-dim, :])
        assert np.array_equal(np.concatenate(test_data, axis=1), data[-dim:, :])
        assert np.array_equal(np.concatenate([train_labels, test_labels]), labels)
    elif isinstance(data, pd.DataFrame):
        assert pd.concat(train_data, axis=1).equals(data.iloc[:-dim, :])
        assert pd.concat(test_data, axis=1).equals(data.iloc[-dim:, :])
        assert pd.concat([train_labels, test_labels]).equals(labels)


def test_vertical_split_wrong_input_type():
    data = list(np.random.rand(60).reshape([10, -1]))
    labels = np.random.rand(10)

    with pytest.raises(TypeError):
        vertical_split(data, labels)
