import numpy as np
import pandas as pd
import pytest

import shfl.data_base.data_base
from shfl.data_base.data_base import DataBase
from shfl.data_base.data_base import LabeledDatabase
from shfl.data_base.data_base import vertical_split


class TestDataBase(DataBase):
    def __init__(self):
        super(TestDataBase, self).__init__()

    def load_data(self):
        self._train_data = np.random.rand(50).reshape([10, 5])
        self._test_data = np.random.rand(50).reshape([10, 5])
        self._train_labels = np.random.rand(10)
        self._test_labels = np.random.rand(10)


class TestDataBasePandas(DataBase):
    def __init__(self):
        super(TestDataBasePandas, self).__init__()

    def load_data(self):
        self._train_data = pd.DataFrame(np.random.rand(50).reshape([10, 5]))
        self._test_data = pd.DataFrame(np.random.rand(50).reshape([10, 5]))
        self._train_labels = pd.Series(np.random.rand(10))
        self._test_labels = pd.Series(np.random.rand(10))


@pytest.mark.parametrize("data, labels",
                         [(np.random.rand(50).reshape([10, -1]),
                           np.random.rand(10)),
                          (pd.DataFrame(np.random.rand(50).reshape([10, -1])),
                           pd.Series(np.random.rand(10)))])
def test_split_train_test(data, labels):
    data = np.random.rand(50).reshape([10, -1])
    labels = np.random.rand(10)
    train_percentage = 0.8
    dim = round(len(data) * (1 - train_percentage))

    rest_data, rest_labels, \
        validation_data, validation_labels = \
        shfl.data_base.data_base.split_train_test(data, labels, train_percentage)

    if isinstance(data, np.ndarray):
        ndata = np.concatenate([rest_data, validation_data])
        nlabels = np.concatenate([rest_labels, validation_labels])
    else:  # Dataframe
        ndata = pd.concat([rest_data, validation_data])
        nlabels = pd.concat([rest_labels, validation_labels])

    data_ravel = np.sort(data.ravel())
    ndata_ravel = np.sort(ndata.ravel())

    assert np.array_equal(data_ravel, ndata_ravel)
    assert np.array_equal(np.sort(labels), np.sort(nlabels))
    assert rest_data.shape[0] == data.shape[0]-dim
    assert rest_labels.shape[0] == labels.shape[0]-dim
    assert validation_data.shape[0] == dim
    assert validation_labels.shape[0] == dim

    # No shuffle:
    rest_data, rest_labels, validation_data, validation_labels = \
        shfl.data_base.data_base.split_train_test(
            data, labels, train_percentage, shuffle=False)
    if isinstance(data, np.ndarray):
        assert np.array_equal(rest_data, data[:-dim, :])
        assert np.array_equal(rest_labels, labels[:-dim])
        assert np.array_equal(validation_data, data[-dim:, :])
        assert np.array_equal(validation_labels, labels[-dim:])
    else:  # Dataframe
        assert rest_data.equals(data.iloc[:-dim, :])
        assert rest_labels.equals(labels.iloc[:-dim])
        assert validation_data.equals(data.iloc[-dim:, :])
        assert validation_labels.equals(labels.iloc[-dim:])


def test_data_base_shuffle_elements():
    data = TestDataBase()
    data.load_data()

    train_data_b, train_labels_b, test_data_b, test_labels_b = data.data

    data.shuffle()

    train_data_a, train_labels_a, test_data_a, test_labels_a = data.data

    train_data_b = np.sort(train_data_b.ravel())
    train_data_a = np.sort(train_data_a.ravel())
    assert np.array_equal(train_data_b, train_data_a)

    test_data_b = np.sort(test_data_b.ravel())
    test_data_a = np.sort(test_data_a.ravel())
    assert np.array_equal(test_data_b, test_data_a)

    assert np.array_equal(np.sort(train_labels_b), np.sort(train_labels_a))
    assert np.array_equal(np.sort(test_labels_b), np.sort(test_labels_a))


def test_data_base_shuffle_elements_pandas():
    data = TestDataBasePandas()
    data.load_data()

    train_data_b, train_labels_b, test_data_b, test_labels_b = data.data

    data.shuffle()

    train_data_a, train_labels_a, test_data_a, test_labels_a = data.data

    train_data_b = train_data_b.sort_index()
    train_data_a = train_data_a.sort_index()
    assert np.array_equal(train_data_b, train_data_a)

    test_data_b = test_data_b.sort_index()
    test_data_a = test_data_a.sort_index()
    assert np.array_equal(test_data_b, test_data_a)

    assert np.array_equal(train_labels_b.sort_index(), train_labels_a.sort_index())
    assert np.array_equal(test_labels_b.sort_index(), test_labels_a.sort_index())


def test_data_base_shuffle_correct():
    data = TestDataBase()
    data.load_data()

    train_data_b, train_labels_b = data.train
    test_data_b, test_labels_b = data.test

    data.shuffle()

    train_data_a, train_labels_a = data.train
    test_data_a, test_labels_a = data.test

    assert (train_data_b == train_data_a).all() == False
    assert (test_data_b == test_data_a).all() == False


def test_data_base_shuffle_correct_pandas():
    data = TestDataBasePandas()
    data.load_data()

    train_data_b, train_labels_b = data.train
    test_data_b, test_labels_b = data.test

    data.shuffle()

    train_data_a, train_labels_a = data.train
    test_data_a, test_labels_a = data.test

    assert (train_data_b.to_numpy() == train_data_a.to_numpy()).all() == False
    assert (test_data_b.to_numpy() == test_data_a.to_numpy()).all() == False


def test_shuffle_wrong_call():
    data = TestDataBase()

    with pytest.raises(TypeError):
        data.shuffle()


def test_labeled_database():
    data = np.random.randint(low=0, high=100, size=100, dtype='l')
    labels = 10 + 2 * data + np.random.normal(loc=0.0, scale=10, size=len(data))
    database = LabeledDatabase(data, labels)
    loaded_data = database.load_data()

    assert loaded_data is not None
    assert len(loaded_data[1]) + len(loaded_data[3]) == len(data)

    # No train/test split:
    database = LabeledDatabase(data, labels, train_percentage=1)
    train_data, train_labels, test_data, test_labels = database.load_data()
    assert np.array_equal(train_data, data)
    assert np.array_equal(train_labels, labels)
    assert not test_data
    assert not test_labels


def test_split_wrong_type():
    data = np.random.rand(50).reshape([10, -1])
    labels = pd.Series(np.random.rand(10))
    train_percentage = 0.8

    with pytest.raises(TypeError):
        shfl.data_base.data_base.split_train_test(data, labels, train_percentage)


@pytest.mark.parametrize("data, labels",
                         [(np.random.rand(60).reshape([10, -1]),
                           np.random.rand(10)),
                          (pd.DataFrame(np.random.rand(60).reshape([10, -1])),
                           pd.Series(np.random.rand(10)))])
def test_vertical_split(data, labels):
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

    # No train/test split:
    train_data, train_labels, test_data, test_labels = \
        vertical_split(data, labels, train_percentage=1)
    assert np.concatenate(train_data, axis=1).shape == data.shape
    for i in range(len(train_data)):
        assert train_data[i].shape[0] == len(train_labels) == len(data)
        assert test_data is None
        assert test_labels is None

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