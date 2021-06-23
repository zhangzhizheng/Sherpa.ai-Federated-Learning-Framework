import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution_iid import IidDataDistribution


class DataBaseArrayTest(DataBase):
    """Creates an array-like database with train and test sets of random values."""
    def load_data(self):
        self._train_data = np.random.rand(200).reshape([40, 5])
        self._test_data = np.random.rand(200).reshape([40, 5])
        self._train_labels = tf.keras.utils.to_categorical(
            np.random.randint(0, 10, 40))
        self._test_labels = tf.keras.utils.to_categorical(
            np.random.randint(0, 10, 40))


class DataBasePandasTest(DataBase):
    """Creates a Pandas database with train, test and validation sets of random values."""
    def load_data(self):
        self._train_data = pd.DataFrame(np.random.rand(200).reshape([40, 5]))
        self._test_data = pd.DataFrame(np.random.rand(200).reshape([40, 5]))
        self._train_labels = pd.DataFrame(
            tf.keras.utils.to_categorical(np.random.randint(0, 10, 40)))
        self._test_labels = pd.DataFrame(
            tf.keras.utils.to_categorical(np.random.randint(0, 10, 40)))


@pytest.fixture(name="federated_data_info")
def fixture_federated_data_info(request):
    """Sets the parameters for creating the federated data."""
    data_base = request.param()
    data_base.load_data()
    train_data, train_label, _, _ = data_base.data
    data_distribution = IidDataDistribution(data_base)

    num_nodes = 3
    percent = 60
    weights = [0.5, 0.25, 0.25]

    return train_data, train_label, data_distribution, \
        num_nodes, percent, weights


@pytest.mark.parametrize("federated_data_info", [DataBaseArrayTest], indirect=True)
def test_make_data_federated_array_without_replacement(federated_data_info):
    """Checks that the data distribution among client is IID without replacement.

    The input format is an array-like object.
    """
    train_data, train_label, data_distribution, \
        num_nodes, percent, weights = federated_data_info

    federated_data, federated_label = \
        data_distribution.make_data_federated(train_data, train_label,
                                              percent=percent, num_nodes=num_nodes, weights=weights)

    centralized_data = np.concatenate(federated_data)
    centralized_label = np.concatenate(federated_label)

    idx = [np.where((data == train_data).all(axis=1))[0][0]
           for data in centralized_data]

    for i, weight in enumerate(weights):
        assert federated_data[i].shape[0] == \
               int(weight * int(percent * train_data.shape[0] / 100))

    assert centralized_data.shape[0] == int(percent * train_data.shape[0] / 100)
    assert num_nodes == len(federated_data) == len(federated_label)
    assert (np.sort(centralized_data.ravel()) == np.sort(train_data[idx, ].ravel())).all()
    assert (np.sort(centralized_label, 0) == np.sort(train_label[idx], 0)).all()


@pytest.mark.parametrize("federated_data_info", [DataBaseArrayTest], indirect=True)
def test_make_data_federated_array_with_replacement(federated_data_info):
    """Checks that the data distribution among client is IID.

    The input format is an array-like object, sampling with replacement.
    """
    train_data, train_label, data_distribution, \
        num_nodes, percent, weights = federated_data_info

    federated_data, federated_label = \
        data_distribution.make_data_federated(train_data, train_label,
                                              percent=percent, num_nodes=num_nodes,
                                              weights=weights, sampling="with_replacement")

    centralized_data = np.concatenate(federated_data)
    centralized_label = np.concatenate(federated_label)

    idx = [np.where((data == train_data).all(axis=1))[0][0]
           for data in centralized_data]

    for i, weight in enumerate(weights):
        assert federated_data[i].shape[0] == \
               int(weight * int(percent * train_data.shape[0] / 100))

    assert centralized_data.shape[0] == int(percent * train_data.shape[0] / 100)
    assert num_nodes == len(federated_data) == len(federated_label)
    assert (np.sort(centralized_data.ravel()) == np.sort(train_data[idx, ].ravel())).all()
    assert (np.sort(centralized_label, 0) == np.sort(train_label[idx], 0)).all()


@pytest.mark.parametrize("federated_data_info", [DataBasePandasTest], indirect=True)
def test_make_data_federated_pandas_without_replacement(federated_data_info):
    """Checks that the data distribution among client is IID.

    The input format is a Pandas dataframe, sampling without replacement.
    """
    train_data, train_label, data_distribution, \
        num_nodes, percent, weights = federated_data_info

    federated_data, federated_label = \
        data_distribution.make_data_federated(train_data, train_label,
                                              percent=percent, num_nodes=num_nodes, weights=weights)

    centralized_data = pd.concat(federated_data)
    centralized_labels = pd.concat(federated_label)

    for i, weight in enumerate(weights):
        assert federated_data[i].shape[0] == \
               int(weight * int(percent * train_data.shape[0] / 100))

    assert centralized_data.shape[0] == int(percent * train_data.shape[0] / 100)
    assert num_nodes == len(federated_data) == len(federated_label)
    pd.testing.assert_frame_equal(centralized_data,
                                  train_data.iloc[centralized_data.index.values])
    pd.testing.assert_frame_equal(centralized_labels,
                                  train_label.iloc[centralized_data.index.values])


@pytest.mark.parametrize("federated_data_info", [DataBasePandasTest], indirect=True)
def test_make_data_federated_pandas_with_replacement(federated_data_info):
    """Checks that the data distribution among client is IID.

    The input format is a Pandas dataframe, sampling with replacement.
    """
    train_data, train_label, data_distribution, \
        num_nodes, percent, weights = federated_data_info

    federated_data, federated_label = \
        data_distribution.make_data_federated(train_data, train_label,
                                              percent=percent, num_nodes=num_nodes, weights=weights,
                                              sampling="with_replacement")
    centralized_data = pd.concat(federated_data)
    centralized_labels = pd.concat(federated_label)

    for i, weight in enumerate(weights):
        assert federated_data[i].shape[0] == \
               int(weight * int(percent * train_data.shape[0] / 100))

    assert centralized_data.shape[0] == int(percent * train_data.shape[0] / 100)
    assert num_nodes == len(federated_data) == len(federated_label)
    pd.testing.assert_frame_equal(centralized_data,
                                  train_data.iloc[centralized_data.index.values])
    pd.testing.assert_frame_equal(centralized_labels,
                                  train_label.iloc[centralized_data.index.values])


@pytest.mark.parametrize("federated_data_info", [DataBaseArrayTest], indirect=True)
def test_make_data_federated_wrong_weights(federated_data_info):
    """Checks that when using wrong weights, these are correctly normalized internally"""
    train_data, train_label, data_distribution, \
        num_nodes, percent, _ = federated_data_info

    wrong_weights = np.array([0.5, 0.5, 0.5])
    federated_data, federated_label = \
        data_distribution.make_data_federated(train_data, train_label,
                                              percent=percent, num_nodes=num_nodes, weights=wrong_weights)

    centralized_data = np.concatenate(federated_data)
    centralized_labels = np.concatenate(federated_label)

    idx = [np.where((data == train_data).all(axis=1))[0][0]
           for data in centralized_data]

    for node_data, weight in zip(federated_data, wrong_weights):
        assert node_data.shape[0] == \
               int(weight / sum(wrong_weights) * int(percent * train_data.shape[0] / 100))

    assert centralized_data.shape[0] == int(percent * train_data.shape[0] / 100)
    assert num_nodes == len(federated_data) == len(federated_label)
    assert (np.sort(centralized_data.ravel()) == np.sort(train_data[idx, ].ravel())).all()
    assert (np.sort(centralized_labels, 0) == np.sort(train_label[idx], 0)).all()
