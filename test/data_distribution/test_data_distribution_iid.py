import numpy as np
import pandas as pd
import pytest

from shfl.data_base.data_base import WrapLabeledDatabase
from shfl.data_distribution.data_distribution_iid import IidDataDistribution


def get_federated_data_info(data, labels):
    """Sets the parameters for creating the federated data."""
    data_base = WrapLabeledDatabase(data, labels)
    data_base.load_data()
    train_data, train_label, _, _ = data_base.data
    data_distribution = IidDataDistribution(data_base)

    num_nodes = 3
    percent = 60
    weights = [0.5, 0.25, 0.25]

    return train_data, train_label, data_distribution, \
        num_nodes, percent, weights


def test_make_data_federated_array_without_replacement(data_and_labels_arrays):
    """Checks that the data distribution among client is IID without replacement.

    The input format is an array-like object.
    """
    train_data, train_label, data_distribution, \
        num_nodes, percent, weights = get_federated_data_info(*data_and_labels_arrays)

    federated_data, federated_label = \
        data_distribution.make_data_federated(train_data, train_label,
                                              percent, num_nodes, weights)

    centralized_data = np.concatenate(federated_data)
    centralized_label = np.concatenate(federated_label)

    idx = [np.where((data == train_data).all(axis=1))[0][0]
           for data in centralized_data]

    for i, weight in enumerate(weights):
        assert np.isclose(federated_data[i].shape[0],
                          weight * percent * train_data.shape[0] / 100,
                          rtol=1)

    assert centralized_data.shape[0] == int(percent * train_data.shape[0] / 100)
    assert num_nodes == len(federated_data) == len(federated_label)
    assert (np.sort(centralized_data.ravel()) == np.sort(train_data[idx, ].ravel())).all()
    assert (np.sort(centralized_label, 0) == np.sort(train_label[idx], 0)).all()


def test_make_data_federated_array_with_replacement(data_and_labels_arrays):
    """Checks that the data distribution among client is IID.

    The input format is an array-like object, sampling with replacement.
    """
    train_data, train_label, data_distribution, \
        num_nodes, percent, weights = get_federated_data_info(*data_and_labels_arrays)

    federated_data, federated_label = \
        data_distribution.make_data_federated(train_data, train_label,
                                              percent, num_nodes, weights,
                                              sampling="with_replacement")

    centralized_data = np.concatenate(federated_data)
    centralized_label = np.concatenate(federated_label)

    idx = [np.where((data == train_data).all(axis=1))[0][0]
           for data in centralized_data]

    for i, weight in enumerate(weights):
        assert np.isclose(federated_data[i].shape[0],
                          weight * percent * train_data.shape[0] / 100,
                          rtol=1)

    assert np.isclose(centralized_data.shape[0], percent * train_data.shape[0] / 100, rtol=1)
    assert num_nodes == len(federated_data) == len(federated_label)
    assert (np.sort(centralized_data.ravel()) == np.sort(train_data[idx, ].ravel())).all()
    assert (np.sort(centralized_label, 0) == np.sort(train_label[idx], 0)).all()


def test_make_data_federated_pandas_without_replacement(data_and_labels_arrays):
    """Checks that the data distribution among client is IID.

    The input format is a Pandas dataframe, sampling without replacement.
    """
    train_data, train_label, data_distribution, num_nodes, percent, weights = \
        get_federated_data_info(pd.DataFrame(data_and_labels_arrays[0]),
                                pd.DataFrame(data_and_labels_arrays[1]))

    federated_data, federated_label = \
        data_distribution.make_data_federated(train_data, train_label,
                                              percent, num_nodes, weights)

    centralized_data = pd.concat(federated_data)
    centralized_labels = pd.concat(federated_label)

    for i, weight in enumerate(weights):
        assert np.isclose(federated_data[i].shape[0],
               weight * int(percent * train_data.shape[0] / 100),
                          rtol=1)

    assert np.isclose(centralized_data.shape[0], percent * train_data.shape[0] / 100, rtol=1)
    assert num_nodes == len(federated_data) == len(federated_label)
    pd.testing.assert_frame_equal(centralized_data,
                                  train_data.loc[centralized_data.index.values])
    pd.testing.assert_frame_equal(centralized_labels,
                                  train_label.loc[centralized_data.index.values])


def test_make_data_federated_pandas_with_replacement(data_and_labels_arrays):
    """Checks that the data distribution among client is IID.

    The input format is a Pandas dataframe, sampling with replacement.
    """
    train_data, train_label, data_distribution, num_nodes, percent, weights = \
        get_federated_data_info(pd.DataFrame(data_and_labels_arrays[0]),
                                pd.DataFrame(data_and_labels_arrays[1]))

    federated_data, federated_label = \
        data_distribution.make_data_federated(train_data, train_label,
                                              percent, num_nodes, weights,
                                              sampling="with_replacement")
    centralized_data = pd.concat(federated_data)
    centralized_labels = pd.concat(federated_label)

    for i, weight in enumerate(weights):
        assert np.isclose(federated_data[i].shape[0],
                          weight * int(percent * train_data.shape[0] / 100),
                          rtol=1)

    assert np.isclose(centralized_data.shape[0], percent * train_data.shape[0] / 100, rtol=1)
    assert num_nodes == len(federated_data) == len(federated_label)
    pd.testing.assert_frame_equal(centralized_data,
                                  train_data.loc[centralized_data.index.values])
    pd.testing.assert_frame_equal(centralized_labels,
                                  train_label.loc[centralized_data.index.values])


@pytest.mark.parametrize("input_weights", [[0.5, 0.5, 0.5], None])
def test_make_data_federated_wrong_or_none_weights(input_weights, data_and_labels_arrays):
    """Checks that when using wrong weights, these are correctly normalized internally"""
    train_data, train_label, data_distribution, \
        num_nodes, percent, _ = get_federated_data_info(*data_and_labels_arrays)

    federated_data, federated_label = \
        data_distribution.make_data_federated(train_data, train_label,
                                              percent, num_nodes, input_weights)

    centralized_data = np.concatenate(federated_data)
    centralized_labels = np.concatenate(federated_label)

    idx = [np.where((data == train_data).all(axis=1))[0][0]
           for data in centralized_data]

    for node_data in federated_data:
        assert np.isclose(node_data.shape[0],
                          1 / num_nodes * percent * train_data.shape[0] / 100,
                          rtol=1)

    assert np.isclose(centralized_data.shape[0], percent * train_data.shape[0] / 100, rtol=1)
    assert num_nodes == len(federated_data) == len(federated_label)
    assert (np.sort(centralized_data.ravel()) == np.sort(train_data[idx, ].ravel())).all()
    assert (np.sort(centralized_labels, 0) == np.sort(train_label[idx], 0)).all()
