import random
import numpy as np

from shfl.data_base.data_base import WrapLabeledDatabase
from shfl.data_distribution.data_distribution_non_iid import NonIidDataDistribution
from shfl.private.utils import unprotected_query


def test_make_data_federated(data_and_labels_arrays):
    """Checks that the clients contain non-IID labels."""
    random.seed(123)
    np.random.seed(123)

    data_base = WrapLabeledDatabase(*data_and_labels_arrays)
    data_base.load_data()
    train_data, train_labels, _, _ = data_base.data
    data_distribution = NonIidDataDistribution(data_base)
    weights = [0.5, 0.25, 0.25]
    percent = 60

    num_nodes = 3
    federated_data, federated_labels = \
        data_distribution.make_data_federated(train_data, train_labels,
                                              percent=percent,
                                              num_nodes=num_nodes,
                                              weights=weights)

    idx = []
    for sample in np.concatenate(federated_data):
        idx.append(np.where((sample == train_data).all(axis=1))[0][0])

    for node_data, weight in zip(federated_data, weights):
        assert np.isclose(node_data.shape[0],
                          weight * percent * train_data.shape[0] / 100,
                          rtol=1)

    assert num_nodes == len(federated_data) == len(federated_labels)
    assert (np.sort(np.concatenate(federated_data).ravel()) ==
            np.sort(train_data[idx, ].ravel())).all()
    assert (np.sort(np.concatenate(federated_labels), 0) ==
            np.sort(train_labels[idx], 0)).all()


def test_make_data_federated_wrong_weights(data_and_labels_arrays):
    """Checks that when wrong weights are used, these are re-normalized internally."""
    random.seed(123)
    np.random.seed(123)

    data_base = WrapLabeledDatabase(*data_and_labels_arrays)
    data_base.load_data()
    train_data, train_labels, _, _ = data_base.data
    data_distribution = NonIidDataDistribution(data_base)

    num_nodes = 3
    wrong_weights = [0.5, 0.55, 0.1]
    percent = 60
    federated_data, federated_labels = \
        data_distribution.make_data_federated(train_data, train_labels,
                                              percent=percent, num_nodes=num_nodes,
                                              weights=wrong_weights,
                                              sampling='without_replacement')

    idx = [np.where((data == train_data).all(axis=1))[0][0]
           for data in np.concatenate(federated_data)]

    for node_data, weight in zip(federated_data, wrong_weights / np.sum(wrong_weights)):
        assert np.isclose(node_data.shape[0],
                          weight * percent * train_data.shape[0] / 100,
                          rtol=1)

    assert num_nodes == len(federated_data) == len(federated_labels)
    assert (np.sort(np.concatenate(federated_data).ravel()) ==
            np.sort(train_data[idx, ].ravel())).all()
    assert (np.sort(np.concatenate(federated_labels), 0) ==
            np.sort(train_labels[idx], 0)).all()


def test_get_federated_data(data_and_labels_arrays):
    """Checks that the non-IID data and labels are correctly returned."""
    data_base = WrapLabeledDatabase(*data_and_labels_arrays)
    data_base.load_data()
    data_distribution = NonIidDataDistribution(data_base)

    # Identifier and num nodes is checked in private test.
    # Percent and weight is checked in idd and no_idd test.
    federated_data, test_data, test_labels = \
        data_distribution.get_nodes_federation(num_nodes=4)

    federated_data.configure_data_access(unprotected_query)
    reference_federated_data = [node_data.query().data for node_data in federated_data]
    reference_federated_labels = [node_data.query().label for node_data in federated_data]

    centralized_data, centralized_labels, \
        reference_test_data, reference_test_labels = data_base.data

    idx = []
    for node in reference_federated_data:
        labels_node = []
        for data_base in node:
            assert data_base in centralized_data
            idx.append(np.where((data_base == centralized_data).all(axis=1))[0][0])
            labels_node.append(centralized_labels[idx[-1]].argmax(axis=-1))

    assert np.array_equal(centralized_data[idx, ].ravel(),
                          np.concatenate(reference_federated_data).ravel())
    assert np.array_equal(centralized_labels[idx, ],
                          np.concatenate(reference_federated_labels))
    assert np.array_equal(test_data.ravel(), reference_test_data.ravel())
    assert np.array_equal(test_labels, reference_test_labels)


def test_make_federated_data_categorical(data_and_labels_arrays):
    """Checks whether categorical labels are correctly distributed."""
    data_base = WrapLabeledDatabase(*data_and_labels_arrays)
    data_base.load_data()
    data_distribution = NonIidDataDistribution(data_base)
    train_data, _ = data_base.train
    n_targets = 3
    train_labels = np.random.randint(0, 2, size=(len(train_data), n_targets))
    num_nodes = 5
    percent = 60
    _, federated_labels = data_distribution.make_data_federated(train_data,
                                                                train_labels,
                                                                percent=percent,
                                                                num_nodes=num_nodes)

    for labels in federated_labels:
        assert labels.shape[1] == n_targets
