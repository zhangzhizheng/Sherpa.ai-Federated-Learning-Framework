import random
import numpy as np
import tensorflow as tf

from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution_non_iid import NonIidDataDistribution
from shfl.private.utils import unprotected_query


class DataBaseTest(DataBase):
    """Can array-like database with train and test sets of random values."""
    def load_data(self):
        self._train_data = np.random.rand(500).reshape([100, 5])
        self._test_data = np.random.rand(250).reshape([50, 5])
        self._train_labels = tf.keras.utils.to_categorical(np.random.randint(0, 3, 100))
        self._test_labels = tf.keras.utils.to_categorical(np.random.randint(0, 3, 50))


def test_make_data_federated():
    """Checks that the clients contain non-IID labels."""
    random.seed(123)
    np.random.seed(123)

    data_base = DataBaseTest()
    data_base.load_data()
    train_data, train_labels, _, _ = data_base.data
    data_distribution = NonIidDataDistribution(data_base)

    num_nodes = 3
    federated_data, federated_labels = \
        data_distribution.make_data_federated(train_data, train_labels, percent=60,
                                              num_nodes=num_nodes,
                                              weights=[0.5, 0.25, 0.25])

    idx = []
    for sample in np.concatenate(federated_data):
        idx.append(np.where((sample == train_data).all(axis=1))[0][0])

    seed_weights = [30, 15, 15]
    for node_data, weight in zip(federated_data, seed_weights):
        assert node_data.shape[0] == weight

    assert num_nodes == len(federated_data) == len(federated_labels)
    assert (np.sort(np.concatenate(federated_data).ravel()) ==
            np.sort(train_data[idx, ].ravel())).all()
    assert (np.sort(np.concatenate(federated_labels), 0) ==
            np.sort(train_labels[idx], 0)).all()


def test_make_data_federated_wrong_weights():
    """Checks that when wrong weights are used, these are re-normalized internally."""
    random.seed(123)
    np.random.seed(123)

    data_base = DataBaseTest()
    data_base.load_data()
    train_data, train_labels, _, _ = data_base.data
    data_distribution = NonIidDataDistribution(data_base)

    num_nodes = 3
    wrong_weights = [0.5, 0.55, 0.1]
    federated_data, federated_labels = \
        data_distribution.make_data_federated(train_data, train_labels,
                                              percent=60, num_nodes=num_nodes,
                                              weights=wrong_weights,
                                              sampling='without_replacement')

    all_data = np.concatenate(federated_data)
    all_label = np.concatenate(federated_labels)

    idx = [np.where((data == train_data).all(axis=1))[0][0]
           for data in all_data]

    seed_weights = [26, 28, 5]
    for node_data, weight in zip(federated_data, seed_weights):
        assert node_data.shape[0] == weight

    assert num_nodes == len(federated_data) == len(federated_labels)
    assert (np.sort(all_data.ravel()) == np.sort(train_data[idx, ].ravel())).all()
    assert (np.sort(all_label, 0) == np.sort(train_labels[idx], 0)).all()


def test_get_federated_data():
    """Checks that the non-IID data and labels are correctly returned."""
    data_base = DataBaseTest()
    data_base.load_data()
    data_distribution = NonIidDataDistribution(data_base)

    # Identifier and num nodes is checked in private test.
    # Percent and weight is checked in idd and no_idd test.
    num_nodes = 4
    federated_data, test_data, test_labels = \
        data_distribution.get_federated_data(num_nodes=num_nodes)

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
