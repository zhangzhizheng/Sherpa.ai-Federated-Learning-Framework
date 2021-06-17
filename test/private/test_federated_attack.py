import numpy as np

from shfl.private.federated_operation import FederatedData
from shfl.private.data import LabeledData
from shfl.private.utils import shuffle_node_query
from shfl.private.federated_attack import FederatedPoisoningDataAttack
from shfl.private.data import UnprotectedAccess


def test_shuffle_node(data_and_labels):
    """Checks that the labels in a node are correctly shuffled."""
    labeled_data = LabeledData(*data_and_labels)
    federated_data = FederatedData()
    federated_data.append_data_node(labeled_data)
    federated_data.apply_data_transformation(shuffle_node_query)
    federated_data.configure_data_access(UnprotectedAccess())

    assert (not np.array_equal(federated_data[0].query().label,
                               data_and_labels[1]))


def test_federated_poisoning_attack(data_and_labels):
    """Checks that the labels in a set of federated nodes are correctly shuffled."""
    num_nodes = 10
    federated_data = FederatedData()

    list_labels = []
    for _ in range(num_nodes):
        data = np.random.rand(*data_and_labels[0].shape)
        label = np.random.randint(0, 10, size=data_and_labels[1].shape)
        list_labels.append(label)
        labeled_data = LabeledData(data, label)
        federated_data.append_data_node(labeled_data)

    percentage = 10
    simple_attack = FederatedPoisoningDataAttack(percentage=percentage)
    simple_attack.apply_attack(federated_data=federated_data)

    adversaries_indices = simple_attack.adversaries

    federated_data.configure_data_access(UnprotectedAccess())
    for node, index in zip(federated_data, range(num_nodes)):
        if index in adversaries_indices:
            assert not np.array_equal(node.query().label, list_labels[index])
        else:
            assert np.array_equal(node.query().label, list_labels[index])
