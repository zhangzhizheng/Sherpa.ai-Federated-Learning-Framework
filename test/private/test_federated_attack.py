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
import random

from shfl.private.federated_operation import NodesFederation
from shfl.private.data import LabeledData
from shfl.private.federated_attack import FederatedPoisoningDataAttack
from shfl.private.utils import unprotected_query


def test_shuffle_node(data_and_labels):
    """Checks that the labels in a node are correctly shuffled."""
    labeled_data = LabeledData(*data_and_labels)
    federated_data = NodesFederation()
    federated_data.append_data_node(labeled_data)
    federated_data.apply_data_transformation(lambda data: random.shuffle(data.label))
    federated_data.configure_data_access(unprotected_query)

    assert (not np.array_equal(federated_data[0].query().label,
                               data_and_labels[1]))


def test_federated_poisoning_attack(data_and_labels):
    """Checks that the labels in a set of federated nodes are correctly shuffled."""
    num_nodes = 10
    federated_data = NodesFederation()

    list_labels = []
    for _ in range(num_nodes):
        data = np.random.rand(*data_and_labels[0].shape)
        label = np.random.randint(0, 10, size=data_and_labels[1].shape)
        list_labels.append(label)
        labeled_data = LabeledData(data, label)
        federated_data.append_data_node(labeled_data)

    percentage = 10
    simple_attack = FederatedPoisoningDataAttack(percentage=percentage)
    simple_attack(nodes_federation=federated_data)

    adversaries_indices = simple_attack.adversaries

    federated_data.configure_data_access(unprotected_query)
    for node, index in zip(federated_data, range(num_nodes)):
        if index in adversaries_indices:
            assert not np.array_equal(node.query().label, list_labels[index])
        else:
            assert np.array_equal(node.query().label, list_labels[index])
