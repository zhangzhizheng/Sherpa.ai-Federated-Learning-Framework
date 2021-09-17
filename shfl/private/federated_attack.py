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

import random
import numpy as np


class FederatedPoisoningDataAttack:
    """Simulates a poisoning data attack.

    This attack consists in shuffling the labels of some nodes, which are
    then considered as *adversarial*.

    # Arguments:
        percentage: Percentage of adversarial nodes.

    # Properties:
        adversaries: Returns adversary nodes' indices.
    """

    def __init__(self, percentage):
        self._percentage = percentage
        self._adversaries = []

    @property
    def adversaries(self):
        """Returns adversary nodes' indices."""
        return self._adversaries

    def __call__(self, nodes_federation):
        """Shuffles the target labels in an adversary node.

        If tagged as adversarial, a node's target label is shuffled
        by a data transformation.

        # Arguments:
            nodes_federation: Object of class
                [NodesFederation](../federated_operation/#nodesfederation-class),
                the set of federated nodes to attack.
        """
        num_nodes = nodes_federation.num_nodes()
        list_nodes = np.arange(num_nodes)
        self._adversaries = random.sample(list(list_nodes),
                                          k=int(self._percentage / 100 * num_nodes))
        boolean_adversaries = [1 if x in self._adversaries else 0 for x in list_nodes]

        for node, boolean in zip(nodes_federation, boolean_adversaries):
            if boolean:
                node.apply_data_transformation(
                    lambda data: np.random.shuffle(data.label))
