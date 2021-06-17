import abc
import random
import numpy as np

from shfl.private.utils import shuffle_node_query


class FederatedDataAttack(abc.ABC):
    """Defines an adversarial attack on a set of federated nodes.

    Abstract method.
    """

    @abc.abstractmethod
    def apply_attack(self, federated_data):
        """Applies an arbitrary adversarial attack on a set of federated nodes.

        # Arguments:
            federated_data: Object of class
                [FederatedData](../federated_operation/#federateddata-class),
                the set of federated nodes to attack.

        # Example:
            See implementation of class
            [FederatedPoisoningDataAttack](./#federatedpoisoningdataattack-class).
        """


class FederatedPoisoningDataAttack(FederatedDataAttack):
    """Simulates a poisoning data attack.

    Implements the class [FederatedDataAttack](./#federateddataattack-class).

    This attack consists in shuffling the labels of some nodes, which are
    then considered as *adversarial*.

    # Arguments:
        percentage: Percentage of adversarial nodes.

    # Properties:
        adversaries: Returns adversary nodes' indices.
    """

    def __init__(self, percentage):
        super().__init__()
        self._percentage = percentage
        self._adversaries = []

    @property
    def adversaries(self):
        """Returns adversary nodes' indices."""
        return self._adversaries

    def apply_attack(self, federated_data):
        """Shuffles the target labels in an adversary node.

        If tagged as adversarial, a node's target label is shuffled
        by a data transformation (see [ShuffleNode](./#shufflenode-class)).

        # Arguments:
            federated_data: Object of class
                [FederatedData](../federated_operation/#federateddata-class),
                the set of federated nodes to attack.
        """
        num_nodes = federated_data.num_nodes()
        list_nodes = np.arange(num_nodes)
        self._adversaries = random.sample(list(list_nodes),
                                          k=int(self._percentage / 100 * num_nodes))
        boolean_adversaries = [1 if x in self._adversaries else 0 for x in list_nodes]

        for node, boolean in zip(federated_data, boolean_adversaries):
            if boolean:
                node.apply_data_transformation(shuffle_node_query)
