import numpy as np

from shfl.data_base.data_base import shuffle_rows
from shfl.data_distribution.data_distribution import DataDistribution


class ExplicitDataDistribution(DataDistribution):
    """Distributes the data using the clients' identifier.

    Implements the class
    [DataDistribution](../data_distribution/#datadistribution-class).

    It is assumed that the data is a Pandas dataframe where each item
    contains one data sample in 2-tuples format as
    `(node_identifier, node_data)`. That is, the first
    tuple's item is the identifier and the second one
    correspond with the data. The data will be grouped by identifier and
    appended to each node's private data.
    """

    def make_data_federated(self, data, labels, **kwargs):
        """Creates the data partition for each client.

        The first column of the data input must contain the node's identifier.

        # Arguments:
            data: The data to federate. It is a Dataframe containing tuples where
                the first element must contain the node's identifier and the
                second the data.
            labels: The target labels associated to each entry in data.

        # Returns:
            federated_data: List containing the data for each client.
            federated_label: List containing the target labels for each client.
        """
        data, labels = shuffle_rows(data, labels)

        nodes = np.unique(data[:, 0])
        idx = dict()
        for i, node_identifier in enumerate(nodes):
            idx[node_identifier] = i

        federated_data = [[] for i in range(len(nodes))]
        federated_label = [[] for i in range(len(nodes))]
        for (node_identifier, data_chunk), label in zip(data, labels):
            federated_data[idx[node_identifier]].append(data_chunk)
            federated_label[idx[node_identifier]].append(label)

        return federated_data, federated_label
