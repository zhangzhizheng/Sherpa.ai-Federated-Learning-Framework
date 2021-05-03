import numpy as np

from shfl.data_base.data_base import shuffle_rows
from shfl.data_distribution.data_distribution import DataDistribution


class ExplicitDataDistribution(DataDistribution):
    """Distributes the data using the clients' identifier.

    Implements class [Data Distribution](../data_distribution/#datadistribution-class).

    It is assumed that the data is organised in 2-tuples where
    the first dimension is the identifier and the second one
    correspond with the data.
    """

    def make_data_federated(self, data, labels, **kwargs):
        """
        Method that makes data and labels argument federated using the first column as the node.

        # Arguments:
            data: Data to federate. The first dimension of the tuple as identifier
            labels: Labels to federate

        # Returns:
              * **federated_data, federated_labels**
        """
        # Shuffle data
        data, labels = shuffle_rows(data, labels)

        nodes = np.unique(data[:, 0])
        idx = dict()
        for i, id in enumerate(nodes):
            idx[id] = i

        federated_data = [[] for i in range(len(nodes))]
        federated_label = [[] for i in range(len(nodes))]
        for (k, d), l in zip(data, labels):
            federated_data[idx[k]].append(d)
            federated_label[idx[k]].append(l)

        return federated_data, federated_label
