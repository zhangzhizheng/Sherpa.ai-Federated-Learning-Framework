import numpy as np
import pandas as pd

from shfl.data_base.data_base import shuffle_rows
from shfl.data_distribution.data_distribution import DataDistribution
from shfl.private import LabeledData, FederatedData


class ExplicitDataDistribution(DataDistribution):
    """
    Implementation of an explicit data distribution using \
        [Data Distribution](../data_distribution/#datadistribution-class)

    In this data distribution we assume that the first column in the data determines the node it belongs to.
    The data and labels may be numpy arrays or pandas dataframe/series.
    """

    def get_federated_data(self, percent=100, *args, **kwargs):
        """
        Method that splits the whole data between the established number of nodes.
        # Arguments:
            num_nodes: Number of nodes to create
            percent: Percent of the data (between 0 and 100) to be distributed (default is 100)
        # Returns:
              * **federated_data, test_data, test_label**
        """

        train_data, train_label = self._database.train
        test_data, test_label = self._database.test

        federated_train_data, federated_train_label, nodes = self.make_data_federated(train_data,
                                                                                      train_label,
                                                                                      percent,
                                                                                      *args, **kwargs)

        federated_data = FederatedData()
        num_nodes = len(nodes)
        for node in range(num_nodes):
            node_data = LabeledData(federated_train_data[node], federated_train_label[node])
            federated_data.add_data_node(node_data, nodes[node])

        return federated_data, test_data, test_label

    def make_data_federated(self, data, labels, percent, *args, **kwargs):
        """
        Method that makes data and labels argument federated using the first column as the node.

        # Arguments:
            data: Data to federate. The first column contains the node identifier
            labels: Labels to federate
            percent: Percent of the data (between 0 and 100) to be distributed (default is 100)

        # Returns:
              * **federated_data, federated_labels**
        """
        # Shuffle data
        data, labels = shuffle_rows(data, labels)

        # Select percent
        data = data[0:int(percent * len(data) / 100)]
        labels = labels[0:int(percent * len(labels) / 100)]

        nodes = np.unique(np.array(data)[:, 0])

        if isinstance(data, (pd.DataFrame, pd.Series)) and isinstance(labels, (pd.DataFrame, pd.Series)):
            federated_data = [data[data.iloc[:, 0] == user].drop(data.columns[0], axis=1) for user in nodes]
            federated_label = [labels[data.iloc[:, 0] == user] for user in nodes]

        elif isinstance(data, np.ndarray) and isinstance(labels, np.ndarray):
            federated_data = [np.delete(data[data[:, 0] == user], 0, 1) for user in nodes]
            federated_label = [labels[data[:, 0] == user] for user in nodes]

            federated_data = np.array(federated_data)
            federated_label = np.array(federated_label)

        return federated_data, federated_label, nodes
