import numpy as np
import pandas as pd

from shfl.data_base.data_base import shuffle_rows
from shfl.data_distribution.data_distribution import DataDistribution


class ExplicitDataDistribution(DataDistribution):
    """
    Implementation of an explicit data distribution using \
        [Data Distribution](../data_distribution/#datadistribution-class)

    In this data distribution we assume that the data is organised in 2-tuples where the first dimension is the \
        identifier and the second one correspond with data.
    """

    def make_data_federated(self, data, labels, percent=None, *args, **kwargs):
        """
        Method that makes data and labels argument federated using the first column as the node.

        # Arguments:
            data: Data to federate. The first dimension of the tuple as identifier
            labels: Labels to federate
            percent: None

        # Returns:
              * **federated_data, federated_labels**
        """
        # Shuffle data
        data, labels = shuffle_rows(data, labels)

        nodes = np.unique(data[:, 0])

        federated_data = np.array([data[data[:, 0] == user, 1] for user in nodes])
        federated_label = np.array([labels[data[:, 0] == user] for user in nodes])

        return federated_data, federated_label
