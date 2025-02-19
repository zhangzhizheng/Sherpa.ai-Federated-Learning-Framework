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

# Disable too many arguments: needed in this case
# pylint: disable=too-many-arguments
import numpy as np

from shfl.data_base.data_base import shuffle_rows
from shfl.data_distribution.data_distribution_sampling import SamplingDataDistribution


class IidDataDistribution(SamplingDataDistribution):
    """Creates a set of federated nodes from a centralized database.

    Implements the class
    [SamplingDataDistribution](../data_distribution/#samplingdatadistribution-class).

    An independent and identically distribution is used, thus each client's
    data will have the same distribution as the centralized data.
    """

    def make_data_federated(self, data, labels, percent=100, num_nodes=1,
                            weights=None, sampling="without_replacement"):
        """Creates the data partition for each client.

        The data and labels may be either Numpy arrays or
        Pandas dataframe/series.

        # Arguments:
            data: Array-like object containing the train data
                to be distributed among a set of federated nodes.
            labels: Array-like object containing the target labels.
            percent: Optional; Percent of the data to be distributed
                (default is 100).
            num_nodes: Optional; Number of nodes to create (default is 1).
            weights: Optional; Array of length `num_nodes` containing the
                distribution weight for each node (default is None,
                in which case all weights are set to be equal).
            sampling: Optional; Sample with or without replacement
                (default is "without_replacement").
            **kwargs: Optional named arguments. These can be passed
                when invoking the class method
                [get_nodes_federation](./#get_nodes_federation).

        # Returns:
            nodes_federation: List containing the data for each client.
            federated_label: List containing the target labels for each client.
        """
        if weights is None:
            weights = np.full(num_nodes, 1/num_nodes)

        data, labels = shuffle_rows(data, labels)

        # Select percent
        data = data[0:int(percent * len(data) / 100)]
        labels = labels[0:int(percent * len(labels) / 100)]

        federated_data = []
        federated_label = []

        if sampling == "without_replacement":
            if sum(weights) > 1:
                weights = [float(i)/sum(weights) for i in weights]

            sum_used = 0
            percentage_used = 0

            for client in range(0, num_nodes):
                federated_data.append(
                    data[sum_used:int((percentage_used + weights[client]) *
                                      len(data))])
                federated_label.append(
                    labels[sum_used:int((percentage_used + weights[client]) *
                                        len(labels))])

                sum_used = int((percentage_used + weights[client]) * len(data))
                percentage_used += weights[client]
        else:
            for client in range(0, num_nodes):
                federated_data.append(
                    data[:int((weights[client]) * len(data))])
                federated_label.append(
                    labels[:int((weights[client]) * len(labels))])

                data, labels = shuffle_rows(data, labels)

        return federated_data, federated_label
