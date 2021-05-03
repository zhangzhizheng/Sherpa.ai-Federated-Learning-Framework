import abc

from shfl.data_distribution.data_distribution import DataDistribution


class SamplingDataDistribution(DataDistribution):
    """Defines the data sampling for the non-IID scenario.

    Implements the class
    [Data Distribution](../data_distribution/#datadistribution-class).
    """

    @abc.abstractmethod
    def make_data_federated(self, data, labels, percent=100, num_nodes=1,
                            weights=None, sampling="without_sampling"):
        """Creates the data partition for each client.

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
                [get_federated_data](./#get_federated_data).

        # Returns:
            federated_data: List containing the data for each client.
            federated_label: List containing the target labels for each client.
        """
