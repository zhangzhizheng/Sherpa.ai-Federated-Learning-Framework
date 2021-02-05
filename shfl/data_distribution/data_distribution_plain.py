from shfl.data_distribution.data_distribution import DataDistribution


class PlainDataDistribution(DataDistribution):
    """
    Distributes data over federated nodes, without any partition nor shuffle.
    It is assumed that the data is provided in a list or dictionary where each
    item contains a single node's data.
    """

    def make_data_federated(self, data, labels, percent=100, *args, **kwargs):
        """
        Simply return data and labels without changing the data split and order.
        If a dictionary is provided, this is converted to a list.

        # Arguments:
            data: Data already divided for each client
            labels: Labels already divided for each client
            percent: 100, use all the data

        # Returns:
            data: unchanged
            labels: unchanged
        """

        federated_data = list(data.values()) if \
            isinstance(data, dict) else data
        federated_label = list(labels.values()) if \
            isinstance(labels, dict) else labels

        return federated_data, federated_label
