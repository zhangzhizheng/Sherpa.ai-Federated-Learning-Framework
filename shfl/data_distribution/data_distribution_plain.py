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
            data: List or Dictionary, each element contain data for one client
            labels: List or Dictionary, each element contain labels for
                one client
            percent: unused, all data is employed.

        # Returns:
            federated_data: List, each element contain data for one client
            federated_label: List, each element contain labels for one client
        """

        del percent

        federated_data = list(data.values()) if \
            isinstance(data, dict) else data
        federated_label = list(labels.values()) if \
            isinstance(labels, dict) else labels

        return federated_data, federated_label
