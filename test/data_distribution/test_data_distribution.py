import numpy as np

from shfl.data_distribution.data_distribution import DataDistribution
from shfl.private.federated_operation import NodesFederation


class DataDistributionTest(DataDistribution):
    """Creates a dummy distribution among federated clients."""
    def make_data_federated(self, data, labels, **kwargs):
        return list(data), list(labels)


def test_data_distribution_private_data(data_and_labels_arrays, labeled_data_base):
    """Checks that a database is correctly encapsulated in a data distribution."""
    data, labels = data_and_labels_arrays
    data_base = labeled_data_base(data, labels)
    _, _, test_data_ref, test_labels_ref = data_base.load_data()

    data_distribution = DataDistributionTest(data_base)
    federated_data, test_data, test_labels = data_distribution.get_nodes_federation()

    assert hasattr(data_distribution, "_database")
    assert isinstance(federated_data, NodesFederation)
    np.testing.assert_array_equal(test_data, test_data_ref)
    np.testing.assert_array_equal(test_labels, test_labels_ref)
