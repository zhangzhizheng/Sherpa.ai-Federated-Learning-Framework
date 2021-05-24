import numpy as np

from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution import DataDistribution
from shfl.private.federated_operation import FederatedData


class DataDistributionTest(DataDistribution):
    """Creates a dummy distribution among federated clients."""
    def make_data_federated(self, data, labels, **kwargs):
        return list(data), list(labels)


class DataBaseTest(DataBase):
    """Creates a database with train, test and validation sets of random values."""
    def load_data(self):
        self._train_data = np.random.rand(50).reshape([10, 5])
        self._test_data = np.random.rand(50).reshape([10, 5])
        self._train_labels = np.random.randint(0, 10, 10)
        self._test_labels = np.random.randint(0, 10, 10)


def test_data_distribution_private_data():
    """Checks that a database is correctly encapsulated in a data distribution."""
    data_base = DataBaseTest()
    data_base.load_data()
    _, _, test_data_ref, test_labels_ref = data_base.data

    data_distribution = DataDistributionTest(data_base)
    federated_data, test_data, test_labels = data_distribution.get_federated_data()

    assert hasattr(data_distribution, "_database")
    assert isinstance(federated_data, FederatedData)
    np.testing.assert_array_equal(test_data, test_data_ref)
    np.testing.assert_array_equal(test_labels, test_labels_ref)
