import numpy as np

from shfl.private.federated_operation import federate_list
from shfl.private.data import UnprotectedAccess


def test_convert_to_federated_data():
    distributed_data = [np.random.rand(50).reshape([10, -1])
                        for _ in range(5)],
    distributed_labels = [np.random.randint(0, 2, size=(10, ))
                          for _ in range(5)]

    federated_data = federate_list(distributed_data,
                                   distributed_labels)
    federated_data.configure_data_access(UnprotectedAccess())

    for i, node in enumerate(federated_data):
        np.testing.assert_array_equal(node.query().data,
                                      distributed_data[i])
        np.testing.assert_array_equal(node.query().label,
                                      distributed_labels[i])


def test_convert_to_federated_data_no_labels():
    distributed_data = [np.random.rand(50).reshape([10, -1])
                        for _ in range(5)],

    federated_data = federate_list(distributed_data)
    federated_data.configure_data_access(UnprotectedAccess())

    for i, node in enumerate(federated_data):
        np.testing.assert_array_equal(node.query().data,
                                      distributed_data[i])
        assert node.query().label is None
