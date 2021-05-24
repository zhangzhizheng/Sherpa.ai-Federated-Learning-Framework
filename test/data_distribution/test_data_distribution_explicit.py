import numpy as np

from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution_explicit import ExplicitDataDistribution


class DataBaseTest(DataBase):
    """Creates a database with train, test and validation sets of random values."""
    def load_data(self):
        self._train_data = np.array([(0, [2, 3, 51]),
                                     (1, [1, 34, 6]),
                                     (2, [22, 33, 7]),
                                     (3, [22, 13, 65]),
                                     (4, [1, 3, 15])], dtype=object)
        self._test_data = np.array([[2, 2, 1],
                                    [0, 22, 4],
                                    [3, 1, 5]])
        self._train_labels = np.array([3, 2, 5, 6, 7])
        self._test_labels = np.array([4, 7, 2])


def test_make_data_federated():
    """Checks that the explicitly divided data is correctly assigned to clients."""
    data_base = DataBaseTest()
    data_base.load_data()
    train_data, train_label, _, _ = data_base.data
    data_distribution = ExplicitDataDistribution(data_base)

    federated_data, federated_label = \
        data_distribution.make_data_federated(train_data, train_label)

    all_data = np.concatenate(federated_data)
    all_label = np.concatenate(federated_label)

    idx = []
    for data_base in all_data:
        ids = np.where((data_base == np.stack(train_data[:, 1])).all(axis=1))[0][0]
        idx.append(train_data[ids, 0])

    assert all_data.shape[0] == train_data.shape[0]
    assert len(federated_data) == len(np.unique(train_data[:, 0]))
    assert (np.sort(all_data.ravel()) == np.sort(np.stack(train_data[idx, 1]).ravel())).all()
    assert (np.sort(all_label, 0) == np.sort(train_label[idx], 0)).all()
    assert np.array_equal(np.sort(idx), np.sort(train_data[:, 0]))
