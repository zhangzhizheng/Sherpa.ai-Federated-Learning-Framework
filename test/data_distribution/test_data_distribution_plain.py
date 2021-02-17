import numpy as np
import pytest

from shfl.data_base.data_base import LabeledDatabase
from shfl.data_distribution.data_distribution_plain import PlainDataDistribution

@pytest.mark.parametrize(
    "data, labels",
    [([np.random.rand(50).reshape([10, -1]) for _ in range(5)],
      [np.random.rand(10) for _ in range(5)]),
     ({node: np.random.rand(50).reshape([10, -1]) for node in range(5)},
      {node: np.random.rand(10) for node in range(5)})])
def test_data_distribution_plain(data, labels):
    data_base = LabeledDatabase(data=data, labels=labels,
                                train_percentage=1, shuffle=False)
    data_base.load_data()
    train_data, train_labels = \
        PlainDataDistribution(data_base).make_data_federated(data, labels)

    for i_node in range(len(data)):
        assert np.array_equal(train_data[i_node], data[i_node])
        assert np.array_equal(train_labels[i_node], labels[i_node])
        assert isinstance(train_data, list)
        assert isinstance(train_labels, list)
        assert len(train_data) == len(data)
        assert len(train_labels) == len(labels)
