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

import numpy as np

from shfl.data_distribution.data_distribution import DataDistribution
from shfl.private.federated_operation import NodesFederation
from shfl.data_base.data_base import WrapLabeledDatabase


class DataDistributionTest(DataDistribution):
    """Creates a dummy distribution among federated clients."""
    def make_data_federated(self, data, labels, **kwargs):
        return list(data), list(labels)


def test_data_distribution_private_data(data_and_labels_arrays):
    """Checks that a database is correctly encapsulated in a data distribution."""
    data, labels = data_and_labels_arrays
    data_base = WrapLabeledDatabase(data, labels)
    _, _, test_data_ref, test_labels_ref = data_base.load_data()

    data_distribution = DataDistributionTest(data_base)
    federated_data, test_data, test_labels = data_distribution.get_nodes_federation()

    assert hasattr(data_distribution, "_database")
    assert isinstance(federated_data, NodesFederation)
    np.testing.assert_array_equal(test_data, test_data_ref)
    np.testing.assert_array_equal(test_labels, test_labels_ref)
