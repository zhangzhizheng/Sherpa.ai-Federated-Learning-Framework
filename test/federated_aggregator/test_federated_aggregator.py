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

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class FederatedAggregatorTest(FederatedAggregator):
    """Creates a dummy class for the federated aggregator."""
    @property
    def axis(self):
        """Returns the percentage."""
        return self._axis

    def aggregate_weights(self, clients_params):
        """Dummy function."""


def test_federated_aggregator_private_data():
    """Checks that the percentage attribute is correctly assigned."""
    axis = 0
    federated_aggregator = FederatedAggregatorTest(axis)

    assert federated_aggregator.axis == axis
