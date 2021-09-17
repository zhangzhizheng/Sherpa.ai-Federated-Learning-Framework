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

# Using method overloading: only one public method needed
# pylint: disable=too-few-public-methods
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic


class FederatedAggregator:
    """Defines different aggregators for model's parameters.

    The aggregators instances must be callable, so the method `__call__`
    is implemented.
    Overloaded functions for aggregating (nested) lists and tuples
    are here provided. Instead, the method to _aggregate array-like
    objects must be implemented.

    # Arguments:
        axis: Optional; Axis or axes along which the aggregation is performed
            (default is 0).
    """

    def __init__(self, axis=0):
        self._axis = axis

    def __call__(self, clients_params):
        """Aggregates clients' parameters.

        # Arguments:
            clients_params: List where each item contains one client's parameters.
                One client's parameters can be a (nested) list or tuples of
                array-like objects.

        # Returns:
            aggregated_params: The aggregated clients' parameters.
        """
        return self._aggregate(*clients_params)

    @dispatch(Variadic[list])
    def _aggregate(self, *params):
        """Aggregates (nested) lists of arrays."""
        aggregated_weights = [self._aggregate(*params)
                              for params in zip(*params)]
        return aggregated_weights

    @dispatch(Variadic[tuple])
    def _aggregate(self, *params):
        """Aggregates (nested) tuples of arrays."""
        aggregated_weights = tuple(self._aggregate(*params)
                                   for params in zip(*params))
        return aggregated_weights
