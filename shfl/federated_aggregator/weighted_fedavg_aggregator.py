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

# In this case, only one method is needed
# pylint: disable=too-few-public-methods
import numpy as np
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic

from shfl.federated_aggregator.fedsum_aggregator import FedSumAggregator


class WeightedFedAggregator(FedSumAggregator):
    """Performs a weighted average of the clients' model's parameters.

     It implements the class
    [FedSumAggregator](./#fedsumaggregator-class).

    The weights are proportional to the amount of data in each client's node.
    In other words, a client possessing more data will have more influence on
    the collaborative (global) model.

    # Arguments:
        axis: Optional; Axis or axes along which the sum is performed
            (see [Numpy sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)).

    # Example:
        In the case of three federated clients that possess, respectively,
        10%, 20% and 70% of the total data,
        we will pass the argument `percentage=[0.1, 0.2, 0.7]`.
    """

    def __call__(self, clients_params, percentage):
        """Aggregates clients' parameters.

        # Arguments:
        clients_params: List where each item contains one client's parameters.
            One client's parameters can be a (nested) list or tuples of
            array-like objects.
        percentage: Proportion (normalized to 1) of the total data
            that each client possesses.

        # Returns:
            aggregated_params: The aggregated clients' parameters.
        """
        weighted_params = \
            [self._weight_params(i_client, i_weight)
             for i_client, i_weight
             in zip(clients_params, percentage)]

        return self._aggregate(*weighted_params)

    @dispatch(Variadic[list, tuple, np.ndarray, np.ScalarType])
    def _aggregate(self, *params):
        return super()._aggregate(*params)

    # Since using method overloading, "self" is needed for coherence
    # pylint: disable=no-self-use
    @dispatch((np.ndarray, np.ScalarType), np.ScalarType)
    def _weight_params(self, params, weight):
        """Applies the weights to arrays."""
        return params * weight

    @dispatch(list, np.ScalarType)
    def _weight_params(self, params, weight):
        """Applies the weights to (nested) lists of arrays."""
        weighted_params = [self._weight_params(i_params, weight)
                           for i_params in params]
        return weighted_params

    @dispatch(tuple, np.ScalarType)
    def _weight_params(self, params, weight):
        """Applies the weights to (nested) lists of arrays."""
        weighted_params = tuple(self._weight_params(i_params, weight)
                                for i_params in params)
        return weighted_params
