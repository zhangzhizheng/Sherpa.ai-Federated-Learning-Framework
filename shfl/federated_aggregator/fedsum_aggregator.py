# Using method overloading: only one public method needed
# pylint: disable=too-few-public-methods
import numpy as np
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class FedSumAggregator(FederatedAggregator):
    """Performs a sum of the clients' model's parameters.

    It implements the class
    [FederatedAggregator](./#federatedaggregator-class).

    # Arguments:
        axis: Optional; Axis or axes along which the sum is performed
            (default is 0; see options in [Numpy sum
            function](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)).
    """

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *params):
        """Sums arrays"""
        return np.sum(np.array(params), axis=self._axis)

    @dispatch(Variadic[list, tuple])
    def _aggregate(self, *params):
        return super()._aggregate(*params)
