import numpy as np
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic

from shfl.federated_aggregator.fedavg_aggregator import FedAvgAggregator


class FedSumAggregator(FedAvgAggregator):
    """Performs a sum of the clients' model's parameters.

    It implements the class
    [FedAvgAggregator](./#fedavgaggregator-class).

    # Arguments:
        percentage: Optional; Proportion of the total data
            that each client possesses. The default is None,
            in which case it is assumed that all clients
            possess a comparable amount of data.
        axis: Optional; Axis or axes along which a sum is performed
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
