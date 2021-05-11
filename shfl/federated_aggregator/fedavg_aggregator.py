import numpy as np
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class FedAvgAggregator(FederatedAggregator):
    """Performs an average of the clients' model's parameters.

    It implements the class
    [FederatedAggregator](./#federatedaggregator-class).

    # Arguments:
        percentage: Optional; Proportion of the total data
            that each client possesses. The default is None,
            in which case it is assumed that all clients
            possess a comparable amount of data.
        axis: Optional; Axis or axes along which a mean is performed
            (default is 0; see options in [Numpy mean
            function](https://numpy.org/doc/stable/reference/generated/numpy.mean.html)).


    # References:
        [Communication-Efficient Learning of Deep Networks
        from Decentralized Data](https://arxiv.org/abs/1602.05629)
    """

    def __init__(self, percentage=None, axis=0):
        super().__init__(percentage=percentage)
        self._axis = axis

    def aggregate_weights(self, clients_params):
        """See base class.
        """

        return self._aggregate(*clients_params)

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *params):
        """Averages arrays."""
        return np.mean(np.array(params), axis=self._axis)

    @dispatch(Variadic[list])
    def _aggregate(self, *params):
        """Averages (nested) lists of arrays."""
        aggregated_weights = [self._aggregate(*params)
                              for params in zip(*params)]
        return aggregated_weights

    @dispatch(Variadic[tuple])
    def _aggregate(self, *params):
        """Sums (nested) tuples of arrays"""
        aggregated_weights = tuple(self._aggregate(*params)
                                   for params in zip(*params))
        return aggregated_weights
