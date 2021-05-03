import numpy as np

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic


class FedSumAggregator(FederatedAggregator):
    """Performs a sum of the clients' model's parameters

    It implements the class
    [FederatedAggregator](./#federatedaggregator-class).
    """

    def aggregate_weights(self, clients_params):
        """
        See base class.
        """

        return self._aggregate(*clients_params)

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *params):
        """Sums arrays"""
        return np.sum(np.array(params), axis=0)

    @dispatch(Variadic[list])
    def _aggregate(self, *params):
        """Sums (nested) lists of arrays"""
        aggregated_weights = [self._aggregate(*params)
                              for params in zip(*params)]
        return aggregated_weights

    @dispatch(Variadic[tuple])
    def _aggregate(self, *params):
        """Sums (nested) tuples of arrays"""
        aggregated_weights = tuple(self._aggregate(*params)
                                   for params in zip(*params))
        return aggregated_weights
