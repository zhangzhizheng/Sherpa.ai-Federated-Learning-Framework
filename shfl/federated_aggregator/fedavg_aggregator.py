import numpy as np
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class FedAvgAggregator(FederatedAggregator):
    """Performs an average of the clients' model's parameters.

    It implements the class
    [FederatedAggregator](./#federatedaggregator-class).
    """

    def aggregate_weights(self, clients_params):
        """
        See base class.

        # References:
            [Communication-Efficient Learning of Deep Networks
            from Decentralized Data](https://arxiv.org/abs/1602.05629)
        """

        return self._aggregate(*clients_params)

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *params):
        """Averages arrays."""
        return np.mean(np.array(params), axis=0)

    @dispatch(Variadic[list])
    def _aggregate(self, *params):
        """Averages (nested) lists of arrays."""
        aggregated_weights = [self._aggregate(*params)
                              for params in zip(*params)]
        return aggregated_weights
