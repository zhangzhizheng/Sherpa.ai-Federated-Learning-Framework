import numpy as np
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class WeightedFedAvgAggregator(FederatedAggregator):
    """Performs a weighted average of the clients' model's parameters.

    It implements the class
    [FederatedAggregator](./#federatedaggregator-class).

    The weights are proportional to the amount of data in each client's node.
    In other words, a client possessing more data will have more influence on
    the collaborative (global) model.

    # Example:
        In the case of three federated clients that possess, respectively,
        10%, 20% and 70% of the total data,
        we will pass the argument `percentage=[0.1, 0.2, 0.7]`.
    """

    def aggregate_weights(self, clients_params):
        """
        See base class.
        """
        ponderated_weights = [self._ponderate_weights(i_client, i_weight)
                              for i_client, i_weight
                              in zip(clients_params, self._percentage)]

        return self._aggregate(*ponderated_weights)

    @dispatch((np.ndarray, np.ScalarType), np.ScalarType)
    def _ponderate_weights(self, params, weight):
        """Applies the weights to arrays."""
        return params * weight

    @dispatch(list, np.ScalarType)
    def _ponderate_weights(self, params, weight):
        """Applies the weights to (nested) lists of arrays."""
        ponderated_weights = [self._ponderate_weights(i_params, weight)
                              for i_params in params]
        return ponderated_weights

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *ponderated_weights):
        """Sums arrays."""
        return np.sum(np.array(ponderated_weights), axis=0)

    @dispatch(Variadic[list])
    def _aggregate(self, *ponderated_weights):
        """Sums ponderated (nested) lists of arrays"""
        aggregated_weights = [self._aggregate(*params)
                              for params in zip(*ponderated_weights)]
        return aggregated_weights
