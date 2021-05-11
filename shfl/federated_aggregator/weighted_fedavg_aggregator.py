import numpy as np
from multipledispatch import dispatch

from shfl.federated_aggregator.fedsum_aggregator import FedSumAggregator


class WeightedFedAvgAggregator(FedSumAggregator):
    """Performs a weighted average of the clients' model's parameters.

    It implements the class
    [FedSumAggregator](./#fedsumaggregator-class).

    The weights are proportional to the amount of data in each client's node.
    In other words, a client possessing more data will have more influence on
    the collaborative (global) model.

    # Arguments:
        percentage: Optional; Proportion (normalized to 1) of the total data
            that each client possesses. The default is None,
            in which case it is assumed that all clients
            possess a comparable amount of data.
        axis: Optional; Axis or axes along which a sum is performed
            (see [Numpy sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)).

    # Example:
        In the case of three federated clients that possess, respectively,
        10%, 20% and 70% of the total data,
        we will pass the argument `percentage=[0.1, 0.2, 0.7]`.
    """

    def aggregate_weights(self, clients_params):
        """
        See base class.
        """
        ponderated_weights = \
            [self._ponderate_weights(client_index, params)
             for client_index, params
             in enumerate(clients_params)]

        return super()._aggregate(*ponderated_weights)

    @dispatch(int, (np.ndarray, np.ScalarType))
    def _ponderate_weights(self, client_index, params):
        """Applies the weights to arrays."""
        client_weight = self._percentage[client_index]
        return params * client_weight

    @dispatch(int, list)
    def _ponderate_weights(self, client_index, params):
        """Applies the weights to (nested) lists of arrays."""
        ponderated_weights = [self._ponderate_weights(client_index, i_params)
                              for i_params in params]
        return ponderated_weights

    @dispatch(int, tuple)
    def _ponderate_weights(self, client_index, params):
        """Applies the weights to (nested) lists of arrays."""
        ponderated_weights = tuple(self._ponderate_weights(client_index, i_params)
                                   for i_params in params)
        return ponderated_weights
