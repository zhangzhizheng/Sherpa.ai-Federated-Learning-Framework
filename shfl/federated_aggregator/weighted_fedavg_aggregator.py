import numpy as np

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator
from shfl.private.query import CheckDataType


class WeightedFedAvgAggregator(FederatedAggregator):
    """
    Implementation of Weighted Federated Averaging Aggregator.
    The aggregation of the parameters is based in the number of data \
    in every node.

    It implements [Federated Aggregator](../federated_aggregator/#federatedaggregator-class)
    """

    def aggregate_weights(self, clients_params):
        """
        Implementation of abstract method of class
        [AggregateWeightsFunction](../federated_aggregator/#federatedaggregator-class)

        # Arguments:
            clients_params: list of multi-dimensional (numeric) arrays.
            Each entry in the list contains the model's parameters of one client.

        # Returns:
            aggregated_weights: aggregator weights representing the global learning model
        """
        _, _, params_is_list = CheckDataType.get(clients_params[0])
        if params_is_list:
            ponderated_weights = [[percentage * params for params in client]
                                  for percentage, client in zip(self._percentage, clients_params)]
            aggregated_weights = [self._aggregate(params) for params in zip(*ponderated_weights)]
        else:
            ponderated_weights = [percentage * client
                                  for percentage, client in zip(self._percentage, clients_params)]
            aggregated_weights = self._aggregate(ponderated_weights)

        return aggregated_weights

    @staticmethod
    def _aggregate(params):
        """Aggregation of parameter arrays"""
        return np.sum(np.array(params), axis=0)
