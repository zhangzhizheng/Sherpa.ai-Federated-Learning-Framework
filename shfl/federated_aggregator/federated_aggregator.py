import abc


class FederatedAggregator(abc.ABC):
    """Defines different aggregators for model's parameters.

    The way the parameters are aggregated must be specified in the
    abstract method `aggregate_weights` of this class.

    # Arguments:
        percentage: Optional; Proportion of the total data
            that each client possesses. The default is None,
            in which case it is assumed that all clients
            possess a comparable amount of data.
    """

    def __init__(self, percentage=None):
        self._percentage = percentage

    @abc.abstractmethod
    def aggregate_weights(self, clients_params):
        """Aggregates clients' model's parameters.

        Abstract method.

        # Arguments:
            clients_params: List, each element contains one client's
                model's parameters.

        # Returns:
            aggregated_params: The aggregated model's parameters.
                Typically, this represents the collaborative (global) model.
        """
