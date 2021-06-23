# Using method overloading:
# pylint: disable=function-redefined
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic


class FederatedAggregator:
    """Defines different aggregators for model's parameters.

    The aggregators instances must be callable, so the method `__call__`
    is implemented.
    Overloaded functions for aggregating (nested) lists and tuples
    are here provided. Instead, the method to aggregate array-like
    objects must be implemented.

    # Arguments:
        axis: Optional; Axis or axes along which the aggregation is performed
            (default is 0).
    """

    def __init__(self, axis=0):
        self._axis = axis

    def __call__(self, clients_params):
        """Aggregates clients' parameters.

        # Arguments:
            clients_params: List where each item contains one client's parameters.
                One client's parameters can be a (nested) list or tuples of
                array-like objects.

        # Returns:
            aggregated_params: The aggregated clients' parameters.
        """
        return self.aggregate(*clients_params)

    @dispatch(Variadic[list])
    def aggregate(self, *params):
        """Aggregates (nested) lists of arrays."""
        aggregated_weights = [self.aggregate(*params)
                              for params in zip(*params)]
        return aggregated_weights

    @dispatch(Variadic[tuple])
    def aggregate(self, *params):
        """Aggregates (nested) tuples of arrays."""
        aggregated_weights = tuple(self.aggregate(*params)
                                   for params in zip(*params))
        return aggregated_weights
