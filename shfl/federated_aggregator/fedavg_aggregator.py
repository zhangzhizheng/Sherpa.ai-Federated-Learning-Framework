# Using method overloading:
# pylint: disable=function-redefined, too-few-public-methods
import numpy as np
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class FedAvgAggregator(FederatedAggregator):
    """Performs an average of the clients' model's parameters.

    It implements the class
    [FederatedAggregator](./#federatedaggregator-class).

    # Arguments:
        axis: Optional; Axis or axes along which the mean is performed
            (default is 0; see options in [Numpy mean
            function](https://numpy.org/doc/stable/reference/generated/numpy.mean.html)).

    # References:
        [Communication-Efficient Learning of Deep Networks
        from Decentralized Data](https://arxiv.org/abs/1602.05629)
    """

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def aggregate(self, *params):
        """Averages arrays."""
        return np.mean(np.array(params), axis=self._axis)

    @dispatch(Variadic[list, tuple])
    def aggregate(self, *params):
        return super().aggregate(*params)
