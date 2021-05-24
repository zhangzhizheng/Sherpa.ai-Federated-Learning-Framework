import numpy as np
from numpy import linalg as la
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic

from shfl.federated_aggregator.fedavg_aggregator import FedAvgAggregator


class NormClipAggregator(FedAvgAggregator):
    """Performs an average of the clients' model's parameters with clipped norm.

    It implements the class
    [FedAvgAggregator](./#fedavgaggregator-class).

    It clips the norm of the client's updates and averages them.

    # Arguments:
        clip: Value used to clip each client's update.
    """

    def __init__(self, clip):
        super().__init__()
        self._clip = clip
        self._data_shape_list = []

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *params):
        """Aggregates arrays."""
        array_params = np.array(params)
        for i, values in enumerate(array_params):
            norm = la.norm(values)
            array_params[i] = np.multiply(values, min(1, self._clip/norm))

        return np.mean(array_params, axis=0)

    @dispatch(Variadic[list])
    def _aggregate(self, *params):
        """Aggregates (nested) lists of arrays."""
        serialized_params = np.array([self._serialize(client)
                                      for client in params])
        serialized_aggregation = self._aggregate(*serialized_params)
        aggregated_weights = self._deserialize(serialized_aggregation)

        return aggregated_weights

    def _serialize(self, data):
        """Converts a list of multidimensional arrays into a list
            of one-dimensional arrays.

        # Arguments:
            data: List of multidimensional arrays.
        """
        data = [np.array(j) for j in data]
        self._data_shape_list = [j.shape for j in data]
        serialized_data = [j.ravel() for j in data]
        serialized_data = np.hstack(serialized_data)
        return serialized_data

    def _deserialize(self, data):
        """Converts a list of one-dimensional arrays into
            a list of multidimensional arrays.

        The multidimensional shape is stored when it is serialized.

        # Arguments:
            data: List of one-dimensional arrays.
        """

        first_index = 0
        deserialized_data = []
        for shp in self._data_shape_list:
            if len(shp) > 1:
                shift = np.prod(shp)
            elif len(shp) == 0:
                shift = 1
            else:
                shift = shp[0]
            tmp_array = data[first_index:first_index+shift]
            tmp_array = tmp_array.reshape(shp)
            deserialized_data.append(tmp_array)
            first_index += shift
        return deserialized_data


class CDPAggregator(NormClipAggregator):
    """Performs an average of the clients' model's parameters
        with differential privacy.

    It implements the class
    [NormClipAggregator](./#normclipaggregator-class).

    Also known as "Central Differential Privacy" aggregation.
    It clips the norm of the client's updates, averages them and
    adds gaussian noise calibrated to `noise_mult * clip / number_of_clients`.

    # Arguments:
        clip: The value used to clip each client's update.
        noise_mult: The amount of noise to add. To ensure proper
            differential privacy, it must be calibrated according
            to some composition theorem.
    """

    def __init__(self, clip, noise_mult):
        super().__init__(clip=clip)
        self._noise_mult = noise_mult

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *params):
        """Aggregation of arrays.

        The gaussian noise is calibrated to
        `noise_mult*clip/number_of_clients`
        """
        clients_params = np.array(params)
        mean = super()._aggregate(*params)
        noise = np.random.normal(
            loc=0.0,
            scale=self._noise_mult * self._clip / len(clients_params),
            size=mean.shape)
        return mean + noise

    @dispatch(Variadic[list])
    def _aggregate(self, *params):
        """Aggregation of (nested) lists of arrays.

        The gaussian noise is calibrated to
        `noise_mult*clip/number_of_clients`
        """
        return super()._aggregate(*params)


class WeakDPAggregator(CDPAggregator):
    """Performs an average of the clients' model's parameters
        with _weak_ differential privacy.

    It implements the class
    [CDPAggregator](./#cdpaggregator-class).

    It clips the norm of the client's updates, averages them and
    adds gaussian noise calibrated to `0.025*clip/number_of_clients`.
    The noise multiplier 0.025 is not enough to
    ensure proper differential privacy.

    # Arguments:
        clip: The value used to clip each client's update.
    """
    def __init__(self, clip, noise_mult=0.025):
        super().__init__(clip=clip, noise_mult=noise_mult)

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *params):
        """Aggregates arrays."""
        return super()._aggregate(*params)

    @dispatch(Variadic[list])
    def _aggregate(self, *params):
        """Aggregates (nested) lists of arrays."""
        return super()._aggregate(*params)
