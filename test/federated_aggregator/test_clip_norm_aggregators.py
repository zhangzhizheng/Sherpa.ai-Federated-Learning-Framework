"""
This file is part of Sherpa Federated Learning Framework.

Sherpa Federated Learning Framework is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3.

Sherpa Federated Learning Framework is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np

from shfl.federated_aggregator import NormClipAggregator
from shfl.federated_aggregator import WeakDPAggregator


class NormClipAggregatorTest(NormClipAggregator):
    """Creates test class for the clipped norm aggregator."""
    def serialize(self, data):
        """Calls the protected member of the parent class."""
        return self._serialize(data)

    def deserialize(self, data):
        """Calls the protected member of the parent class."""
        return self._deserialize(data)


def test_norm_clip_aggregator_list_of_arrays(params_definition, helpers):
    """Checks that the clipped norm aggregator correctly aggregates
    clients' parameters in a list of arrays with different sizes."""
    _, _, _, \
        num_layers, layers_shapes, clients_params = params_definition

    aggregator = NormClipAggregator(clip=100)
    aggregated_params = aggregator(clients_params)

    true_aggregation = helpers.average_list_of_arrays(clients_params, layers_shapes)

    for i in range(num_layers):
        assert np.array_equal(true_aggregation[i], aggregated_params[i])
    assert len(aggregated_params) == num_layers


def test_weak_dp_aggregator_list_of_arrays(params_definition, helpers):
    """Checks that the weak differential privacy aggregator correctly
    aggregates clients' parameters for lists of arrays."""
    num_clients, _, _, num_layers, layers_shapes, clients_params = params_definition
    seed = 1231231
    clip = 100

    np.random.seed(seed)
    aggregator = WeakDPAggregator(clip=clip)
    aggregated_params = aggregator(clients_params)

    true_aggregation = helpers.sum_list_of_arrays(clients_params, layers_shapes)
    np.random.seed(seed)
    for i_params in range(num_layers):
        noise = np.random.normal(loc=0.0,
                                 scale=0.025*clip/num_clients,
                                 size=true_aggregation[i_params].shape)
        true_aggregation[i_params] = true_aggregation[i_params] / num_clients + noise

    for i_params in range(num_layers):
        assert np.allclose(true_aggregation[i_params], aggregated_params[i_params])
        assert aggregated_params[i_params].shape == true_aggregation[i_params].shape


def test_norm_clip_aggregator_multidimensional_2d(params_definition, helpers):
    """Checks that the clipped norm aggregator correctly aggregates
    clients' parameters for 2D arrays."""
    num_clients, num_rows, num_cols, \
        _, _, _ = params_definition
    clip = 100

    clients_params = []
    for _ in range(num_clients):
        clients_params.append(np.random.rand(num_rows, num_cols))

    aggregator = NormClipAggregator(clip=clip)
    aggregated_params = aggregator(clients_params)

    true_aggregation = helpers.average_arrays(clients_params, (num_rows, num_cols))

    assert np.array_equal(true_aggregation, aggregated_params)
    assert aggregated_params.shape == true_aggregation.shape


def test_weak_dp_aggregator_multidimensional_2d(params_definition, helpers):
    """Checks that the weak differential privacy aggregator correctly
    aggregates clients' parameters for 2D arrays."""
    num_clients, num_rows, num_cols, \
        _, _, _ = params_definition
    clip = 100

    clients_params = []
    for _ in range(num_clients):
        clients_params.append(np.random.rand(num_rows, num_cols))

    np.random.seed(0)
    aggregator = WeakDPAggregator(clip=clip)
    aggregated_params = aggregator(clients_params)

    true_aggregation = helpers.average_arrays(clients_params, (num_rows, num_cols))

    np.random.seed(0)
    noise = np.random.normal(loc=0.0,
                             scale=0.025*clip/num_clients,
                             size=true_aggregation.shape)
    true_aggregation += noise

    assert np.array_equal(true_aggregation, aggregated_params)
    assert len(aggregated_params) == true_aggregation.shape[0]


def test_norm_clip_aggregator_multidimensional_3d(params_definition, helpers):
    """Checks that the clipped norm aggregator correctly aggregates
    clients' parameters for 3D arrays."""
    num_clients, num_rows, num_cols, \
        _, _, _ = params_definition
    num_k = 5
    clip = 1000

    clients_params = []
    for _ in range(num_clients):
        clients_params.append(np.random.rand(num_rows,
                                             num_cols,
                                             num_k))

    aggregator = NormClipAggregator(clip=clip)
    aggregated_params = aggregator(clients_params)

    true_aggregation = helpers.average_arrays(clients_params, (num_rows, num_cols, num_k))

    assert np.array_equal(true_aggregation, aggregated_params)
    assert aggregated_params.shape == true_aggregation.shape


def test_weak_dp_aggregator_multidimensional_3d(params_definition, helpers):
    """Checks that the weak differential privacy aggregator correctly
    aggregates clients' parameters for 3D arrays."""
    num_clients, num_rows, num_cols, \
        _, _, _ = params_definition
    num_k = 5
    clip = 1000

    clients_params = []
    for _ in range(num_clients):
        clients_params.append(np.random.rand(num_rows, num_cols, num_k))

    np.random.seed(0)
    aggregator = WeakDPAggregator(clip=clip)
    aggregated_params = aggregator(clients_params)

    true_aggregation = helpers.average_arrays(clients_params, (num_rows, num_cols, num_k))

    np.random.seed(0)
    noise = np.random.normal(loc=0.0,
                             scale=0.025*clip/num_clients,
                             size=true_aggregation.shape)
    true_aggregation += noise

    assert np.array_equal(true_aggregation, aggregated_params)
    assert len(aggregated_params) == true_aggregation.shape[0]


def test_serialization_deserialization_multidimensional_3d_array(params_definition):
    """Checks the back and forth serialization of 3d arrays."""
    num_clients, num_rows, num_cols, \
        _, _, _ = params_definition
    num_k = 5

    clients_params = [np.random.rand(num_rows, num_cols, num_k)
                      for _ in range(num_clients)]

    aggregator = NormClipAggregatorTest(clip=100)

    serialized_params = np.array([aggregator.serialize(client)
                                  for client in clients_params])
    deserialized_params = np.array([aggregator.deserialize(client)
                                    for client in serialized_params])

    assert np.array_equal(deserialized_params, clients_params)


def test_serialization_deserialization_multidimensional_2d_array(params_definition):
    """Checks the back and forth serialization of 2d arrays."""
    num_clients, num_rows, num_cols, \
        _, _, _ = params_definition

    clients_params = [np.random.rand(num_rows, num_cols)
                      for _ in range(num_clients)]

    aggregator = NormClipAggregatorTest(clip=100)

    serialized_params = np.array([aggregator.serialize(client)
                                  for client in clients_params])
    deserialized_params = np.array([aggregator.deserialize(client)
                                    for client in serialized_params])

    assert np.array_equal(deserialized_params, clients_params)


def test_serialization_deserialization_list_of_arrays(params_definition):
    """Checks the back and forth serialization of a list of arrays
    having different shapes."""
    _, _, _, _, _, clients_params = params_definition

    aggregator = NormClipAggregatorTest(clip=100)

    serialized_params = np.array([aggregator.serialize(client)
                                  for client in clients_params])
    for i, client in enumerate(serialized_params):
        deserialized = aggregator.deserialize(client)
        for j, arr in enumerate(deserialized):
            assert np.array_equal(arr, clients_params[i][j])


def test_serialization_deserialization_float_and_arrays(params_definition):
    """Checks the back and forth serialization of a list of arrays
    together with float numbers."""
    num_clients, _, _, _, _, _ = params_definition

    clients_params = []
    for _ in range(num_clients):
        clients_params.append([np.random.rand(),
                               np.random.rand(20, 30),
                               np.random.rand(50, 40)])

    aggregator = NormClipAggregatorTest(clip=100)

    serialized_params = np.array([aggregator.serialize(client)
                                  for client in clients_params])
    for i, client in enumerate(serialized_params):
        deserialized = aggregator.deserialize(client)
        for j, arr in enumerate(deserialized):
            assert np.array_equal(arr, clients_params[i][j])
