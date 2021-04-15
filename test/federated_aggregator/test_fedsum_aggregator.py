import numpy as np
import pytest

from shfl.federated_aggregator.fedsum_aggregator import FedSumAggregator


@pytest.mark.parametrize("iterable_type", [list, tuple])
def test_aggregate_list_or_tuple_of_numpy_arrays(iterable_type):
    num_clients = 10
    layers_dims = [[128, 64, 32], [32, 64, 32],
                   [32, 128], [12, 32], [32, 10, 1]]

    clients_params = []
    for i in range(num_clients):
        clients_params.append(iterable_type(np.random.rand(*dims)
                              for dims in layers_dims))

    true_aggregation = [np.zeros(dims) for dims in layers_dims]
    for i_client in range(num_clients):
        for i_layer in range(len(layers_dims)):
            true_aggregation[i_layer] += clients_params[i_client][i_layer]

    aggregator = FedSumAggregator()
    aggregated_weights = aggregator.aggregate_weights(clients_params)

    for true, computed in zip(true_aggregation, aggregated_weights):
        np.testing.assert_array_equal(true, computed)
    assert isinstance(aggregated_weights, iterable_type)


def test_aggregate_single_numpy_arrays():
    num_clients = 10
    layers_dims = [128, 64, 32]

    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(*layers_dims))

    true_aggregation = np.zeros(layers_dims)
    for i_client in range(num_clients):
        true_aggregation += clients_params[i_client]

    aggregator = FedSumAggregator()
    aggregated_weights = aggregator.aggregate_weights(clients_params)
    np.testing.assert_array_equal(true_aggregation, aggregated_weights)
    assert isinstance(aggregated_weights, np.ndarray)


def test_aggregate_single_scalar():
    num_clients = 10

    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(1).item())

    true_aggregation = 0.0
    for i_client in range(num_clients):
        true_aggregation += clients_params[i_client]

    aggregator = FedSumAggregator()
    aggregated_weights = aggregator.aggregate_weights(clients_params)
    np.testing.assert_almost_equal(true_aggregation,
                                   aggregated_weights,
                                   decimal=8)
    assert isinstance(aggregated_weights, np.ScalarType)
