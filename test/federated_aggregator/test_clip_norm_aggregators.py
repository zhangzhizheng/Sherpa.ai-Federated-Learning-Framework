import numpy as np

from shfl.federated_aggregator import NormClipAggregator
from shfl.federated_aggregator import WeakDPAggregator


def test_aggregated_weights_norm_clip():
    num_clients = 10
    num_layers = 5
    tams = [[128, 64], [64, 64], [64, 64], [64, 32], [32, 10]]

    clients_params = []
    for i in range(num_clients):
        clients_params.append([np.random.rand(tams[j][0], tams[j][1])
                               for j in range(num_layers)])

    aggregator = NormClipAggregator(clip=100)
    aggregated_weights = aggregator.aggregate_weights(clients_params)

    own_agg = [np.zeros(shape=shape) for shape in tams]
    for client in range(num_clients):
        for layer in range(num_layers):
            own_agg[layer] += clients_params[client][layer]
    own_agg = [params / num_clients for params in own_agg]

    for i in range(num_layers):
        assert np.array_equal(own_agg[i], aggregated_weights[i])
    assert len(aggregated_weights) == num_layers


def test_aggregated_weights_weak_dp():
    num_clients = 10
    num_layers = 5
    tams = [[128, 64], [64, 64], [64, 64], [64, 32], [32, 10]]
    clip = 100

    np.random.seed(0)
    clients_params = []
    for i in range(num_clients):
        clients_params.append([np.random.rand(tams[j][0], tams[j][1])
                               for j in range(num_layers)])

    np.random.seed(0)
    aggregator = WeakDPAggregator(clip=clip)
    aggregated_weights = aggregator.aggregate_weights(clients_params)

    np.random.seed(0)
    serialized_params = [aggregator._serialize(v) for v in clients_params]
    for i, v in enumerate(serialized_params):
        serialized_params[i] = v
    clients_params = [aggregator._deserialize(v) for v in serialized_params]

    own_agg = [np.zeros(shape=shape) for shape in tams]
    for client in range(num_clients):
        for layer in range(num_layers):
            own_agg[layer] += clients_params[client][layer]
    own_agg = [params / num_clients for params in own_agg]

    for i, v in enumerate(own_agg):
        noise = np.random.normal(loc=0.0,
                                 scale=0.025*clip/num_clients,
                                 size=own_agg[i].shape)
        own_agg[i] = v + noise

    for i in range(num_layers):
        assert np.array_equal(own_agg[i], aggregated_weights[i])
    assert len(aggregated_weights) == num_layers


def test_aggregated_weights_multidimensional_2d_array_norm_clip():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9

    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(num_rows_params, num_cols_params))
    clients_params = np.array(clients_params)

    aggregator = NormClipAggregator(clip=10)
    aggregated_weights = aggregator.aggregate_weights(clients_params)

    own_agg = np.zeros((num_rows_params, num_cols_params))
    for i_client in range(num_clients):
        own_agg += clients_params[i_client]
    own_agg = own_agg / num_clients

    assert np.array_equal(own_agg, aggregated_weights)
    assert aggregated_weights.shape == own_agg.shape


def test_aggregated_weights_multidimensional_2d_array_weak_dp():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9
    clip = 100

    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(num_rows_params, num_cols_params))
    clients_params = np.array(clients_params)

    np.random.seed(0)
    aggregator = WeakDPAggregator(clip=clip)
    aggregated_weights = aggregator.aggregate_weights(clients_params)

    np.random.seed(0)
    own_agg = np.zeros((num_rows_params, num_cols_params))
    for v in clients_params:
        own_agg += v
    own_agg = own_agg / num_clients
    noise = np.random.normal(loc=0.0,
                             scale=0.025*clip/num_clients,
                             size=own_agg.shape)
    own_agg += noise

    assert np.array_equal(own_agg, aggregated_weights)
    assert len(aggregated_weights) == own_agg.shape[0]


def test_aggregated_weights_multidimensional_3d_array_norm_clip():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9
    num_k_params = 5

    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(num_rows_params,
                                             num_cols_params,
                                             num_k_params))
    clients_params = np.array(clients_params)

    aggregator = NormClipAggregator(clip=10)
    aggregated_weights = aggregator.aggregate_weights(clients_params)

    own_agg = np.zeros((num_rows_params, num_cols_params, num_k_params))
    for i_client in range(num_clients):
        own_agg += clients_params[i_client]
    own_agg = own_agg / num_clients

    assert np.array_equal(own_agg, aggregated_weights)
    assert aggregated_weights.shape == own_agg.shape


def test_aggregated_weights_multidimensional_3d_array_weak_dp():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9
    num_k_params = 5
    clip = 10

    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(num_rows_params,
                                             num_cols_params,
                                             num_k_params))
    clients_params = np.array(clients_params)

    np.random.seed(0)
    aggregator = WeakDPAggregator(clip=clip)
    aggregated_weights = aggregator.aggregate_weights(clients_params)

    np.random.seed(0)
    own_agg = np.zeros((num_rows_params, num_cols_params, num_k_params))
    for v in clients_params:
        own_agg += v
    own_agg = own_agg / num_clients
    noise = np.random.normal(loc=0.0,
                             scale=0.025*clip/num_clients,
                             size=own_agg.shape)
    own_agg += noise

    assert np.array_equal(own_agg, aggregated_weights)
    assert len(aggregated_weights) == own_agg.shape[0]


def test_aggregated_weights_list_of_arrays_norm_clip():
    num_clients = 10

    clients_params = []
    for i_client in range(num_clients):
        clients_params.append([np.random.rand(30, 20),
                               np.random.rand(20, 30),
                               np.random.rand(50, 40)])

    aggregator = NormClipAggregator(clip=100)
    aggregated_weights = aggregator.aggregate_weights(clients_params)

    own_agg = [np.zeros((30, 20)),
               np.zeros((20, 30)),
               np.zeros((50, 40))]
    for i_client in range(num_clients):
        for i_params in range(len(clients_params[0])):
            own_agg[i_params] += clients_params[i_client][i_params]
    for i_params in range(len(clients_params[0])):
        own_agg[i_params] = own_agg[i_params] / num_clients

    for i_params in range(len(clients_params[0])):
        assert np.array_equal(own_agg[i_params], aggregated_weights[i_params])
        assert aggregated_weights[i_params].shape == own_agg[i_params].shape


def test_aggregated_weights_list_of_arrays_weak_dp():
    num_clients = 10
    seed = 1231231
    clip = 100

    clients_params = []
    for i_client in range(num_clients):
        clients_params.append([np.random.rand(30, 20),
                               np.random.rand(20, 30),
                               np.random.rand(50, 40)])
    np.random.seed(seed)
    aggregator = WeakDPAggregator(clip=clip)
    aggregated_weights = aggregator.aggregate_weights(clients_params)

    np.random.seed(seed)
    own_agg = [np.zeros((30, 20)),
               np.zeros((20, 30)),
               np.zeros((50, 40))]
    for i_client in range(num_clients):
        for i_params in range(len(clients_params[0])):
            own_agg[i_params] += clients_params[i_client][i_params] 
    for i_params in range(len(clients_params[0])):
        noise = np.random.normal(loc=0.0,
                                 scale=0.025*clip/num_clients,
                                 size=own_agg[i_params].shape)
        own_agg[i_params] = own_agg[i_params] / num_clients + noise

    for i_params in range(len(clients_params[0])):
        assert np.allclose(own_agg[i_params], aggregated_weights[i_params])
        assert aggregated_weights[i_params].shape == own_agg[i_params].shape


def test_serialization_deserialization_multidimensional_3d_array():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9
    num_k_params = 5

    clients_params = []
    for i in range(num_clients):
        clients_params.append(
            np.random.rand(num_rows_params, num_cols_params, num_k_params))

    aggregator = NormClipAggregator(clip=100)

    serialized_params = np.array([aggregator._serialize(client)
                                  for client in clients_params])
    deserialized_params = np.array([aggregator._deserialize(client)
                                    for client in serialized_params])
    
    assert np.array_equal(deserialized_params, clients_params)


def test_serialization_deserialization_multidimensional_2d_array():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9

    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(num_rows_params, num_cols_params))

    aggregator = NormClipAggregator(clip=100)

    serialized_params = np.array([aggregator._serialize(client)
                                  for client in clients_params])
    deserialized_params = np.array([aggregator._deserialize(client)
                                    for client in serialized_params])
    
    assert np.array_equal(deserialized_params, clients_params)


def test_serialization_deserialization():
    num_clients = 10
    num_layers = 5
    tams = [[128, 64], [64, 64], [64, 64], [64, 32], [32, 10]]
    
    weights = []
    for i in range(num_clients):
        weights.append([np.random.rand(tams[j][0], tams[j][1])
                        for j in range(num_layers)])

    clients_params = weights

    aggregator = NormClipAggregator(clip=100)

    serialized_params = np.array([aggregator._serialize(client)
                                  for client in clients_params])
    for i, client in enumerate(serialized_params):
        deserialized = aggregator._deserialize(client)
        for j, arr in enumerate(deserialized):
            assert np.array_equal(arr, clients_params[i][j])


def test_serialization_deserialization_list_of_arrays():
    num_clients = 10

    clients_params = []
    for i_client in range(num_clients):
        clients_params.append([np.random.rand(30, 20),
                               np.random.rand(20, 30),
                               np.random.rand(50, 40)])

    aggregator = NormClipAggregator(clip=100)

    serialized_params = np.array([aggregator._serialize(client)
                                  for client in clients_params])
    for i, client in enumerate(serialized_params):
        deserialized = aggregator._deserialize(client)
        for j, arr in enumerate(deserialized):
            assert np.array_equal(arr, clients_params[i][j])


def test_serialization_deserialization_mixed_list():
    num_clients = 10

    clients_params = []
    for i_client in range(num_clients):
        clients_params.append([np.random.rand(),
                               np.random.rand(20, 30),
                               np.random.rand(50, 40)])

    aggregator = NormClipAggregator(clip=100)

    serialized_params = np.array([aggregator._serialize(client)
                                  for client in clients_params])
    for i, client in enumerate(serialized_params):
        deserialized = aggregator._deserialize(client)
        for j, arr in enumerate(deserialized):
            assert np.array_equal(arr, clients_params[i][j])
