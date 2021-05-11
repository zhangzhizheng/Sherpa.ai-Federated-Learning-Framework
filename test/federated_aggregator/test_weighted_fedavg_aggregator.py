import numpy as np

from shfl.federated_aggregator.weighted_fedavg_aggregator import WeightedFedAvgAggregator


def test_aggregated_weights():
    num_clients = 10
    num_layers = 5
    tams = [[128, 64], [64, 64], [64, 64], [64, 32], [32, 10]]

    clients_params = []
    for i in range(num_clients):
        clients_params.append([np.random.rand(tams[j][0], tams[j][1])
                               for j in range(num_layers)])

    percentage = np.random.dirichlet(np.ones(num_clients), size=1)[0]

    aggregator = WeightedFedAvgAggregator(percentage=percentage)
    aggregated_weights = aggregator.aggregate_weights(clients_params)

    own_ponderated_weights = [[np.zeros(shape=shape) for shape in tams]
                              for _ in range(num_clients)]
    for client in range(num_clients):
        for layer in range(num_layers):
            own_ponderated_weights[client][layer] = \
                clients_params[client][layer] * percentage[client]

    own_agg = [np.zeros(shape=shape) for shape in tams]
    for client in range(num_clients):
        for layer in range(num_layers):
            own_agg[layer] += own_ponderated_weights[client][layer]

    for i in range(num_layers):
        assert np.array_equal(own_agg[i], aggregated_weights[i])
    assert len(aggregated_weights) == num_layers


def test_weighted_aggregated_weights_list_of_arrays():
    num_clients = 10

    clients_params = []
    for i_client in range(num_clients):
        clients_params.append([np.random.rand(30, 20),
                               np.random.rand(20, 30),
                               np.random.rand(50, 40)])

    percentage = np.random.dirichlet(np.ones(num_clients), size=1)[0]
    aggregator = WeightedFedAvgAggregator(percentage=percentage)
    aggregated_weights = aggregator.aggregate_weights(clients_params)

    own_ponderated_weights = []
    for i_client in range(num_clients):
        own_ponderated_weights.append(
            [clients_params[i_client][i_params] * percentage[i_client]
             for i_params in range(len(clients_params[0]))])

    own_agg = [np.zeros((30, 20)),
               np.zeros((20, 30)),
               np.zeros((50, 40))]
    for i_client in range(num_clients):
        for i_params in range(len(clients_params[0])):
            own_agg[i_params] += own_ponderated_weights[i_client][i_params]

    for i_params in range(len(clients_params[0])):
        assert np.array_equal(own_agg[i_params], aggregated_weights[i_params])


def test_weighted_aggregated_weights_tuple_of_arrays():
    num_clients = 10

    clients_params = []
    for i_client in range(num_clients):
        clients_params.append((np.random.rand(30, 20),
                               np.random.rand(20, 30),
                               np.random.rand(50, 40)))

    percentage = np.random.dirichlet(np.ones(num_clients), size=1)[0]
    aggregator = WeightedFedAvgAggregator(percentage=percentage)
    aggregated_weights = aggregator.aggregate_weights(clients_params)

    own_ponderated_weights = []
    for i_client in range(num_clients):
        own_ponderated_weights.append(
            tuple(clients_params[i_client][i_params] * percentage[i_client]
                  for i_params in range(len(clients_params[0]))))

    own_agg = [np.zeros((30, 20)),
               np.zeros((20, 30)),
               np.zeros((50, 40))]
    for i_client in range(num_clients):
        for i_params in range(len(clients_params[0])):
            own_agg[i_params] += own_ponderated_weights[i_client][i_params]
    own_agg = tuple(own_agg)

    for i_params in range(len(clients_params[0])):
        assert np.array_equal(own_agg[i_params], aggregated_weights[i_params])
