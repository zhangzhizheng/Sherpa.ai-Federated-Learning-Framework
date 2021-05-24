import numpy as np

from shfl.federated_aggregator.weighted_fedavg_aggregator import WeightedFedAvgAggregator


def test_aggregate_weighted_list_of_arrays(params_definition, helpers):
    """Checks that lists of arrays of different shapes are correctly aggregated.

    A random weight is assigned to each client. """
    num_clients, _, _, \
        num_layers, layers_shapes, clients_params = params_definition

    percentage = np.random.dirichlet(np.ones(num_clients), size=1)[0]

    aggregator = WeightedFedAvgAggregator(percentage=percentage)
    aggregated_weights = aggregator.aggregate_weights(clients_params)

    own_ponderated_weights = [[np.zeros(shape=shape) for shape in layers_shapes]
                              for _ in range(num_clients)]
    for client in range(num_clients):
        for layer in range(num_layers):
            own_ponderated_weights[client][layer] = \
                clients_params[client][layer] * percentage[client]

    true_aggregation = helpers.sum_list_of_arrays(own_ponderated_weights, layers_shapes)

    for i in range(num_layers):
        assert np.array_equal(true_aggregation[i], aggregated_weights[i])
    assert len(aggregated_weights) == num_layers


def test_weighted_aggregated_weights_tuple_of_arrays(params_definition, helpers):
    """Checks that tuples of arrays of different shapes are correctly aggregated.

    A random weight is assigned to each client. """
    num_clients, _, _, \
        _, layers_shapes, _ = params_definition

    clients_params = []
    for i_client in range(num_clients):
        clients_params.append(tuple(np.random.rand(*shapes)
                                    for shapes in layers_shapes))

    percentage = np.random.dirichlet(np.ones(num_clients), size=1)[0]
    aggregator = WeightedFedAvgAggregator(percentage=percentage)
    aggregated_weights = aggregator.aggregate_weights(clients_params)

    own_ponderated_weights = []
    for i_client in range(num_clients):
        own_ponderated_weights.append(
            tuple(clients_params[i_client][i_params] * percentage[i_client]
                  for i_params in range(len(clients_params[0]))))

    true_aggregation = helpers.sum_list_of_arrays(own_ponderated_weights, layers_shapes)
    true_aggregation = tuple(true_aggregation)

    for i_params in range(len(clients_params[0])):
        assert np.array_equal(true_aggregation[i_params], aggregated_weights[i_params])
