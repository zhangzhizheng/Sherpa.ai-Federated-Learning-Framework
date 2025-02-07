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

from shfl.federated_aggregator.fedavg_aggregator import FedAvgAggregator


def test_aggregated_weights_list_of_arrays(params_definition, helpers):
    """Checks that the parameters are correctly aggregated for lists of arrays
    of different sizes."""
    _, _, _, num_layers, layers_shapes, clients_params = params_definition

    aggregator = FedAvgAggregator()
    aggregated_weights = aggregator(clients_params)

    true_aggregation = helpers.average_list_of_arrays(clients_params, layers_shapes)

    for i in range(num_layers):
        assert np.array_equal(true_aggregation[i], aggregated_weights[i])
    assert len(aggregated_weights) == num_layers


def test_aggregated_weights_multidimensional_2d_array(params_definition, helpers):
    """Checks that the parameters are correctly aggregated for 2d arrays."""
    num_clients, num_rows, num_cols, _, _, _ = params_definition

    clients_params = [np.random.rand(num_rows, num_cols)
                      for _ in range(num_clients)]

    aggregator = FedAvgAggregator()
    aggregated_weights = aggregator(clients_params)

    true_aggregation = helpers.average_arrays(clients_params, (num_rows, num_cols))

    assert np.array_equal(true_aggregation, aggregated_weights)
    assert aggregated_weights.shape == true_aggregation.shape


def test_aggregated_weights_multidimensional_3d_array(params_definition, helpers):
    """Checks that the parameters are correctly aggregated for 3d arrays."""
    num_clients, num_rows, num_cols, _, _, _ = params_definition
    num_k = 7

    clients_params = [np.random.rand(num_rows, num_cols, num_k)
                      for _ in range(num_clients)]

    aggregator = FedAvgAggregator()
    aggregated_weights = aggregator(clients_params)

    true_aggregation = helpers.average_arrays(clients_params, (num_rows, num_cols, num_k))

    assert np.array_equal(true_aggregation, aggregated_weights)
    assert aggregated_weights.shape == true_aggregation.shape
