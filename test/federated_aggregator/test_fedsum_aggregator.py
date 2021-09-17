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

import pytest
import numpy as np

from shfl.federated_aggregator.fedsum_aggregator import FedSumAggregator


@pytest.mark.parametrize("iterable_type", [list, tuple])
def test_aggregate_list_or_tuple_of_numpy_arrays(iterable_type,
                                                 params_definition,
                                                 helpers):
    """Checks that lists and tuples of arrays of variable size are properly aggregated.

    A list or tuple of arrays must be returned as aggregated parameters."""
    num_clients, _, _, \
        _, layers_shapes, _ = params_definition

    clients_params = []
    for _ in range(num_clients):
        clients_params.append(iterable_type(np.random.rand(*shapes)
                                            for shapes in layers_shapes))

    aggregator = FedSumAggregator()
    aggregated_weights = aggregator(clients_params)

    true_aggregation = helpers.sum_list_of_arrays(clients_params, layers_shapes)

    for true, computed in zip(true_aggregation, aggregated_weights):
        np.testing.assert_array_equal(true, computed)
    assert isinstance(aggregated_weights, iterable_type)


def test_aggregate_single_numpy_arrays(params_definition, helpers):
    """Checks that a single arrays is properly aggregated."""
    num_clients, num_rows, num_cols, \
        _, _, _ = params_definition
    num_k = 12

    clients_params = []
    for _ in range(num_clients):
        clients_params.append(np.random.rand(num_rows, num_cols, num_k))

    aggregator = FedSumAggregator()
    aggregated_weights = aggregator(clients_params)

    true_aggregation = helpers.sum_arrays(clients_params,
                                          (num_rows, num_cols, num_k))

    np.testing.assert_array_equal(true_aggregation, aggregated_weights)
    assert isinstance(aggregated_weights, np.ndarray)


def test_aggregate_single_scalar(params_definition):
    """Checks that single scalars are properly aggregated."""
    num_clients, _, _, _, _, _ = params_definition

    clients_params = []
    for _ in range(num_clients):
        clients_params.append(np.random.rand())

    true_aggregation = 0.0
    for i_client in range(num_clients):
        true_aggregation += clients_params[i_client]

    aggregator = FedSumAggregator()
    aggregated_weights = aggregator(clients_params)
    np.testing.assert_almost_equal(true_aggregation,
                                   aggregated_weights,
                                   decimal=8)
    assert isinstance(aggregated_weights, np.ScalarType)
