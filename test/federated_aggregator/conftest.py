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

"""Contains fixtures used across the module."""
import pytest
import numpy as np


@pytest.fixture()
def params_definition():
    """Defines the parameters to be aggregated.

    The parameters are defined by a list of 2d arrays."""
    num_clients = 10
    num_rows = 129
    num_cols = 65
    num_layers = 6
    layers_shapes = [[np.random.randint(low=1, high=num_rows),
                      np.random.randint(low=1, high=num_cols)]
                     for _ in range(num_layers)]

    clients_params = []
    for _ in range(num_clients):
        clients_params.append([np.random.rand(*shapes)
                               for shapes in layers_shapes])

    return num_clients, num_rows, num_cols, \
        num_layers, layers_shapes, clients_params


class Helpers:
    """Delivers static helper functions to avoid duplicated code."""

    @staticmethod
    def sum_list_of_arrays(clients_params, layers_shapes):
        """Computes the sum of a nested list of arrays item-wise."""
        num_clients = len(clients_params)
        num_layers = len(layers_shapes)
        aggregated_params = [np.zeros(shape=shapes) for shapes in layers_shapes]
        for client in range(num_clients):
            for layer in range(num_layers):
                aggregated_params[layer] += clients_params[client][layer]

        return aggregated_params

    @staticmethod
    def average_list_of_arrays(clients_params, layers_shapes):
        """Computes the average of a nested list of arrays item-wise."""
        num_clients = len(clients_params)
        aggregated_params = Helpers.sum_list_of_arrays(clients_params, layers_shapes)
        aggregated_params = [params / num_clients for params in aggregated_params]

        return aggregated_params

    @staticmethod
    def sum_arrays(clients_params, array_shape):
        """Computes the sum of a set of arrays of the same shape."""
        num_clients = len(clients_params)
        aggregated_params = np.zeros(shape=array_shape)
        for client in range(num_clients):
            aggregated_params += clients_params[client]

        return aggregated_params

    @staticmethod
    def average_arrays(clients_params, array_shape):
        """Computes the average of a set of arrays of the same shape."""
        num_clients = len(clients_params)
        aggregated_params = Helpers.sum_arrays(clients_params, array_shape)
        aggregated_params /= num_clients

        return aggregated_params


@pytest.fixture
def helpers():
    """Returns the helpers class."""
    return Helpers
