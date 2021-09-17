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


class Helpers:
    """Delivers static helper functions to avoid duplicated code."""

    @staticmethod
    def check_wrong_data(wrapped_model, data, labels):
        """Checks that the wrapped model raises an error if wrong shape
        input data are used."""
        wrong_data_sets = (Helpers.append_column(data),
                           Helpers.remove_column(data))

        for wrong_data in wrong_data_sets:
            with pytest.raises(AssertionError):
                wrapped_model.train(wrong_data, labels)

    @staticmethod
    def append_column(data_array):
        """Appends a column to the input array.

        For simplicity, the last column is replicated."""
        return np.concatenate((data_array, data_array[:, [-1]]), axis=1)

    @staticmethod
    def remove_column(data_array):
        """Removes a column from the input array."""
        return data_array[:, :-1]


@pytest.fixture
def helpers():
    """Returns the helpers class."""
    return Helpers
