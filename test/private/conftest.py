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


@pytest.fixture(name="data_and_labels")
def fixture_data_and_labels():
    """Returns a random data set with labels."""
    num_data = 50
    n_features = 9
    n_targets = 3
    data = np.random.rand(num_data, n_features)
    labels = np.random.randint(low=0, high=10, size=(num_data, n_targets))

    return data, labels
