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


@pytest.fixture(name="data_and_labels_arrays")
def fixture_data_and_labels_arrays():
    """Returns data and labels arrays containing random values."""
    n_samples = 100
    n_features = 5
    data = np.random.rand(n_samples, n_features)
    labels = np.random.randint(0, 2, size=(n_samples,))

    return data, labels
