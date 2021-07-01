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
