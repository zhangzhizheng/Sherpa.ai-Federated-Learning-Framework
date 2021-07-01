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
