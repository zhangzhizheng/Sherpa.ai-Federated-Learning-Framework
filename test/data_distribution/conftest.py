"""Contains fixtures used across the module."""
import pytest
import numpy as np

from shfl.data_base.data_base import LabeledDatabase


class LabeledDatabaseTest(LabeledDatabase):
    """Creates a test class for a random data base."""

    def __init__(self, data, labels):
        super().__init__()
        self._data = data
        self._labels = labels

    def load_data(self, train_proportion=0.8, shuffle=True):
        """Loads the train and test data."""
        self.split_data(train_proportion, shuffle)

        return self.data


@pytest.fixture
def labeled_data_base():
    """Returns the helpers class."""
    return LabeledDatabaseTest


@pytest.fixture(name="data_and_labels_arrays")
def fixture_data_and_labels_arrays():
    """Returns data and labels arrays containing random values."""
    data = np.random.rand(60, 12)
    labels = np.random.randint(0, 2, size=(60,))

    return data, labels
