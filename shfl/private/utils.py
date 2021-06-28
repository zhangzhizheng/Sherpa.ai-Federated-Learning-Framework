import random
import numpy as np


def mean_query(data):
    """Computes the mean over data."""
    return np.mean(data)


def unprotected_query(data):
    """Returns the data. """
    return data


def normalize_query(data, mean, std):
    """Applies a normalization over the input data.

    # Arguments:
        data: Input data.
        mean: Mean used for the normalization.
        std: Standard deviation used for the normalization.
    """
    data.data = (data.data - mean) / std


def shuffle_node_query(data):
    """Shuffles the target labels in a node."""
    random.shuffle(data.label)
