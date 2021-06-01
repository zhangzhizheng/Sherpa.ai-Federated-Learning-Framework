import numpy as np

from shfl.private.query import IdentityFunction
from shfl.private.query import Mean


def test_identity_function(data_and_labels):
    """Checks that the identity query returns the plain input data unchanged."""
    input_data = data_and_labels[0]
    query = IdentityFunction()

    output_data = query.get(input_data)

    np.testing.assert_array_equal(input_data, output_data)


def test_mean(data_and_labels):
    """Checks that the mean query returns the average of the input data."""
    input_data = data_and_labels[0]
    query = Mean()

    output_data = query.get(input_data)

    np.testing.assert_array_equal(output_data, np.mean(input_data))
