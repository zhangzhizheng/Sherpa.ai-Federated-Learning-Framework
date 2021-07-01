import numpy as np

from shfl.private.utils import unprotected_query


def test_identity_function(data_and_labels):
    """Checks that the identity query returns the plain input data unchanged."""
    input_data = data_and_labels[0]

    output_data = unprotected_query(input_data)

    np.testing.assert_array_equal(input_data, output_data)
