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

import numpy as np

from shfl.private.utils import unprotected_query


def test_identity_function(data_and_labels):
    """Checks that the identity query returns the plain input data unchanged."""
    input_data = data_and_labels[0]

    output_data = unprotected_query(input_data)

    np.testing.assert_array_equal(input_data, output_data)
