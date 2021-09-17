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

from shfl.private.data import LabeledData


def test_initialization(data_and_labels):
    """Checks that the labeled data is correctly initialized."""
    labeled_data = LabeledData(*data_and_labels)

    assert hasattr(labeled_data, "_data")
    assert hasattr(labeled_data, "_label")


def test_get_data_and_labels(data_and_labels):
    """Checks that the labeled data gets the data correctly."""
    labeled_data = LabeledData(*data_and_labels)

    np.testing.assert_array_equal(labeled_data.data, data_and_labels[0])
    np.testing.assert_array_equal(labeled_data.label, data_and_labels[1])


def test_set_data_and_labels(data_and_labels):
    """Checks that the labeled data sets the data correctly."""
    labeled_data = LabeledData(data=None, label=None)
    new_data = np.random.rand(*data_and_labels[0].shape)
    new_label = np.random.rand(*data_and_labels[1].shape)
    labeled_data.data = new_data
    labeled_data.label = new_label

    np.testing.assert_array_equal(labeled_data.data, new_data)
    np.testing.assert_array_equal(labeled_data.label, new_label)
