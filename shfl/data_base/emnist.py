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
import emnist

from shfl.data_base.data_base import LabeledDatabase


class Emnist(LabeledDatabase):
    """Loads the EMNIST dataset.

    Implements base class [LabeledDatabase](./#labeleddatabase-class).

    # References:
        [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
    """

    # False positive since using **kwargs
    # pylint: disable=arguments-differ
    def load_data(self):
        """Loads the train and test data.

        The data is originally already split into train and test.
        """

        self._train_data, self._train_labels = emnist.extract_training_samples('digits')
        self._train_labels = np.eye(10)[self._train_labels]
        self._test_data, self._test_labels = emnist.extract_test_samples('digits')
        self._test_labels = np.eye(10)[self._test_labels]

        return self.data
