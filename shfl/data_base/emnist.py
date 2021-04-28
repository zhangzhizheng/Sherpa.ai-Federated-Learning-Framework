import numpy as np
import emnist

from shfl.data_base.data_base import DataBase


class Emnist(DataBase):
    """Loads the EMNIST dataset.

    Implements base class [DataBase](./#database-class).

    # References:
        [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
    """

    def load_data(self):
        self._train_data, self._train_labels = emnist.extract_training_samples('digits')
        self._train_labels = np.eye(10)[self._train_labels]
        self._test_data, self._test_labels = emnist.extract_test_samples('digits')
        self._test_labels = np.eye(10)[self._test_labels]

        return self.data
