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
