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

# Tensorflow warning
# pylint: disable=no-name-in-module
from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf

from shfl.data_base.data_base import LabeledDatabase


class FashionMnist(LabeledDatabase):
    """Loads the FASHION-MNIST dataset.

    Implements base class [LabeledDatabase](./#labeleddatabase-class).

    # References:
        [FASHION-MNIST dataset](https://keras.io/datasets/
            #fashion-mnist-database-of-fashion-articles)
    """

    # False positive since using **kwargs
    # pylint: disable=arguments-differ
    def load_data(self):
        """Loads the train and test data.

        The data is originally already split into train and test.
        """

        ((self._train_data, self._train_labels),
         (self._test_data, self._test_labels)) = fashion_mnist.load_data()

        self._train_labels = tf.keras.utils.to_categorical(self._train_labels)
        self._test_labels = tf.keras.utils.to_categorical(self._test_labels)

        return self.data
