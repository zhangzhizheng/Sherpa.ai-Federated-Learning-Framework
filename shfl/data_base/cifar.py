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
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
import tensorflow as tf

from shfl.data_base.data_base import LabeledDatabase


class Cifar10(LabeledDatabase):
    """Loads the CIFAR10 dataset.

    Implements base class [LabeledDatabase](./#labeleddatabase-class).

    # References:
        [CIFAR10 dataset](https://keras.io/api/datasets/cifar10)
    """

    # False positive since using **kwargs
    # pylint: disable=arguments-differ
    def load_data(self):
        """Loads the train and test data.

        The data is originally already split into train and test.
        """

        ((self._train_data, self._train_labels),
         (self._test_data, self._test_labels)) = cifar10.load_data()

        self._train_labels = tf.keras.utils.to_categorical(self._train_labels)
        self._test_labels = tf.keras.utils.to_categorical(self._test_labels)

        return self.data


class Cifar100(LabeledDatabase):
    """Loads the CIFAR100 dataset.

    Implements base class [LabeledDatabase](./#labeleddatabase-class).

    # References:
        [CIFAR100 dataset](https://keras.io/api/datasets/cifar100)
    """

    # False positive since using **kwargs
    # pylint: disable=arguments-differ
    def load_data(self):
        """Loads the train and test data.

        The data is originally already split into train and test.
        """
        ((self._train_data, self._train_labels),
         (self._test_data, self._test_labels)) = cifar100.load_data()

        self._train_labels = tf.keras.utils.to_categorical(self._train_labels)
        self._test_labels = tf.keras.utils.to_categorical(self._test_labels)

        return self.data
