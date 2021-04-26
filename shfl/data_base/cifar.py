from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
import tensorflow as tf

from shfl.data_base.data_base import DataBase


class Cifar10(DataBase):
    """Loads the CIFAR10 dataset.

    # References:
        [CIFAR10 dataset](https://keras.io/api/datasets/cifar10)
    """
    def load_data(self):
        ((self._train_data, self._train_labels),
         (self._test_data, self._test_labels)) = cifar10.load_data()

        self._train_labels = tf.keras.utils.to_categorical(self._train_labels)
        self._test_labels = tf.keras.utils.to_categorical(self._test_labels)
        
        return self.data


class Cifar100(DataBase):
    """Loads the CIFAR100 dataset.

    # References:
        [CIFAR100 dataset](https://keras.io/api/datasets/cifar100)
    """
    def load_data(self):
        ((self._train_data, self._train_labels),
         (self._test_data, self._test_labels)) = cifar100.load_data()

        self._train_labels = tf.keras.utils.to_categorical(self._train_labels)
        self._test_labels = tf.keras.utils.to_categorical(self._test_labels)
        
        return self.data
