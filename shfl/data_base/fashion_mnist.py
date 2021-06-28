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
