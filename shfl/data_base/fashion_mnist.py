from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf

from shfl.data_base.data_base import DataBase


class FashionMnist(DataBase):
    """Loads the FASHION-MNIST dataset.

    # References:
        [FASHION-MNIST dataset](https://keras.io/datasets/
            #fashion-mnist-database-of-fashion-articles)
    """

    def load_data(self):
        ((self._train_data, self._train_labels),
         (self._test_data, self._test_labels)) = fashion_mnist.load_data()

        self._train_labels = tf.keras.utils.to_categorical(self._train_labels)
        self._test_labels = tf.keras.utils.to_categorical(self._test_labels)

        return self.data
