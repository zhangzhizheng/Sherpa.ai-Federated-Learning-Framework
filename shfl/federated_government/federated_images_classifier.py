from enum import Enum
import numpy as np
import tensorflow as tf

from shfl.federated_government.federated_government import FederatedGovernment
from shfl.federated_aggregator.fedavg_aggregator import FedAvgAggregator
from shfl.data_distribution.data_distribution_iid import IidDataDistribution
from shfl.data_distribution.data_distribution_non_iid import NonIidDataDistribution
from shfl.private.federated_operation import FederatedTransformation
from shfl.model.deep_learning_model import DeepLearningModel
from shfl.data_base.emnist import Emnist
from shfl.data_base.fashion_mnist import FashionMnist
from shfl.private.federated_operation import Normalize


class FederatedImagesClassifier(FederatedGovernment):
    """Runs a federated image classification with minimal user input.

    It overrides the class [FederatedGovernment](./#federatedgovernment-class).

    Runs an image classification federated learning experiment
    with predefined values. This way, it suffices to just specify
    which dataset to use.

    # Arguments:
        data_base_name_key: Key of a valid data base (see possibilities in class
            [ImagesDataBases](./#imagesdatabases-class)).
        iid: Optional; Boolean specifying whether the data distribution IID or
            non-IID. By default set to `iid=True`.
        num_nodes: Optional; number of client nodes (default is 20).
        percent: Optional; Percentage of the database to distribute
            among nodes (by default set to 100, in which case
            all the available data is used).
    """

    def __init__(self, data_base_name_key, iid=True, num_nodes=20, percent=100):
        if data_base_name_key in ImagesDataBases.__members__.keys():
            module = ImagesDataBases.__members__[data_base_name_key].value
            data_base = module()
            train_data, train_labels, \
                test_data, test_labels = data_base.load_data()

            if iid:
                distribution = IidDataDistribution(data_base)
            else:
                distribution = NonIidDataDistribution(data_base)

            federated_data, self._test_data, self._test_labels = \
                distribution.get_federated_data(num_nodes=num_nodes,
                                                percent=percent)

            self._test_data = np.reshape(
                self._test_data,
                (self._test_data.shape[0],
                 self._test_data.shape[1],
                 self._test_data.shape[2], 1))

            federated_data.apply_data_transformation(Reshape())
            mean = np.mean(train_data.data)
            std = np.std(train_data.data)
            federated_data.apply_data_transformation(Normalize(mean, std))

            aggregator = FedAvgAggregator()

            super().__init__(self.model_builder(), federated_data, aggregator)

        else:
            raise ValueError("The data base " + data_base_name_key +
                             " is not included. Try with: " +
                             str(", ".join([e.name for e in ImagesDataBases])))

    def run_rounds(self, n=5, **kwargs):
        """See base class.
        """
        super().run_rounds(n, self._test_data, self._test_labels, **kwargs)

    @staticmethod
    def model_builder():
        """Creates a Tensorflow model for image classification.

        # Returns:
            model: Object of class
                [DeepLearningModel](../model/supervised/#deeplearningmodel),
                the Tensorflow model to use.
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), padding='same',
            activation='relu', strides=1, input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=2, strides=2, padding='valid'))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), padding='same',
            activation='relu', strides=1))
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=2, strides=2, padding='valid'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        criterion = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.RMSprop()
        metrics = [tf.keras.metrics.CategoricalAccuracy()]

        return DeepLearningModel(model=model, loss=criterion,
                                 optimizer=optimizer, metrics=metrics)


class Reshape(FederatedTransformation):
    """Reshapes the data in the set of federated nodes.
    """
    def apply(self, labeled_data):
        """See base class."""
        labeled_data.data = np.reshape(
            labeled_data.data,
            (labeled_data.data.shape[0],
             labeled_data.data.shape[1],
             labeled_data.data.shape[2], 1))


class ImagesDataBases(Enum):
    """Enumerates the available databases for image classification.

    Options are: `"EMNIST", "FASHION_EMNIST"`.
    """
    EMNIST = Emnist
    FASHION_EMNIST = FashionMnist
