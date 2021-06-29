# Disable warning from tensorflow callbacks: not our responsibility
# Disable too many arguments: needed for this case
# pylint: disable=no-name-in-module, too-many-arguments

import copy
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from shfl.model.model import TrainableModel
from .utils import check_labels_size


class DeepLearningModel(TrainableModel):
    """Wraps Keras and Tensorflow models.

    Implements the class [TrainableModel](../#trainablemodel-class).

    # Arguments:
        model: Compiled model, ready to train.
        loss: Loss function.
        optimizer: Optimizer.
        batch_size: Optional; batch size.
        epochs: Optional; Number of epochs.
        metrics: Optional; List of metrics to use in the evaluation.

    # References:
        [TensorFlow](https://www.tensorflow.org/)
    """

    def __init__(self, model, loss, optimizer,
                 batch_size=None, epochs=1, metrics=None):
        self._model = model
        self._in_out_sizes = (model.layers[0].get_input_shape_at(0)[1:],
                              model.layers[-1].get_output_shape_at(0)[1:])
        self._batch_size = batch_size
        self._epochs = epochs
        self._loss = loss
        self._optimizer = optimizer
        self._metrics = metrics

        self._model.compile(optimizer=self._optimizer,
                            loss=self._loss,
                            metrics=self._metrics)

    def train(self, data, labels, **kwargs):
        """Trains the model.

        # Arguments:
            data: Array-type object containing the data to train the model.
            labels: Array-type object containing the target labels.
            **kwargs: Optional named parameters.
        """
        self._check_data(data)
        check_labels_size(self._in_out_sizes, labels)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5,
                                       verbose=0, mode='min')
        self._model.fit(x=data, y=labels, batch_size=self._batch_size,
                        epochs=self._epochs, validation_split=0.2,
                        verbose=0, shuffle=False, callbacks=[early_stopping])

    def predict(self, data):
        """Makes a prediction on input data.

        # Arguments:
            data: Array-type object containing the input data
                on which to make the prediction.

        # Returns:
            prediction: Model's prediction using the input data.
        """
        self._check_data(data)

        return self._model.predict(data, batch_size=self._batch_size).argmax(axis=-1)

    def evaluate(self, data, labels):
        """Evaluates the performance of the model.

        # Arguments:
            data: Array-type object containing the data
                on which to make the evaluation.
            labels: Array-type object containing the true labels.

        # Returns:
            metrics: Metrics for the evaluation.
        """
        self._check_data(data)
        check_labels_size(self._in_out_sizes, labels)

        return self._model.evaluate(data, labels, verbose=0)

    def performance(self, data, labels):
        """Evaluates the performance of the model using
            the most representative metrics.

        # Arguments:
            data: Array-type object containing the data
                on which to make the evaluation.
            labels: Array-type object containing the true labels.

        # Returns:
            metrics: Most representative metrics for the evaluation.
        """
        self._check_data(data)
        check_labels_size(self._in_out_sizes, labels)

        return self._model.evaluate(data, labels, verbose=0)[0]

    def get_model_params(self):
        """See base class."""
        return self._model.get_weights()

    def set_model_params(self, params):
        """See base class."""
        self._model.set_weights(params)

    def _check_data(self, data):
        """Checks if the data dimension is correct.
        """
        if data.shape[1:] != self._in_out_sizes[0]:
            raise AssertionError(
                "Data need to have the same shape described by the model " +
                str(self._in_out_sizes[0]) + " .Current data has shape " +
                str(data.shape[1:]) + ".")

    def __deepcopy__(self, memo):
        """Overwrites deepcopy method.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key == "_model":
                model = tf.keras.models.clone_model(value)
                model.set_weights(value.get_weights())
                setattr(result, key, model)
            else:
                setattr(result, key, copy.deepcopy(value, memo))
        result._model.compile(optimizer=result._optimizer,
                              loss=result._loss,
                              metrics=result._metrics)
        return result
