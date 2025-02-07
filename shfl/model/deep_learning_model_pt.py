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

# Disable too many arguments and instances: needed for Pytorch train loop
# pylint: disable=too-many-arguments, too-many-instance-attributes
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from shfl.model.model import TrainableModel
from .utils import  check_labels_size


class DeepLearningModelPyTorch(TrainableModel):
    """Wraps PyTorch models.

    Implements the class [TrainableModel](../#trainablemodel-class).

    # Arguments:
        model: Compiled model, ready to train.
        loss: Loss function.
        optimizer: Optimizer.
        batch_size: Optional; batch size.
        epochs: Optional; Number of epochs.
        metrics: Optional; Dictionary {name: function to apply, ...}
            containing the metrics for the evaluation (default shows loss).
        device: Optional; Device where to run (default is cpu).

    # References:
        [PyTorch](https://pytorch.org/)
    """

    def __init__(self, model, loss, optimizer,
                 batch_size=32, epochs=1, metrics=None, device="cpu"):
        self._model = model
        self._in_out_sizes = (self.get_model_params()[0].shape[1],
                              self.get_model_params()[-1].shape)
        self._loss = loss
        self._optimizer = optimizer

        self._batch_size = batch_size
        self._epochs = epochs
        self._device = device
        self._metrics = metrics

    def train(self, data, labels, **kwargs):
        """Trains the model.

        # Arguments:
            data: Numpy array containing the data to train the model.
            labels: Numpy array containing the target labels.
            **kwargs: Optional named parameters.
        """
        self._check_data(data)
        check_labels_size(self._in_out_sizes, labels)

        dataset = TensorDataset(torch.from_numpy(data),
                                torch.from_numpy(labels))
        train_loader = DataLoader(dataset, self._batch_size, False)

        self._model.to(self._device)
        for _ in range(self._epochs):
            for element in train_loader:
                inputs, y_true = element[0].float().to(self._device), \
                                 element[1].float().to(self._device)

                self._optimizer.zero_grad()

                y_predicted = self._model(inputs)

                if y_true.shape[1] > 1:
                    y_true = torch.argmax(y_true, -1)
                loss = self._loss(y_predicted, y_true)

                self._model.zero_grad()

                loss.backward()
                self._optimizer.step()

    def predict(self, data):
        """Makes a prediction on input data.

        # Arguments:
            data: Numpy array containing the input data
                on which to make the prediction.

        # Returns:
            prediction: Model's prediction using the input data.
        """
        self._check_data(data)

        dataset = TensorDataset(torch.from_numpy(data))
        data_loader = DataLoader(dataset, self._batch_size, False)

        y_predicted = []
        self._model.to(self._device)
        with torch.no_grad():
            for element in data_loader:
                inputs = element[0].float().to(self._device)

                batch_y_predicted = self._model(inputs)

                y_predicted.extend(batch_y_predicted.cpu().numpy())

        return np.array(y_predicted)

    def evaluate(self, data, labels):
        """Evaluates the performance of the model.

        # Arguments:
            data: Numpy array containing the data
                on which to make the evaluation.
            labels: Numpy array containing the true labels.

        # Returns:
            metrics: Metrics for the evaluation.
        """

        self._check_data(data)
        check_labels_size(self._in_out_sizes, labels)

        with torch.no_grad():
            all_y_pred = self.predict(data)

            all_y_pred = torch.from_numpy(all_y_pred).float()
            labels_t = torch.from_numpy(labels).float()
            if labels_t.shape[1] > 1:
                labels_t = torch.argmax(labels_t, -1)
            val_loss = self._loss(all_y_pred, labels_t)

            metrics = [val_loss.item()]
            if self._metrics is not None:
                for _, metric in self._metrics.items():
                    metrics.append(metric(all_y_pred.cpu().numpy(), labels))

        return metrics

    def performance(self, data, labels):
        """Evaluates the performance of the model using
            the most representative metrics.

        # Arguments:
            data: Numpy array containing the data
                on which to make the evaluation.
            labels: Numpy array object containing the true labels.

        # Returns:
            metrics: Most representative metrics for the evaluation.
        """
        return self.evaluate(data, labels)[0]

    def get_model_params(self):
        """See base class."""
        weights = []
        for param in self._model.parameters():
            weights.append(param.cpu().data.numpy())

        return weights

    def set_model_params(self, params):
        """See base class."""
        with torch.no_grad():
            for ant, post in zip(self._model.parameters(), params):
                ant.data = torch.from_numpy(post).float()

    def _check_data(self, data):
        if data.shape[1] != self._in_out_sizes[0]:
            raise AssertionError(
                "Data need to have the same dimension described by the model " +
                str(self._in_out_sizes[0]) + ". Current data have dimension " +
                str(data.shape[1]) + ".")
