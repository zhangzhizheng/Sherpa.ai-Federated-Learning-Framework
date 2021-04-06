import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

from shfl.model import DeepLearningModelPyTorch


class TensorDatasetIndex(TensorDataset):
    """
    TensorDataset that additionally returns the indices of samples.
    """

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors), index


class VerticalNeuralNetClient(DeepLearningModelPyTorch):
    """
    This class that represents a model of a Client node in a
    Vertical Federated Learning (FL) setting.
    Essentially, the collaborative training might take place using different
    features in each node, but the same training samples are used (i.e. each
    node possesses a vertical chunk of the features matrix).
    The key difference with respect Horizontal FL is that clients'
    models might differ from client to client (e.g. different model type
    and/or architecture), and might not even possess the target labels.
    In the present implementation a Pytorch model is employed, and it
    is assumed that the clients do not possess the target labels.

    # Arguments:
        model: Compiled model, ready to train
        criterion: Loss function to apply
        optimizer: Optimizer to apply
        batch_size: batch_size to apply
        epochs: Number of epochs
        metrics: Metrics for apply. Dictionary {name: function to apply, ...}.
            Default shows loss and accuracy
        device: Device where it will run. Default cpu
    """

    def __init__(self, model, loss, optimizer, batch_size=32,
                 epochs=1, metrics=None, device="cpu"):

        super().__init__(model, loss, optimizer, batch_size,
                         epochs, metrics, device)

        self._embeddings = None
        self._embeddings_indices = None
        self._batch_counter = 0
        self._epoch_counter = 0
        self._data_loader = None

    def train(self, data, labels, **kwargs):
        """
        Implementation of abstract method of class
        [TrainableModel](../Model/#trainablemodel-class).
        The training on the Vertical client node is comprised of two stages:
        1) Feedforward, where the embeddings are computed using the local model;
        2) Backpropagation of the received gradients of the loss with respect
           the embeddings.

        # Arguments
            data: Data, array-like of shape (n_samples, n_features)
            labels: Target classes, array-like of shape (n_samples,)
            kwargs: dictionary containing the gradients w.r.t. the embeddings
                and their indices (used only in the backpropagation stage
                of the training)
        """

        if "meta_params" not in kwargs:
            # Forward step:

            if self._data_loader is None:
                self._check_data(data)
                data = TensorDatasetIndex(data)
                data_loader = DataLoader(data, self._batch_size)
                self._data_loader = [item for item in data_loader]

            data_batch, self._embeddings_indices = \
                self._data_loader[self._batch_counter]

            self._embeddings = self._model(data_batch[0])

        else:
            # Backpropagation of received gradient:

            grad_embeddings, _ = kwargs.get("meta_params")
            grad_embeddings = torch.from_numpy(grad_embeddings)

            self._optimizer.zero_grad()
            self._embeddings.backward(gradient=grad_embeddings)
            self._optimizer.step()

            self._batch_counter += 1
            if self._batch_counter == len(self._data_loader):
                self._batch_counter = 0
                self._epoch_counter += 1

    def get_meta_params(self):
        """ Return computed embeddings. """

        return self._embeddings.detach().cpu().numpy(), \
            self._embeddings_indices.detach().cpu().numpy()


class VerticalNeuralNetServer(DeepLearningModelPyTorch):
    """
    This class that represents a model of a Server node in a
    Vertical Federated Learning (FL) setting.
    Essentially, the collaborative training might take place using different
    features in each node, but the same training samples are used (i.e. each
    node possesses a vertical chunk of the features matrix).
    The key difference with respect Horizontal FL is that clients'
    models might differ from client to client (e.g. different model type
    and/or architecture), and might not even possess the target labels.
    In the present implementation a Pytorch model is wrapped, and the server
    is assumed to possess the target labels.

    # Arguments:
        model: Compiled model, ready to train
        criterion: Loss function to apply
        optimizer: Optimizer to apply
        batch_size: batch_size to apply
        epochs: Number of epochs
        metrics: Metrics for apply. Dictionary {name: function to apply, ...}.
            Default roc_auc score
        device: Device where it will run. Default cpu
    """

    def __init__(self, model, loss, optimizer, batch_size=32,
                 epochs=1, metrics=None, device="cpu"):
        super().__init__(model, loss, optimizer, batch_size,
                         epochs, metrics, device)

        self._embeddings_indices = None
        self._grad_embeddings = None

    def train(self, data, labels, **kwargs):
        """
        Implementation of abstract method of class
        [TrainableModel](../Model/#trainablemodel-class).
        The local server's model is trained using as input the
        received embeddings. The gradients of the loss w.r.t. the received
        embeddings is computed and sent back to the clients.

        # Arguments
            data: Data, array-like of shape (n_samples, n_features)
            labels: Target classes, array-like of shape (n_samples,)
            kwargs: dictionary containing the clients' embeddings
                and their indices
        """

        embeddings, self._embeddings_indices = kwargs.get("meta_params")
        embeddings = [torch.from_numpy(i_embeddings)
                      for i_embeddings in embeddings]
        embeddings = torch.stack(embeddings, dim=0).sum(dim=0)
        embeddings = torch.autograd.Variable(embeddings, requires_grad=True)

        labels = torch.from_numpy(labels).float()[self._embeddings_indices]

        self._optimizer.zero_grad()
        prediction = self._model(embeddings)
        loss = self._loss(prediction, labels)
        loss.backward()
        self._optimizer.step()

        self._grad_embeddings = embeddings.grad

    def get_meta_params(self):
        """ Returns computed embeddings' gradients. """

        return self._grad_embeddings.detach().cpu().numpy(), \
            self._embeddings_indices
