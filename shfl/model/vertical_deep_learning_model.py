import torch
import numpy as np

from shfl.model import DeepLearningModelPyTorch


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
    In the present implementation a Pytorch model is employed.

    # Arguments:
        model: Compiled model, ready to train
        criterion: Loss function to apply
        optimizer: Optimizer to apply
        batch_size: batch_size to apply
        epochs: Number of epochs
        metrics: Metrics for apply. Dictionary {name: function to apply, ...}.
            Default shows loss.
        device: Device where it will run. Default is cpu.

    # References:
        [VAFL: a Method of Vertical Asynchronous Federated Learning]
        (https://arxiv.org/abs/2007.06081)
    """

    def __init__(self, model, loss, optimizer, batch_size=32,
                 epochs=1, metrics=None, device="cpu"):

        super().__init__(model, loss, optimizer, batch_size,
                         epochs, metrics, device)

        self._embeddings = None
        self._embeddings_indices = None
        self._batch_counter = 0
        self._epoch_counter = 0

    def train(self, data, labels, **kwargs):
        """
        Implementation of abstract method of class
        [TrainableModel](../Model/#trainablemodel-class).
        The training on the Vertical client node is comprised of two stages:
        1) Feedforward, where the embeddings are computed using the local model;
        2) Backpropagation of the received gradients of the loss with respect
           the embeddings.

        # Arguments:
            data: Data, array-like of shape (n_samples, n_features)
            labels: Target classes, array-like of shape (n_samples,)
            kwargs: dictionary containing the gradients w.r.t. the embeddings
                and samples' indices, assumed to be Numpy arrays of shape
                (n_samples, n_embeddings) and (n_samples), respectively.
                This is used only in the backpropagation stage
                of the training.
        """

        if not kwargs:
            # Forward step:

            start_index = self._batch_counter * self._batch_size
            end_index = start_index + self._batch_size
            if end_index > len(data) - 1:
                end_index = len(data)
            self._embeddings_indices = np.arange(start_index, end_index)
            data_batch = torch.as_tensor(data[self._embeddings_indices],
                                         device=self._device)
            self._embeddings = self._model(data_batch)

        else:
            # Backpropagation of received gradient:

            grad_embeddings, _ = kwargs.get("meta_params")
            grad_embeddings = torch.as_tensor(grad_embeddings,
                                              device=self._device)

            self._optimizer.zero_grad()
            self._embeddings.backward(gradient=grad_embeddings)
            self._optimizer.step()

            self._batch_counter += 1
            if self._embeddings_indices[-1] == (len(data) - 1):
                self._batch_counter = 0
                self._epoch_counter += 1

    def get_meta_params(self):
        """ Return computed embeddings and associated indices. """

        return self._embeddings.detach().cpu().numpy(), \
            self._embeddings_indices


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
    In the present implementation a Pytorch model is wrapped.

    # Arguments:
        model: Compiled model, ready to train
        criterion: Loss function to apply
        optimizer: Optimizer to apply
        batch_size: batch_size to apply
        epochs: Number of epochs
        metrics: Metrics for apply. Dictionary {name: function to apply, ...}.
            Default is loss.
        device: Device where it will run. Default is cpu.

    # References:
        [VAFL: a Method of Vertical Asynchronous Federated Learning]
        (https://arxiv.org/abs/2007.06081)
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
        received embeddings. The gradients of the loss w.r.t.
        the received embeddings is computed.

        # Arguments
            data: Data, array-like of shape (n_samples, n_features)
            labels: Target classes, array-like of shape (n_samples,)
            kwargs: dictionary containing the clients' embeddings
                and samples' indices, assumed to be Numpy arrays of shape
                (n_samples, n_embeddings) and (n_samples), respectively.
        """

        embeddings, self._embeddings_indices = kwargs.get("meta_params")
        embeddings = torch.as_tensor(embeddings,
                                     device=self._device)
        embeddings = torch.autograd.Variable(embeddings, requires_grad=True)
        labels = torch.as_tensor(labels[self._embeddings_indices],
                                 device=self._device)

        self._optimizer.zero_grad()
        prediction = self._model(embeddings)
        loss = self._loss(prediction, labels)
        loss.backward()
        self._optimizer.step()

        self._grad_embeddings = embeddings.grad

    def get_meta_params(self):
        """
        Returns gradient of loss with respect the embeddings and
        samples' indices.
        """

        return self._grad_embeddings.detach().cpu().numpy(), \
            self._embeddings_indices
