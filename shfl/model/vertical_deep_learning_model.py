import torch
import numpy as np


from shfl.model.deep_learning_model_pt import DeepLearningModelPyTorch


class VerticalNeuralNetClient(DeepLearningModelPyTorch):
    """Represents a **client's model** in a Vertical Federated Learning setting.

    Implements the class [DeepLearningModelPyTorch](./#deeplearningmodelpytorch).

    In Vertical Federated Learning (FL), the collaborative training
    might take place using different features in each node,
    but the same training samples are used (i.e. each
    node possesses a vertical chunk of the features matrix).
    As opposed to Horizontal FL, in Vertical FL the clients
    might all have different models (e.g. different model type
    and/or architecture), and the clients might not even possess
    the target labels. See also the documentation of the class
    [VerticalServerDataNode](../../private/federated_operation/#verticalserverdatanode-class).

    # Arguments:
        See base class.

    # References:
        [VAFL, a Method of Vertical Asynchronous
            Federated Learning](https://arxiv.org/abs/2007.06081)

        [Federated Machine Learning:
            Concept and Applications](https://arxiv.org/abs/1902.04885)
    """

    _embeddings = None
    _embeddings_indices = None
    _batch_counter = 0
    _epoch_counter = 0

    def train(self, data, labels, **kwargs):
        """Trains the model.

        The training on the Vertical client node is comprised of two stages:
        1) Feedforward, where the embeddings are computed using the local model;
        2) Backpropagation of the received gradients of the loss with respect
           the embeddings.

        # Arguments:
            data: Data, array-like object of shape (n_samples, n_features).
            labels: Target classes, array-like object of shape
                (n_samples, n_classes).
            kwargs: Optional; Dictionary containing the gradients w.r.t.
                the embeddings and samples' indices, assumed to be
                Numpy arrays of shape (n_samples, n_embeddings) and
                (n_samples, ), respectively. It is used only in the
                backpropagation stage of the training.
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
        """Returns the computed embeddings and associated indices."""

        return self._embeddings.detach().cpu().numpy(), \
            self._embeddings_indices


class VerticalNeuralNetServer(DeepLearningModelPyTorch):
    """Represents a **server's model** in a Vertical Federated Learning setting.

    Implements the class [DeepLearningModelPyTorch](./#deeplearningmodelpytorch).

    In Vertical Federated Learning (FL), the collaborative training
    might take place using different features in each node,
    but the same training samples are used (i.e. each
    node possesses a vertical chunk of the features matrix).
    As opposed to Horizontal FL, in Vertical FL the clients
    might all have different models (e.g. different model type
    and/or architecture), and the clients might not even possess
    the target labels. See also the documentation of the class
    [VerticalServerDataNode](../../private/federated_operation/#verticalserverdatanode-class).

    # Arguments:
        See base class.

    # References:
        [VAFL, a Method of Vertical Asynchronous
            Federated Learning](https://arxiv.org/abs/2007.06081)

        [Federated Machine Learning:
            Concept and Applications](https://arxiv.org/abs/1902.04885)
    """

    _embeddings_indices = None
    _grad_embeddings = None

    def train(self, data, labels, **kwargs):
        """Implementation of abstract method of class
        [TrainableModel](../Model/#trainablemodel-class).

        The local server's model is trained using as input the
        received embeddings from the clients. The gradients of the loss w.r.t.
        the received embeddings is computed.

        # Arguments
            data: Data, array-like of shape (n_samples, n_features).
            labels: Target classes, array-like of shape (n_samples, n_classes).
            kwargs: Dictionary containing the clients' embeddings
                and samples' indices, assumed to be Numpy arrays of shape
                (n_samples, n_embeddings) and (n_samples, ), respectively.
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
        """Returns gradient of loss with respect clients' embeddings and
            associated indices.
        """

        return self._grad_embeddings.detach().cpu().numpy(), \
            self._embeddings_indices
