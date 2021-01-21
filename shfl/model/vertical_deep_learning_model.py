import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from .vertical_deep_learning_utils import _expit_b, _logsig, _linear_activation_forward, _linear_activation_backward
# from utils.utils import UniformDistribution
# from shfl.differential_privacy import SensitivitySampler
# from shfl.differential_privacy import L1SensitivityNorm
from shfl.model.model import TrainableModel


class VerticalNeuralNetClient(TrainableModel):
    """
    Class that represents a data client node (passive node) for the
    Vertical Federated Learning using Neural Network as local model.
    """

    def __init__(self,
                 n_features=1,
                 layer_dims=None,
                 params={"lam": 0.01, "learning_rate": 0.001},
                 epsilon=None):
        self._layer_dims = layer_dims if layer_dims is not None else []
        self._deploy_model(n_features)

        self._lam = params["lam"]
        self._learning_rate = params["learning_rate"]
        self._epsilon = epsilon
        self._sensitivity = None
        self._an, self._cache = None, None

    def _deploy_model(self, n_features):
        self._nn_model = NeuralNetHelper(self._layer_dims)
        self._nn_model.layer_dims.insert(0, n_features)
        self._nn_model.layer_dims.insert(len(self._nn_model.layer_dims), 1)
        self._nn_model.initialize_parameters_deep()

    def train(self, data, labels, **kwargs):
        """
        Client method:
        Implementation of abstract method of class
        [TrainableModel](../Model/#trainablemodel-class)

        # Arguments
            data: Data, array-like of shape (n_samples, n_features)
            labels: Target classes, array-like of shape (n_samples,)
        """
        self._an, self._cache = self._nn_model.n_model_forward(data)

    def predict(self, data):
        """Client method: compute embeddings"""
        return self._nn_model.n_model_forward(data)[0][0]

    def get_model_params(self):
        """Client method: output parameters are the embeddings."""

        return self._an[0]

    def set_model_params(self, embedding_grads):
        """Client method: update local parameters given the embeddings' gradients"""
        # Forward: precomputed in training
        an = self._an
        cache = self._cache
        # Backward
        grads = self._nn_model.n_model_backward_helper(an, cache, embedding_grads)
        # Update
        self._nn_model.update_parameters(grads, self._learning_rate, self._lam)

    def performance(self, data, labels=None):
        pass

    def evaluate(self, data, labels):
        pass

    def get(self, data):
        """
        Client method: Outputs model's parameters used in the communication.

        # Arguments:
            data: training data
        # Returns:
            embeddings: clients embeddings of the trained model
        """

        return self.predict(data)

    def explain(self):
        return self._nn_model.feature_importance_client()


class VerticalLogLinearServer(TrainableModel):
    """
    Class that represents a server node (active node) for the
    Vertical Federated Learning using Neural Network as local model.
    """

    def __init__(self,
                 params={"lam": 0.01, "learning_rate": 0.001},
                 epsilon=None):
        self._theta0 = 0.01
        self._lam = params["lam"]
        self._learning_rate = params["learning_rate"]
        self._epsilon = epsilon
        self._sensitivity = None

    def train(self,  data, labels,  **kwargs):
        """
        Server method:
        Implementation of abstract method of class [TrainableModel](../Model/#trainablemodel-class)

        # Arguments
            data: clients' data
            labels: Target classes, array-like of shape (n_samples,)
            embeddings: clients' embeddings
        """
        embeddings = kwargs.get("embeddings")
        embeddings_grads = self._compute_gradients(embeddings, labels)
        self.set_model_params(embeddings_grads)

    def predict(self, data):
        """Server method: compute prediction using clients' embeddings"""

        embedding_client = data
        exponent = self._theta0 + sum(embedding_client)
        prediction = np.exp(_logsig(exponent))

        return prediction

    def evaluate(self, data, labels):
        """
        Evaluation of global model.

        # Arguments
            data: predicted labels
            labels: true labels
        """

        auc = roc_auc_score(labels, data)

        return auc

    def get_model_params(self):
        """
        Server method: output parameters are the embeddings' gradients.
        """

        return self._s

    def set_model_params(self, embedding_grads):
        """Server method: update parameters"""

        self._theta0 = self._theta0 - \
                       self._learning_rate * np.mean(embedding_grads)

    def performance(self, data, labels):
        return self.evaluate(data, labels)

    def _compute_gradients(self, embedding_client, y_train):
        """Server method: compute gradients"""

        exponent = self._theta0 + sum(embedding_client)
        y_train = np.asarray(y_train)
        self._s = _expit_b(exponent, y_train)

        return self._s

    def _compute_loss(self, embedding_client, y_true):
        """Server method: compute loss"""

        exponent = self._theta0 + sum(embedding_client)
        y_true = np.asarray(y_true)

        return np.mean((1 - y_true) * exponent - _logsig(exponent))


class NeuralNetHelper:
    """
    layer_dims - - list containing the dimensions of each hidden layer in our network
    """

    def __init__(self, layer_dims):
        self.layer_dims = layer_dims.copy()
        self.parameters = None

    def initialize_parameters_deep(self):
        """
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WN", "bN":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        parameters = {}
        n = len(self.layer_dims)  # number of layers in the network

        for layer in range(1, n):
            parameters['W' + str(layer)] = np.random.normal(size=(self.layer_dims[layer], self.layer_dims[layer - 1]),
                                                            scale=0.1)
            parameters['b' + str(layer)] = np.zeros((self.layer_dims[layer], 1))

            assert (parameters['W' + str(layer)].shape == (self.layer_dims[layer], self.layer_dims[layer - 1]))
            assert (parameters['b' + str(layer)].shape == (self.layer_dims[layer], 1))

        self.parameters = parameters

    def n_model_forward(self, x):
        """
        Implement forward propagation for the [LINEAR->RELU]*(n-1)->LINEAR->IDENTITY computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)

        Returns:
        an -- last post-activation value
        caches -- list of caches containing:
                  every cache of linear_relu_forward() (there are n-1 of them, indexed from 0 to n-2)
                  the cache of linear_forward() (there is one, indexed n-1)
        """
        caches = []
        a = x
        n = len(self.parameters) // 2  # number of layers in the neural network

        for layer in range(1, n):
            a_prev = a
            a, cache = _linear_activation_forward(a_prev,
                                                  self.parameters['W' + str(layer)],
                                                  self.parameters['b' + str(layer)],
                                                  activation='relu')
            caches.append(cache)

        an, cache = _linear_activation_forward(a,
                                               self.parameters['W' + str(n)],
                                               self.parameters['b' + str(n)],
                                               activation='identity')
        caches.append(cache)
        assert (an.shape == (1, x.shape[1]))
        return an, caches

    @staticmethod
    def n_model_backward_helper(an, caches, embedding_grads):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (n-1) -> LINEAR->IDENTITY group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing every cache of linear_activation_forward() with "relu"
                  the cache of linear_forward() (it's caches[n-1])

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        n = len(caches)  # the number of layers

        # Initializing the backpropagation
        dan = np.array(embedding_grads * an.shape[0]).reshape(
            an.shape)  # - (np.divide(y, an) - np.divide(1 - y, 1 - an))

        current_cache = caches[-1]
        grads["dA" + str(n)], grads["dW" + str(n)], grads["db" + str(n)] = _linear_activation_backward(dan,
                                                                                                       current_cache,
                                                                                                       "identity")
        for layer in reversed(range(n - 1)):
            current_cache = caches[layer]
            da_prev_temp, dw_temp, db_temp = _linear_activation_backward(grads["dA" + str(layer + 2)],
                                                                         current_cache, "relu")
            grads["dA" + str(layer + 1)] = da_prev_temp
            grads["dW" + str(layer + 1)] = dw_temp
            grads["db" + str(layer + 1)] = db_temp

        return grads

    def update_parameters(self, grads, learning_rate, lam):
        """
        Update parameters using gradient descent

        Arguments:
        grads -- python dictionary containing your gradients, output of L_model_backward
        """
        n = len(self.parameters) // 2

        for layer in range(n - 1):
            self.parameters["W" + str(layer + 1)] -= learning_rate * (grads["dW" + str(layer + 1)] +
                                                                      lam * self.parameters["W" + str(layer + 1)])
            self.parameters["b" + str(layer + 1)] -= learning_rate * (grads["db" + str(layer + 1)] +
                                                                      lam * self.parameters["b" + str(layer + 1)])
        self.parameters["W" + str(n)] -= learning_rate * (grads["dW" + str(n)] +
                                                          lam * self.parameters["W" + str(n)])

    def feature_importance_client(self):
        number_of_features = len(self.parameters["W1"][0])
        sample = np.random.normal(size=(number_of_features, 1000))
        embedding_shift_mean = {}
        for i in range(number_of_features):
            epsilon = 0.01
            sample_shifted = sample.copy()
            sample_shifted[i] = sample[i] + epsilon
            embedding_shift_mean[i] = np.mean(np.abs(self.n_model_forward(sample)[0][0] -
                                                     self.n_model_forward(sample_shifted)[0][0])) / epsilon
        return embedding_shift_mean
