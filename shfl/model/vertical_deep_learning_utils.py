import numpy as np


def _logsig(x):
    """Compute the log-sigmoid function component-wise."""
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out


def _expit_b(x, b):
    """Compute sigmoid(x) - b component-wise."""
    idx = x < 0
    out = np.zeros_like(x)
    exp_x = np.exp(x[idx])
    b_idx = b[idx]
    out[idx] = ((1 - b_idx) * exp_x - b_idx) / (1 + exp_x)
    exp_nx = np.exp(-x[~idx])
    b_nidx = b[~idx]
    out[~idx] = ((1 - b_nidx) - b_nidx * exp_nx) / (1 + exp_nx)
    return out


def _sigmoid(z):
    """
    Implements the sigmoid activation

    Arguments:
    Z -- numpy array of any shape

    Returns:
    a -- output of sigmoid(z), same shape as z
    cache -- returns z as well, useful during backpropagation
    """
    a = 1 / (1 + np.exp(-z))
    cache = z
    return a, cache


def _relu(z):
    """
    Implement the RELU function.

    Arguments:
    z -- Output of the linear layer, of any shape

    Returns:
    a -- Post-activation parameter, of the same shape as z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    a = np.maximum(0, z)
    cache = z
    return a, cache


def _identity(z):
    """
    Implement the identity function.

    Arguments:
    z -- Output of the linear layer, of any shape

    Returns:
    a -- Post-activation parameter, of the same shape as z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    a = z
    cache = z
    return a, cache


def _relu_backward(da, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    da -- post-activation gradient, of any shape
    cache -- 'z' where we store for computing backward propagation efficiently

    Returns:
    dz -- Gradient of the cost with respect to z
    """
    z = cache
    dz = np.array(da, copy=True)  # just converting dz to a correct object.
    dz[z <= 0] = 0
    return dz


def _identity_backward(da):
    """
    Implement the backward propagation for an single identity unit.

    Arguments:
    da -- post-activation gradient, of any shape
    cache -- 'z' where we store for computing backward propagation efficiently

    Returns:
    dz -- Gradient of the cost with respect to z
    """
    dz = np.array(da, copy=True)  # just converting dz to a correct object.
    return dz


def _sigmoid_backward(da, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    da -- post-activation gradient, of any shape
    cache -- 'z' where we store for computing backward propagation efficiently

    Returns:
    dz -- Gradient of the cost with respect to z
    """
    z = cache
    s = 1 / (1 + np.exp(-z))
    dz = da * s * (1 - s)
    return dz


def _linear_forward(a, w, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    a -- activations from previous layer (or input data): (size of previous layer, number of examples)
    w -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    z = np.dot(w, a) + b
    assert (z.shape == (w.shape[0], a.shape[1]))
    cache = (a, w, b)
    return z, cache


def _linear_activation_forward(a_prev, w, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    a_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    w -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    a -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    z, linear_cache = _linear_forward(a_prev, w, b)
    a, activation_cache = None, None

    if activation == "sigmoid":
        a, activation_cache = _sigmoid(z)
    elif activation == "relu":
        a, activation_cache = _relu(z)
    elif activation == "identity":
        a, activation_cache = _identity(z)

    cache = (linear_cache, activation_cache)

    return a, cache


def _compute_cost(an, y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    an -- probability vector corresponding to your label predictions, shape (1, number of examples)
    y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = y.shape[1]
    # Compute loss from aL and y.
    cost = (-1 / m) * np.sum(np.multiply(y, np.log(an)) + np.multiply(1 - y, np.log(1 - an)))
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost


def _linear_backward(dz, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dz -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (a_prev, w, b) coming from the forward propagation in the current layer

    Returns:
    da_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as a_prev
    dw -- Gradient of the cost with respect to W (current layer l), same shape as w
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    a_prev, w, b = cache
    m = a_prev.shape[1]

    dw = (1. / m) * np.dot(dz, a_prev.T)
    db = (1. / m) * np.sum(dz, axis=1, keepdims=True)
    da_prev = np.dot(w.T, dz)

    assert (da_prev.shape == a_prev.shape)
    assert (dw.shape == w.shape)
    assert (db.shape == b.shape)

    return da_prev, dw, db


def _linear_activation_backward(da, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    dz = None
    if activation == "relu":
        dz = _relu_backward(da, activation_cache)
    elif activation == "sigmoid":
        dz = _sigmoid_backward(da, activation_cache)
    elif activation == "identity":
        dz = _identity_backward(da)

    da_prev, dw, db = _linear_backward(dz, linear_cache)

    return da_prev, dw, db
