import numpy as np
import pytest

import shfl
from shfl.private import DataNode
from shfl.differential_privacy.mechanism import RandomizedResponseBinary
from shfl.differential_privacy.mechanism import RandomizedResponseCoins
from shfl.differential_privacy.mechanism import LaplaceMechanism
from shfl.differential_privacy.mechanism import ExponentialMechanism
from shfl.differential_privacy.mechanism import GaussianMechanism
from shfl.differential_privacy.probability_distribution import NormalDistribution


def test_randomized_response_coins_epsilon_delta():
    """Checks the getter of epsilon and delta parameters for the coins
    randomized mechanism."""
    randomized_response_coins = RandomizedResponseCoins()

    assert randomized_response_coins.epsilon_delta is not None


def test_randomized_response_coins_single_node():
    """Checks that the coins randomized mechanism properly changes the values
        of a binary array contained in a single node."""
    array = np.ones(100)
    node = DataNode()
    node.set_private_data(name="array", data=array)

    node.configure_data_access("array", RandomizedResponseCoins())

    result = node.query("array")
    differences = 0
    for i in range(100):
        if result[i] != array[i]:
            differences = differences + 1

    assert not np.isscalar(result)
    assert 0 < differences < 100
    assert np.mean(result) < 1


def test_randomized_response_coins_federated_array():
    """Checks that the coins randomized mechanism properly changes the values
        of a binary arrays contained in a set of federated nodes."""
    data_size = 1000
    array = np.ones(data_size)
    federated_array = shfl.private.federated_operation.federate_array(array, data_size)

    federated_array.configure_data_access(RandomizedResponseCoins())

    result = federated_array.query()
    differences = 0
    for i in range(data_size):
        if result[i] != array[i]:
            differences = differences + 1

    assert 0 < differences < data_size
    assert np.mean(result) < 1


def test_randomized_response_coins_almost_always_true_values():
    """Checks the influence of the heads probability in the coins randomized mechanism.

    If a low heads probability in the first coin flip is used, more true values
    are expected.
    """
    array = np.ones(1000)
    node = DataNode()
    node.set_private_data(name="array", data=array)

    # Very low heads probability in the first coin flip:
    # the mean should be near the true value
    data_access_definition = RandomizedResponseCoins(prob_head_first=0.01,
                                                     prob_head_second=0.9)
    node.configure_data_access("array", data_access_definition)

    result = node.query("array")

    assert 1 - np.mean(result) < 0.05


def test_randomized_response_coins_almost_always_random_values():
    """Checks the influence of the heads probability in the coins randomized mechanism.

    If a high heads probability in the first coin flip is used, more random values
    are expected.
    """
    array = np.ones(1000)
    node = DataNode()
    node.set_private_data(name="array", data=array)

    # Very high heads probability in the first coin flip:
    # the mean should be near the second coin's head probability
    data_access_definition = RandomizedResponseCoins(prob_head_first=0.99,
                                                     prob_head_second=0.1)
    node.configure_data_access("array", data_access_definition)

    result = node.query("array")

    assert np.abs(0.1 - np.mean(result)) < 0.05


@pytest.mark.parametrize("binary_value", [0, 1])
def test_randomized_response_coins_scalar_value(binary_value):
    """Checks that the coins randomized mechanism properly changes the value
    of a binary scalar contained in a single node."""
    scalar = binary_value
    node = DataNode()
    node.set_private_data(name="scalar", data=scalar)

    node.configure_data_access("scalar", RandomizedResponseCoins())

    result = node.query(private_property="scalar")

    assert np.isscalar(result)
    assert result in (0, 1)


def test_randomized_response_coins_non_binary_values():
    """Checks that the coins randomized mechanism throws an error
    if applied on non-binary data."""
    data_size = 1000
    array = np.random.rand(data_size)
    federated_array = shfl.private.federated_operation.federate_array(array, data_size)

    federated_array.configure_data_access(RandomizedResponseCoins())

    with pytest.raises(ValueError):
        federated_array.query()


def test_randomized_response_binary_epsilon_delta():
    """Checks the getter of epsilon and delta parameters for the binary
    randomized mechanism."""
    randomized_response_binary = RandomizedResponseBinary(f0=0.1, f1=0.9, epsilon=1)

    assert randomized_response_binary.epsilon_delta is not None


def test_randomized_response_binary_input_parameters():
    """Checks that when wrong input parameters are used, an error is thrown."""
    with pytest.raises(ValueError):
        RandomizedResponseBinary(0.1, 2, epsilon=20)

    with pytest.raises(ValueError):
        RandomizedResponseBinary(0.8, 0.8, epsilon=0.1)


def test_randomized_response_binary_deterministic_output():
    """Checks the input parameters of the binary randomized mechanism.

    Setting f0=1, f1=1, the mechanism provides a deterministic
    (i.e. non randomized) response. An error is thrown in this case.
    """
    array = np.array([0, 1])
    node = DataNode()
    node.set_private_data(name="A", data=array)
    with pytest.raises(ValueError):
        RandomizedResponseBinary(f0=1, f1=1, epsilon=1)


def test_randomized_response_binary_array():
    """Checks that the binary randomized mechanism properly changes the values
        of an array contained in a single node."""
    data_size = 100
    array = np.ones(data_size)
    node = DataNode()
    node.set_private_data(name="A", data=array)
    data_access_definition = RandomizedResponseBinary(f0=0.5, f1=0.5, epsilon=1)
    node.configure_data_access("A", data_access_definition)

    result = node.query(private_property="A")

    differences = 0
    for i in range(data_size):
        if result[i] != array[i]:
            differences = differences + 1

    assert 0 < differences < data_size
    assert np.mean(result) < 1


@pytest.mark.parametrize("binary_value", [0, 1])
def test_randomized_response_binary_scalar_value(binary_value):
    """Checks that the binary randomized mechanism properly changes the value
    of a scalar contained in a single node."""
    scalar = binary_value
    node = DataNode()
    node.set_private_data(name="scalar", data=scalar)
    data_access_definition = RandomizedResponseBinary(f0=0.5, f1=0.5, epsilon=1)
    node.configure_data_access("scalar", data_access_definition)

    result = node.query(private_property="scalar")

    assert np.isscalar(result)
    assert result in (0, 1)


def test_randomized_response_binary_almost_always_true_values_ones():
    """Checks the influence of the input parameters in the binary randomized mechanism.

    If a high f1 parameter is set, more true values are expected. If the true
    values are ones, the mean output should be close to 1.
    """
    array = np.ones(1000)
    node = DataNode()
    node.set_private_data(name="array", data=array)

    # If f1 is almost 1 given true values of ones:
    # the mean of the output should be close to 1
    data_access_definition = RandomizedResponseBinary(f0=0.5, f1=0.99, epsilon=5)
    node.configure_data_access("array", data_access_definition)

    result = node.query("array")

    assert 1 - np.mean(result) < 0.05


def test_randomized_response_binary_almost_always_true_values_zeros():
    """Checks the influence of the input parameters in the binary randomized mechanism.

    If a high f0 parameter is set, more true values are expected. If the true
    values are zeros, the mean output should be close to 0.
    """
    array = np.zeros(1000)
    node = DataNode()
    node.set_private_data(name="array", data=array)

    # If f0 is almost 1 given true values of zeros:
    # the mean of the output should be close to 0
    data_access_definition = RandomizedResponseBinary(f0=0.99, f1=0.5, epsilon=5)
    node.configure_data_access("array", data_access_definition)

    result = node.query("array")

    assert np.mean(result) < 0.05


def test_randomized_response_binary_almost_always_false_values_ones():
    """Checks the influence of the input parameters in the binary randomized mechanism.

    If a low f1 parameter is set, more false values are expected. If the true
    values are ones, the mean output should be close to 0.
    """
    array = np.ones(1000)
    node = DataNode()
    node.set_private_data(name="array", data=array)

    # If f1 is almost 0 given true values of ones:
    # the mean of the output should be close to 0
    data_access_definition = RandomizedResponseBinary(f0=0.5, f1=0.01, epsilon=1)
    node.configure_data_access("array", data_access_definition)

    result = node.query("array")

    assert np.mean(result) < 0.05


def test_randomized_response_binary_array_almost_always_false_values_zeros():
    """Checks the influence of the input parameters in the binary randomized mechanism.

    If a low f0 parameter is set, more false values are expected. If the true
    values are zeros, the mean output should be close to 1.
    """
    array = np.zeros(1000)
    node = DataNode()
    node.set_private_data(name="array", data=array)

    # If f0 is almost 0 given true values of zeros:
    # the mean of the output should be close to 1
    data_access_definition = RandomizedResponseBinary(f0=0.01, f1=0.5, epsilon=1)
    node.configure_data_access("array", data_access_definition)

    result = node.query("array")

    assert 1 - np.mean(result) < 0.05


def test_randomized_response_binary_non_binary_array():
    """Checks that the binary randomized mechanism throws an error
    if applied on non-binary array."""
    array = np.random.rand(1000)
    federated_array = shfl.private.federated_operation.federate_array(array, 1000)
    federated_array.configure_data_access(RandomizedResponseBinary(f0=0.5, f1=0.5, epsilon=1))

    with pytest.raises(ValueError):
        federated_array.query()


def test_randomized_response_binary_non_binary_scalar():
    """Checks that the binary randomized mechanism throws an error
    if applied on non-binary scalar."""
    scalar = 0.1
    node = DataNode()
    node.set_private_data(name="scalar", data=scalar)
    data_access_definition = RandomizedResponseBinary(f0=0.5, f1=0.5, epsilon=1)
    node.configure_data_access("scalar", data_access_definition)

    with pytest.raises(ValueError):
        node.query("scalar")


def test_laplace_epsilon_delta():
    """Checks the getter of epsilon and delta parameters for the laplace mechanism."""
    laplace_mechanism = LaplaceMechanism(sensitivity=0.1, epsilon=1)

    assert laplace_mechanism.epsilon_delta is not None


def test_laplace_array():
    """Checks that the laplace mechanism properly changes the values
    of an array contained in a single node."""
    data_size = 1000
    array = NormalDistribution(175, 7).sample(data_size)
    federated_array = shfl.private.federated_operation.federate_array(array, data_size)
    federated_array.configure_data_access(LaplaceMechanism(1, 1))
    result = federated_array.query()

    differences = 0
    for i in range(data_size):
        if result[i] != array[i]:
            differences = differences + 1

    assert differences == data_size
    assert np.mean(array) - np.mean(result) < 5


def test_laplace_scalar():
    """Checks that the laplace mechanism properly changes the values
    of a scalar contained in a single node."""
    scalar = 175
    node = DataNode()
    node.set_private_data("scalar", scalar)
    node.configure_data_access("scalar", LaplaceMechanism(1, 1))

    result = node.query("scalar")

    assert scalar != result
    assert np.abs(scalar - result) < 100


def test_laplace_list_of_arrays():
    """Checks that the laplace mechanism properly changes the values
    of data arrays in a set of federated nodes."""
    n_nodes = 15
    data = [[np.random.rand(3, 2), np.random.rand(2, 3)]
            for _ in range(n_nodes)]

    federated_list = shfl.private.federated_operation.NodesFederation()
    for node in range(n_nodes):
        federated_list.append_data_node(data[node])

    federated_list.configure_data_access(
        LaplaceMechanism(sensitivity=0.01, epsilon=1))
    result = federated_list.query()
    for i_node in range(n_nodes):
        for i_list in range(len(data[i_node])):
            assert (data[i_node][i_list] != result[i_node][i_list]).all()
            assert np.abs(np.mean(data[i_node][i_list]) -
                          np.mean(result[i_node][i_list])) < 1


def test_laplace_dictionary():
    """Checks that the laplace mechanism properly changes the values
    in a dictionary."""
    dictionary = {0: np.array([[2, 4, 5], [2, 3, 5]]),
                  1: np.array([[1, 3, 1], [1, 4, 6]])}

    node = DataNode()
    node.set_private_data("dictionary", dictionary)
    node.configure_data_access("dictionary", LaplaceMechanism(1, 1))

    result = node.query("dictionary")

    assert dictionary.keys() == result.keys()
    assert np.mean(dictionary[0]) - np.mean(result[0]) < 5


def test_laplace_dictionary_sensitivity():
    """Checks that the laplace mechanism properly changes the values
    in a dictionary with component-wise sensitivity."""
    dictionary = {0: np.array([[2, 4, 5], [2, 3, 5]]),
                  1: np.array([[1, 3, 1], [1, 4, 6]])}

    sensitivity = {0: np.array([[1, 1, 2], [2, 1, 1]]),
                   1: np.array([[3, 1, 1], [1, 1, 2]])}

    node = DataNode()
    node.set_private_data("dictionary", dictionary)
    dp_access_mechanism = LaplaceMechanism(sensitivity, 1)
    node.configure_data_access("dictionary", dp_access_mechanism)

    result = node.query("dictionary")

    assert dictionary.keys() == result.keys()
    assert np.mean(dictionary[0]) - np.mean(result[0]) < 5


def test_laplace_dictionary_wrong_sensitivity():
    """Checks that the laplace mechanism throws an error if applied to
    a dictionary with wrong component-wise sensitivity.

    One value of the sensitivity is negative."""
    dictionary = {0: np.array([[2, 4, 5], [2, 3, 5]]),
                  1: np.array([[1, 3, 1], [1, 4, 6]])}

    sensitivity = {0: np.array([[-1, 1, 2], [2, 1, 1]]),
                   1: np.array([[3, 1, 1], [1, 1, 2]])}

    node = DataNode()
    node.set_private_data("dictionary", dictionary)
    dp_access_mechanism = LaplaceMechanism(sensitivity, 1)
    node.configure_data_access("dictionary", dp_access_mechanism)

    with pytest.raises(ValueError):
        node.query("dictionary")


def test_laplace_dictionary_wrong_keys():
    """Checks that the laplace mechanism throws an error if applied to
    a dictionary with wrong component-wise sensitivity.

    One key of the sensitivity dictionary does not match the original data."""
    dictionary = {3: np.array([[2, 4, 5], [2, 3, 5]]),
                  1: np.array([[1, 3, 1], [1, 4, 6]])}

    sensitivity = {0: np.array([[1, 1, 2], [2, 1, 1]]),
                   1: np.array([[3, 1, 1], [1, 1, 2]])}

    node = DataNode()
    node.set_private_data("dictionary", dictionary)
    dp_access_mechanism = LaplaceMechanism(sensitivity, 1)
    node.configure_data_access("dictionary", dp_access_mechanism)

    with pytest.raises(KeyError):
        node.query("dictionary")


def test_laplace_dictionary_wrong_shapes():
    """Checks that the laplace mechanism throws an error if applied to
    a dictionary with wrong component-wise sensitivity.

    The shape of the sensitivity dictionary does not match the original data."""
    dictionary = {0: np.array([2, 3, 5]),
                  1: np.array([[1, 3, 1], [1, 4, 6]])}

    sensitivity = {0: np.array([[1, 1, 2], [2, 1, 1]]),
                   1: np.array([3, 1, 11, 1, 2])}

    node = DataNode()
    node.set_private_data("dictionary", dictionary)
    dp_access_mechanism = LaplaceMechanism(sensitivity, 1)
    node.configure_data_access("dictionary", dp_access_mechanism)

    with pytest.raises(ValueError):
        node.query("dictionary")


def test_gaussian_epsilon_delta():
    """Checks the getter of epsilon and delta parameters for the gaussian mechanism."""
    epsilon_delta = (0.1, 0.005)
    gaussian_mechanism = GaussianMechanism(sensitivity=0.1, epsilon_delta=epsilon_delta)

    assert gaussian_mechanism.epsilon_delta == epsilon_delta


def test_gaussian_array():
    """Checks that the gaussian mechanism properly changes the values
        of a federated array. """
    data_size = 1000
    array = NormalDistribution(175, 7).sample(data_size)
    federated_array = shfl.private.federated_operation.federate_array(array, data_size)

    federated_array.configure_data_access(GaussianMechanism(1, epsilon_delta=(0.1, 1)))
    result = federated_array.query()

    differences = 0
    for i in range(data_size):
        if result[i] != array[i]:
            differences = differences + 1

    assert differences == data_size
    assert np.mean(array) - np.mean(result) < 5


def test_gaussian_scalar():
    """Checks that the gaussian mechanism properly changes the value
    of a scalar contained in a single node."""
    scalar = 175

    node = DataNode()
    node.set_private_data("scalar", scalar)
    node.configure_data_access("scalar",
                               GaussianMechanism(1, epsilon_delta=(0.1, 1)))

    result = node.query("scalar")

    assert scalar != result
    assert np.abs(scalar - result) < 100


def test_exponential_epsilon_delta():
    """Checks the getter of epsilon and delta parameters for the exponential mechanism."""
    def utility():
        pass

    response_range = np.arange(0, 2, 0.001)
    delta_u = response_range.max(initial=None)
    epsilon = 1
    exponential_mechanism = ExponentialMechanism(utility, response_range, delta_u, epsilon)

    assert exponential_mechanism.epsilon_delta == (epsilon, 0.)


def test_exponential_pricing():
    """Checks the exponential mechanism for the pricing example.

    The mechanism must properly output correct best and worst revenue prices,
    without revealing the true bids.

    # References:
        [Section 3.4 of The algorithmic foundations
        of differential privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
    """
    def utility(bids, prices):
        revenue = np.zeros(len(prices))
        for i, price in enumerate(prices):
            revenue[i] = price * sum(np.greater_equal(bids, price))
        return revenue

    true_bids = [1.00, 1.00, 1.00, 3.01]      # Input data set: the true bids
    price_range = np.arange(0, 3.5, 0.001)    # Set the interval of possible outputs r
    delta_u = price_range.max(initial=None)   # In this specific case, Delta u = max(r)

    node = DataNode()
    node.set_private_data(name="bids", data=np.array(true_bids))
    data_access_definition = ExponentialMechanism(utility, price_range,
                                                  delta_u, epsilon=5, size=10000)
    node.configure_data_access("bids", data_access_definition)
    result = node.query("bids")

    # Define bins edges to include true bids values:
    bins_edges = np.linspace(start=[0, 1, 3.01], stop=[1, 3.01, 3.5],
                             num=int(np.sqrt(len(result))),
                             axis=1, endpoint=False).flatten()

    y_bin, x_bin = np.histogram(a=result,
                                bins=bins_edges,
                                range=(price_range.min(initial=None),
                                       price_range.max(initial=None)),
                                density=True)

    # Min and max revenues: we take the Left edge of the bin
    max_revenue_price = x_bin[np.where(y_bin == y_bin.max(initial=None))]
    min_revenue_price = x_bin[np.where(y_bin == y_bin.min(initial=None))]

    # Check the best revenue prices are close 1.00, but not greater:
    for max_price in max_revenue_price:
        assert 0.5 < max_price < 1.0
    # Check the lowest revenue prices are either close to zero or greater than 3.01:
    for min_price in min_revenue_price:
        assert min_price < 0.5 or min_price >= 3.01


def test_exponential_obtain_laplace():
    """Checks that the laplace mechanism is a special case of the exponential.

    # References:
        [Notebook on the exponential mechanism](https://github.com/sherpaai/
        Sherpa.ai-Federated-Learning-Framework/blob/master/notebooks/
        differential_privacy/differential_privacy_exponential.ipynb)
    """

    def utility_laplace(dataset, domain_range):
        utility = -np.absolute(dataset - domain_range)
        return utility

    domain = np.arange(-20, 20, 0.001)   # Set the interval of possible outputs domain_range
    true_dataset = 3.5                         # Set a value for the data set
    delta_u = 1                     # We simply set it to one
    epsilon = 1                     # Set a value for epsilon
    size = 100000                   # We want to repeat the query this many times

    node = DataNode()
    node.set_private_data(name="identity", data=np.array(true_dataset))

    data_access_definition = ExponentialMechanism(utility_laplace, domain,
                                                  delta_u, epsilon, size)
    node.configure_data_access("identity", data_access_definition)
    result = node.query("identity")

    # Check all outputs are within range:
    assert (result >= domain.min(initial=None)).all() and \
           (result <= domain.max(initial=None)).all()
    # Check the mean output is close to true value:
    assert np.absolute(np.mean(result) - true_dataset) < (delta_u/epsilon)


def test_gaussian_input_parameters():
    """Checks that when wrong input parameters are used, an error is thrown."""
    with pytest.raises(ValueError):
        GaussianMechanism(1, epsilon_delta=(1, 1, 1))

    with pytest.raises(ValueError):
        GaussianMechanism(1, epsilon_delta=(-0.5, 1))

    with pytest.raises(ValueError):
        GaussianMechanism(1, epsilon_delta=(0.5, -1))

    with pytest.raises(ValueError):
        GaussianMechanism(1, epsilon_delta=(1, 1))


def test_gaussian_sensitivity_wrong_input():
    """Checks that when the sensitivity is not broadcastable to data,
    an error is thrown by the gaussian mechanism."""

    epsilon_delta = (0.1, 1)

    # Scalar query result, Too many sensitivity values provided:
    scalar = 175
    sensitivity = [0.1, 0.5]
    node = DataNode()
    node.set_private_data("scalar", scalar)
    node.configure_data_access("scalar",
                               GaussianMechanism(sensitivity=sensitivity,
                                                 epsilon_delta=epsilon_delta))
    with pytest.raises(ValueError):
        node.query("scalar")

    # Both query result and sensitivity are 1D-arrays, but non-broadcastable:
    data_array = [10, 10, 10, 10]
    sensitivity = [0.1, 10, 100, 1000, 1000]
    node = DataNode()
    node.set_private_data("data_array", data_array)
    node.configure_data_access("data_array",
                               GaussianMechanism(sensitivity=sensitivity,
                                                 epsilon_delta=epsilon_delta))
    with pytest.raises(ValueError):
        node.query("data_array")

    # ND-array query result and 1D-array sensitivity, but non-broadcastable:
    data_array = [[10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10]]
    sensitivity = [0.1, 10, 100]
    node = DataNode()
    node.set_private_data("data_ndarray", data_array)
    node.configure_data_access("data_ndarray",
                               GaussianMechanism(sensitivity=sensitivity,
                                                 epsilon_delta=epsilon_delta))
    with pytest.raises(ValueError):
        node.query("data_ndarray")

    # Both query result and sensitivity are ND-arrays, but non-broadcastable:
    data_array = [[10, 10, 10, 10],
                  [10, 10, 10, 10],
                  [10, 10, 10, 10]]
    sensitivity = [[0.1, 10, 100, 1000, 10000],
                   [0.1, 10, 100, 1000, 10000],
                   [0.1, 10, 100, 1000, 10000]]
    node = DataNode()
    node.set_private_data("data_ndarray", data_array)
    node.configure_data_access("data_ndarray",
                               GaussianMechanism(sensitivity=sensitivity,
                                                 epsilon_delta=epsilon_delta))
    with pytest.raises(ValueError):
        node.query("data_ndarray")


def test_laplace_sensitivity_wrong_input():
    """Checks that when the sensitivity is not broadcastable to data,
    an error is thrown by the laplace mechanism."""

    epsilon = 1

    # Query result is a list of arrays. Sensitivity must be either a scalar,
    # or a list of the same length as query:
    data_list = [np.random.rand(30, 20),
                 np.random.rand(20, 30),
                 np.random.rand(50, 40)]
    sensitivity = np.array([1, 1])  # Array instead of scalar
    node = DataNode()
    node.set_private_data("data_list", data_list)
    node.configure_data_access("data_list",
                               LaplaceMechanism(sensitivity=sensitivity,
                                                epsilon=epsilon))
    with pytest.raises(ValueError):
        node.query("data_list")

    data_array = [[10, 10, 10, 10],
                  [10, 10, 10, 10],
                  [10, 10, 10, 10]]
    sensitivity = [1, 1]  # List of wrong length
    node = DataNode()
    node.set_private_data("data_list", data_array)
    node.configure_data_access("data_list",
                               LaplaceMechanism(sensitivity=sensitivity,
                                                epsilon=epsilon))
    with pytest.raises(IndexError):
        node.query("data_list")

    # Query result is wrong data structure: so far, tuples are not allowed
    data_tuple = (1, 2, 3, 4, 5)
    sensitivity = 2
    node = DataNode()
    node.set_private_data("data_tuple", data_tuple)
    node.configure_data_access("data_tuple",
                               LaplaceMechanism(sensitivity=sensitivity,
                                                epsilon=epsilon))
    with pytest.raises(NotImplementedError):
        node.query("data_tuple")
