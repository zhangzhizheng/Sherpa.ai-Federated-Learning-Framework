from math import sqrt
from math import log
import copy
import numpy as np
import scipy
from multipledispatch import dispatch


from shfl.private.data import DPDataAccessDefinition
from shfl.private.query import IdentityFunction


class RandomizedResponseCoins(DPDataAccessDefinition):
    """Implements the randomized response mechanism for differential privacy.

    It implements the class
    [DPDataAccessDefinition](../../private/data/#dpdataaccessdefinition-class).

    The algorithm uses two coin flips and is described as follows:

    1) Flip a coin

    2) If tails, then respond truthfully.

    3) If heads, then flip a second coin and respond "Yes"
        if heads and "No" if tails.

    The input data must be binary, otherwise an exception will be raised.

    # Arguments
        prob_head_first: Optional; Float in [0,1], the probability of the first
            coin flip in the algorithm. It represents the probability
            of having a random response instead of the true value
            (default is 0.5).
        prob_head_second: Optional; Float in [0,1], the probability of the second
            coin flip in the algorithm. It represents the probability
            of responding "true" when the first coin turned up tails.

    # Properties:
        epsilon_delta: Returns the epsilon and delta values of the
            differentially-private mechanism.

    # Example:
        See the [notebook on the randomized response using two coins](https://github.com/
        sherpaai/Sherpa.ai-Federated-Learning-Framework/blob/master/
        notebooks/differential_privacy/differential_privacy_basic_concepts.ipynb).

    # References
        [The algorithmic foundations of differential privacy](
           https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
    """

    def __init__(self, prob_head_first=0.5, prob_head_second=0.5):
        self._prob_head_first = prob_head_first
        self._prob_head_second = prob_head_second
        self._epsilon_delta = (log(3), 0)

    @property
    def epsilon_delta(self):
        return self._epsilon_delta

    def apply(self, data, **kwargs):
        """Randomizes the input binary data.

        # Arguments:
            data: Array-type object containing the binary data to be randomized.

        # Returns:
            result: Array-type object of same shape as the input
                containing the differentially-private randomized data.
        """
        data = np.asarray(data)
        self._check_binary_data(data)

        first_coin_flip = scipy.stats.bernoulli.rvs(
            p=(1 - self._prob_head_first), size=data.shape)
        second_coin_flip = scipy.stats.bernoulli.rvs(
            p=self._prob_head_second, size=data.shape)

        result = data * first_coin_flip + \
            (1 - first_coin_flip) * second_coin_flip

        return result


class RandomizedResponseBinary(DPDataAccessDefinition):
    """Implements the general randomized response mechanism
        for differential privacy.

    It implements the class
    [DPDataAccessDefinition](../../private/data/#dpdataaccessdefinition-class).

    This is the most general binary randomized response algorithm.
    The algorithm is defined through the conditional probabilities:

    - p00 = P( output=0 | input=0 ) = f0
    - p10 = P( output=1 | input=0 ) = 1 - f0
    - p11 = P( output=1 | input=1 ) = f1
    - p01 = P( output=0 | input=1 ) = 1 - f1

    For f0=f1=0 or 1, the algorithm is not random.
    It is maximally random for f0=f1=1/2.
    This algorithm is epsilon-differentially private
    if epsilon >= log max{ p00/p01, p11/p10} = log \
    max { f0/(1-f1), f1/(1-f0)}.
    The class [RandomizedResponseCoins](./#randomizedresponsecoins) is a
    special case of this class for specific values of f0 and f1.

    Input data must be binary, otherwise an exception will be raised.

    # Arguments
        f0: Float in [0,1] representing the probability of getting 0
            when the input is 0.
        f1: Float in [0,1] representing the probability of getting 1
            when the input is 1.

    # Properties:
        epsilon_delta: Returns the epsilon and delta values of the
            differentially-private mechanism.

    # Example:
        See the [notebook on the binary randomized response](https://github.com/
        sherpaai/Sherpa.ai-Federated-Learning-Framework/blob/master/
        notebooks/differential_privacy/differential_privacy_binary_average_attack.ipynb).

    # References:
        [Using randomized response for differential privacy
            preserving data collection](http://csce.uark.edu/~xintaowu/publ/DPL-2014-003.pdf)
    """

    def __init__(self, f0, f1, epsilon):
        self._check_epsilon_delta((epsilon, 0))
        if f0 <= 0 or f0 >= 1:
            raise ValueError(
                "f0 argument must be between 0 an 1, "
                "{} was provided".format(f0) + ".")
        if f1 <= 0 or f1 >= 1:
            raise ValueError(
                "f1 argument must be between 0 an 1, "
                "{} was provided".format(f1) + ".")
        if epsilon < log(max(f0 / (1 - f1), f1 / (1 - f0))):
            raise ValueError(
                "To ensure epsilon differential privacy, "
                "the following inequality mus be satisfied " +
                "{}=epsilon >= {}=log max ( f0 / (1 - f1), f1 / (1 - f0))"
                .format(epsilon, log(max(f0 / (1 - f1), f1 / (1 - f0)))) + ".")
        self._f0 = f0
        self._f1 = f1
        self._epsilon = epsilon

    @property
    def epsilon_delta(self):
        return self._epsilon, 0

    def apply(self, data, **kwargs):
        """Randomizes the input binary data.

        # Arguments:
            data: Array-type object containing the binary data to be randomized.

        # Returns:
            result: Array-type object of same shape as the input
                containing the differentially-private randomized data.
        """
        data = np.asarray(data)
        self._check_binary_data(data)

        probabilities = np.empty(data.shape)
        x_zero = data == 0
        probabilities[x_zero] = 1 - self._f0
        probabilities[~x_zero] = self._f1
        x_response = scipy.stats.bernoulli.rvs(p=probabilities)

        return x_response


class LaplaceMechanism(DPDataAccessDefinition):
    """Implements the Laplace mechanism for differential privacy.

    It implements the class
    [DPDataAccessDefinition](../../private/data/#dpdataaccessdefinition-class).

    Note that the Laplace mechanism is a randomization algorithm
    that depends on the l1 sensitivity, which can be regarded as a
    numeric query. One can show that this mechanism is
    epsilon-differentially private with epsilon = l1-sensitivity/b
    where b is a constant and l1-sensitivity is the query's l1 sensitivity
    (intuitively, the maximum output difference).

    A different sample of the Laplace distribution is taken
    for each element of the query output. For example,
    if the query output is a list containing three
    arrays of size n_rounds * m, then 3*n_rounds*m samples are taken from
    the Laplace distribution using the provided sensitivity.

    # Arguments:
        sensitivity: Scalar, array, list or dictionary representing the
            sensitivity of the query. It must be consistent with
            the query output (see examples below).
        epsilon: Scalar representing the desired epsilon.
        query: Optional; Function to apply over the private data
            (see class [Query](../../private/query/#query-class)).
            The default is None, in which case the identity function
            is used.

    # Properties:
        epsilon_delta: Returns the epsilon and delta values of the
            differentially-private mechanism.

    # Example 1:
        If the query output is an array of size n_rounds * m, and a scalar
        sensitivity is provided, then the same sensitivity is applied to
        each entry in the output.
        Instead, providing a vector of sensitivities of size m, then the
        sensitivity is applied column-wise over the query output.
        Finally, providing a sensitivity array of size n_rounds * m,
        a different sensitivity value si applied to each element of the
        query output. Note that in all the cases, n_rounds*m Laplace samples are
        taken, i.e. each value of the query output is perturbed with a
        different noise value.

    # Example 2:
        If the query output is a list or a dictionary
        containing arrays, sensitivity should be provided as a list
        or dictionary with the same length.
        Then for each array in the list or dictionary, the same
        considerations as for Example 1 hold. For instance, providing simply
        a scalar will apply the same sensitivity to each array in the list
        or dictionary.

    # Example 3:
        See the [notebook on the Laplace mechanism](https://github.com/
        sherpaai/Sherpa.ai-Federated-Learning-Framework/blob/master/
        notebooks/differential_privacy/differential_privacy_laplace.ipynb).

    # References:
        [The algorithmic foundations of differential privacy](
           https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
    """

    def __init__(self, sensitivity, epsilon, query=None):
        if query is None:
            query = IdentityFunction()

        self._check_epsilon_delta((epsilon, 0))
        self._sensitivity = sensitivity
        self._epsilon = epsilon
        self._query = query

    @property
    def epsilon_delta(self):
        return self._epsilon, 0

    @staticmethod
    def _seq_iter(obj):
        return obj if isinstance(obj, dict) else range(len(obj))

    @staticmethod
    def _pick_sensitivity(sensitivity, i):
        try:
            return sensitivity[i] if isinstance(sensitivity, (dict, list)) \
                else sensitivity
        except KeyError as wrong_shape:
            raise KeyError("The sensitivity does not contain "
                           "the key {}".format(i) + ".") from wrong_shape

    def apply(self, data, **kwargs):
        """Applies Laplace noise to the input data.

        # Arguments:
            data: Array-type object containing the binary data to be randomized.

        # Returns:
            result: Array-type object of same shape as the input
                containing the differentially-private randomized data.
        """
        query_result = self._query.get(data)

        return self._add_noise(query_result, self._sensitivity)

    @dispatch((np.ndarray, np.ScalarType), (np.ndarray, np.ScalarType))
    def _add_noise(self, obj, sensitivity):
        """Adds Laplace noise to arrays."""
        sensitivity_array = np.asarray(sensitivity)
        obj = np.asarray(obj)
        self._check_sensitivity_positive(sensitivity_array)
        self._check_sensitivity_shape(sensitivity_array, obj)
        scale = sensitivity / self._epsilon
        output = obj + np.random.laplace(loc=0.0, scale=scale, size=obj.shape)
        return output

    @dispatch((dict, list), (dict, list, np.ndarray, np.ScalarType))
    def _add_noise(self, obj, sensitivity):
        """Adds Laplace noise to a (nested) list or a dictionary of arrays."""
        output = copy.deepcopy(obj)
        for i in self._seq_iter(obj):
            sensitivity_tmp = self._pick_sensitivity(sensitivity, i)
            output[i] = self._add_noise(obj[i], sensitivity_tmp)
        return output


class GaussianMechanism(DPDataAccessDefinition):
    """Implements the Gaussian mechanism for differential privacy.

    It implements the class
    [DPDataAccessDefinition](../../private/data/#dpdataaccessdefinition-class).

    Note that the Gaussian mechanism is a randomization algorithm
    that depends on the l2-sensitivity, which can be regarded
    as a numeric query. One can show that this mechanism is
    (epsilon, delta)-differentially private where the noise is drawn
    from a Gaussian distribution with zero mean and standard deviation
    equal to sqrt(2 * ln(1,25/delta)) * l2-sensitivity / epsilon
    where epsilon is in the interval (0, 1).

    # Arguments:
        sensitivity: Scalar, array, list or dictionary representing the
            sensitivity of the query. It must be consistent with
            the query output (see examples below).
        epsilon: Float representing the desired epsilon.
        delta: Float representing the desired delta.
        query: Optional; Function to apply over the private data
            (see class [Query](../../private/query/#query-class)).
            The default is None, in which case the identity function
            is used.

    # Properties:
        epsilon_delta: Returns the epsilon and delta values of the
            differentially-private mechanism.

    # References:
        [The algorithmic foundations of differential privacy](
           https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
    """

    def __init__(self, sensitivity, epsilon_delta, query=None):
        if query is None:
            query = IdentityFunction()

        self._check_epsilon_delta(epsilon_delta)
        if epsilon_delta[0] >= 1:
            raise ValueError(
                "In the Gaussian mechanism epsilon have to be "
                "greater than 0 and less than 1.")
        self._check_sensitivity_positive(sensitivity)
        self._sensitivity = sensitivity
        self._epsilon_delta = epsilon_delta
        self._query = query

    @property
    def epsilon_delta(self):
        return self._epsilon_delta

    def apply(self, data, **kwargs):
        """Adds Gaussian noise to the input data.

        # Arguments:
            data: Array-type object containing the binary data to be randomized.

        # Returns:
            result: Array-type object of same shape as the input
                containing the differentially-private randomized data.
        """
        query_result = np.asarray(self._query.get(data))
        sensitivity = np.asarray(self._sensitivity)
        self._check_sensitivity_shape(sensitivity, query_result)
        std = sqrt(2 * np.log(1.25 / self._epsilon_delta[1])) * \
            sensitivity / self._epsilon_delta[0]

        return query_result + np.random.normal(loc=0.0,
                                               scale=std,
                                               size=query_result.shape)


class ExponentialMechanism(DPDataAccessDefinition):
    """Implements the exponential mechanism for differential privacy.

    It implements the class
    [DPDataAccessDefinition](../../private/data/#dpdataaccessdefinition-class).

    This is the most general mechanism for differential privacy.
    It can be shown that the other mechanisms
    (Laplace, Gaussian, Randomized response) are a special case of this one
    (see the [notebook on the Exponential mechanism](https://github.com/
    sherpaai/Sherpa.ai-Federated-Learning-Framework/blob/master/
    notebooks/differential_privacy/differential_privacy_exponential.ipynb)).

    # Arguments:
        utility_function: Utility function with arguments x and response_range.
            It should be vectorized, so that for a
            particular database x, it returns
            as many values as given in response_range.
        response_range: Array representing the response space.
        delta_u: Float representing the sensitivity of the utility function.
        epsilon: Float representing the desired epsilon.
        size: Optional; The number of queries to perform at once (default is 1).

    # Properties:
        epsilon_delta: Returns the epsilon and delta values of the
            differentially-private mechanism.

    # Example:
        See the [notebook on the Exponential mechanism](https://github.com/
        sherpaai/Sherpa.ai-Federated-Learning-Framework/blob/master/
        notebooks/differential_privacy/differential_privacy_exponential.ipynb).

    # References:
        [The algorithmic foundations of differential privacy](
           https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
    """

    def __init__(self, utility_function, response_range,
                 delta_u, epsilon, size=1):
        self._check_epsilon_delta((epsilon, 0))
        self._utility_function = utility_function
        self._response_range = response_range
        self._delta_u = delta_u
        self._epsilon = epsilon
        self._size = size

    @property
    def epsilon_delta(self):
        return self._epsilon, 0

    def apply(self, data, **kwargs):
        """Adds Exponential noise to the input data.

        # Arguments:
            data: Array-type object containing the binary data to be randomized.

        # Returns:
            result: Array-type object of same shape as the input
                containing the differentially-private randomized data.
        """
        u_points = self._utility_function(data, self._response_range)
        probability = np.exp(self._epsilon * u_points / (2 * self._delta_u))
        probability /= probability.sum()
        sample = np.random.choice(
            a=self._response_range, size=self._size, replace=True, p=probability)

        return sample
