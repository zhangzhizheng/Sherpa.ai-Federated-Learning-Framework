import numpy as np
import abc


class ProbabilityDistribution(abc.ABC):
    """Implements a probability distribution for sensitivity sampling.

    This interface must be implemented to define
    the probability distribution to sample the data from
    when estimating the sensitivity of a query
    (see [SensitivitySampler](../sensitivity_sampler/)).
    """

    @abc.abstractmethod
    def sample(self, size):
        """Samples from the specified probability distribution.

        Abstract method.

        # Arguments:
            size: Sample size.

        # Returns:
            sample: Array-like object of length equal to `size`.
        """


class NormalDistribution(ProbabilityDistribution):
    """Samples from a Normal distribution.

    It implements the class
    [ProbabilityDistribution](./#probabilitydistribution-class).

    # Arguments:
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
    """
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

    def sample(self, size):
        """See base class.
        """
        return np.random.normal(self._mean, self._std, size)


class GaussianMixture(ProbabilityDistribution):
    """Samples from a mixture of normal distributions.

    It implements the class
    [ProbabilityDistribution](./#probabilitydistribution-class).

    # Arguments:
        params: Array of arrays with mean and std
            for every gaussian distribution.
        weights: Array of weights for every distribution with sum 1.

    # Example:

    ```python
        # Parameters for two Gaussian
        mu_M = 178
        mu_F = 162
        sigma_M = 7
        sigma_F = 7

        # Parameters
        norm_params = np.array([[mu_M, sigma_M],
                               [mu_F, sigma_F]])
        weights = np.ones(2) / 2.0

        # Creating combination of gaussian
        distribution = GaussianMixture(norm_params, weights)
    ```
    """
    def __init__(self, params, weights):
        self._gaussian_distributions = []
        for param in params:
            self._gaussian_distributions.append(
                NormalDistribution(param[0], param[1]))
        self._weights = weights

    def sample(self, size):
        """See base class.
        """

        mixture_idx = np.random.choice(len(self._weights),
                                       size=size,
                                       replace=True,
                                       p=self._weights)

        values = []
        for i in mixture_idx:
            gaussian_distributions = self._gaussian_distributions[i]
            values.append(gaussian_distributions.sample(1))

        return np.fromiter(values, dtype=np.float64)
