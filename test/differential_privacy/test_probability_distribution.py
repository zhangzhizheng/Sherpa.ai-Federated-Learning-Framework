import numpy as np

from shfl.differential_privacy.probability_distribution import NormalDistribution
from shfl.differential_privacy.probability_distribution import GaussianMixture


def test_normal_distribution():
    """Checks that the sampling from the normal distribution is correct."""
    data_size = 1000
    array = NormalDistribution(175, 7).sample(data_size)

    assert len(array) == 1000
    assert np.abs(np.mean(array) - 175) < 5


def test_gaussian_mixture():
    """Checks that the sampling from the mixture of two gaussian distributions
    is correct."""
    sample_size = 1000

    mu_first = 178
    mu_second = 162
    sigma_first = 7
    sigma_second = 7
    params = np.array([[mu_first, sigma_first],
                      [mu_second, sigma_second]])

    weights = np.ones(2) / 2.0
    array = GaussianMixture(params, weights).sample(sample_size)

    assert len(array) == sample_size
    assert np.abs(np.mean(array) - 170) < 5
