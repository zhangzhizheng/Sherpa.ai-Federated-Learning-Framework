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
