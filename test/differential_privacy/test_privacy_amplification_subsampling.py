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

from math import log
from math import exp
import numpy as np
import pytest

from shfl.private import DataNode
from shfl.differential_privacy.mechanism import LaplaceMechanism
from shfl.differential_privacy.mechanism import GaussianMechanism
from shfl.differential_privacy.mechanism import RandomizedResponseBinary
from shfl.differential_privacy.mechanism import RandomizedResponseCoins
from shfl.differential_privacy.mechanism import ExponentialMechanism
from shfl.differential_privacy.privacy_amplification_subsampling import SampleWithoutReplacement


def test_sample_without_replacement_multidimensional_array():
    """Checks that the rows of a multidimensional array are correctly sampled."""
    array = np.ones((100, 2))
    sample_size = 50
    node = DataNode()
    node.set_private_data(name="array", data=array)

    access_definition = LaplaceMechanism(sensitivity=1, epsilon=1)
    sampling_method = SampleWithoutReplacement(
        access_definition, sample_size, array.shape)
    node.configure_data_access("array", sampling_method)
    result = node.query("array")

    assert result.shape[0] == sample_size


def test_sample_without_replacement():
    """Checks that the elements of a 1D array are correctly sampled using
    various differential privacy mechanisms."""
    array = np.ones(100)
    node_single = DataNode()
    node_single.set_private_data(name="array", data=array)
    sample_size = 50

    def utility(data_base, response_range):
        utility_scores = np.zeros(len(response_range))
        for i, response in enumerate(response_range):
            utility_scores[i] = response * sum(np.greater_equal(data_base, response))
        return utility_scores

    response = np.arange(0, 3.5, 0.001)
    delta_u = response.max(initial=None)
    epsilon = 5
    exponential_mechanism = ExponentialMechanism(
        utility, response, delta_u, epsilon, size=sample_size)

    access_definitions = [LaplaceMechanism(1, 1), GaussianMechanism(1, (0.5, 0.5)),
                          RandomizedResponseBinary(0.5, 0.5, 1), RandomizedResponseCoins(),
                          exponential_mechanism]

    for access in access_definitions:
        sampling_method = SampleWithoutReplacement(access, sample_size, array.shape)
        node_single.configure_data_access("array", sampling_method)
        result = node_single.query("array")
        assert result.shape[0] == sample_size


def test_sample_size_error():
    """Checks that the sample size is not larger than the private data."""
    array = np.ones(100)
    node_single = DataNode()
    node_single.set_private_data(name="array", data=array)
    sample_size = 101

    with pytest.raises(ValueError):
        SampleWithoutReplacement(LaplaceMechanism(1, 1), sample_size, array.shape)


@pytest.mark.parametrize("data_shape, sample_size", [((100, ), 50),
                                                     ((20, 5), 10)])
def test_epsilon_delta_reduction(data_shape, sample_size):
    """Checks that epsilon and delta are reduced through sampling as expected."""
    epsilon = 1
    access_definition = LaplaceMechanism(1, epsilon)
    sampling_method = SampleWithoutReplacement(access_definition, sample_size, data_shape)
    proportion = sample_size / data_shape[0]
    assert sampling_method.epsilon_delta == (
        log(1 + proportion * (exp(epsilon) - 1)), 0)
