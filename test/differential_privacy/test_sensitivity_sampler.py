import numpy as np
import pytest

from shfl.differential_privacy.probability_distribution import NormalDistribution
from shfl.differential_privacy import SensitivitySampler
from shfl.differential_privacy import L1SensitivityNorm
from shfl.differential_privacy import L2SensitivityNorm


def mean_query(data):
    """Computes the mean over data."""
    return np.mean(data)


def test_sample_sensitivity_gamma():
    """Checks sensitivity sampler using gamma input."""
    distribution = NormalDistribution(0, 1)

    sampler = SensitivitySampler()
    _, mean = sampler.sample_sensitivity(mean_query, L1SensitivityNorm(),
                                         distribution, n_data_size=100,
                                         gamma=0.33)

    assert np.abs(mean - 0) < 0.5


def test_sample_sensitivity_m_sample_size():
    """Checks sensitivity sampler using sample size input."""
    distribution = NormalDistribution(0, 1)

    sampler = SensitivitySampler()
    _, mean = sampler.sample_sensitivity(mean_query, L1SensitivityNorm(),
                                         distribution, n_data_size=100,
                                         m_sample_size=285)

    assert np.abs(mean - 0) < 0.5


def test_sample_sensitivity_gamma_m_sample_size():
    """Checks sensitivity sampler using both gamma and sample size inputs."""
    distribution = NormalDistribution(0, 1)

    sampler = SensitivitySampler()
    _, mean = sampler.sample_sensitivity(mean_query, L1SensitivityNorm(),
                                         distribution, n_data_size=100,
                                         m_sample_size=285, gamma=0.33)

    assert np.abs(mean - 0) < 0.5


def test_l2_sensitivity_norm():
    """Checks sensitivity sampler using the L2 norm."""
    distribution = NormalDistribution(0, 1)

    sampler = SensitivitySampler()
    _, mean = sampler.sample_sensitivity(mean_query, L2SensitivityNorm(),
                                         distribution, n_data_size=100,
                                         m_sample_size=285, gamma=0.33)

    assert np.abs(mean - 0) < 0.5


@pytest.mark.parametrize("data_structure", [list, tuple])
@pytest.mark.parametrize("sensitivity_norm", [L1SensitivityNorm,
                                              L2SensitivityNorm])
def test_sensitivity_norm_list_of_arrays(sensitivity_norm, data_structure):
    """Checks that the sensitivity norm is computed on each item of a list of arrays."""

    def reshape_to_list(data):
        """Reshapes the input data into a list of arrays."""
        return data_structure(np.reshape(data, (20, 30, 40)))

    distribution = NormalDistribution(0, 1.5)
    sampler = SensitivitySampler()

    max_sensitivity, mean_sensitivity = \
        sampler.sample_sensitivity(query=reshape_to_list,
                                   sensitivity_norm=sensitivity_norm(),
                                   oracle=distribution,
                                   n_data_size=20 * 30 * 40,
                                   m_sample_size=285,
                                   gamma=0.33)

    assert isinstance(max_sensitivity, list)
    assert isinstance(mean_sensitivity, list)
    for sensitivity in max_sensitivity:
        assert sensitivity.sum() < 2 * 1.5
    for sensitivity in mean_sensitivity:
        assert sensitivity.sum() < 2 * 1.5


class SensitivitySamplerTest(SensitivitySampler):
    """Allows to access the private members of the parent class."""
    def concatenate(self, x_1, x_2):
        """Calls the protected member of the parent class."""
        return self._concatenate(x_1, x_2)

    def sort_sensitivity(self, sensitivity_sampled, k_highest):
        """Calls the protected member of the parent class."""
        return self._sort_sensitivity(*sensitivity_sampled, k_highest=k_highest)


def test_multiple_dispatch_concatenate_dictionary_of_arrays():
    """Checks that two dictionaries are concatenated item by item."""
    dict_a = {1: np.random.rand(30, 40),
              2: np.random.rand(20, 50),
              3: np.random.rand(60, 80)}
    dict_b = {3: np.random.rand(1, 40),
              4: np.random.rand(1, 50),
              5: np.random.rand(1, 80)}
    sampler = SensitivitySamplerTest()

    concatenated_dict = sampler.concatenate(dict_a, dict_b)

    for i, j, k in zip(dict_a, dict_b, concatenated_dict):
        assert concatenated_dict[k].shape[0] == \
               dict_a[i].shape[0] + dict_b[j].shape[0]
        assert concatenated_dict[k].shape[1] == \
               dict_a[i].shape[1] == dict_b[j].shape[1]


def test_sort_sensitivities_list_of_scalars():
    """Checks that a list array of sensitivities is correctly sorted in increasing order."""
    sensitivity_sample_size = 5
    sampled_sensitivities = list(np.random.rand(sensitivity_sample_size))
    sampler = SensitivitySamplerTest()

    sensitivity_k_moment, sensitivity_mean = \
        sampler.sort_sensitivity(sampled_sensitivities,
                                 k_highest=sensitivity_sample_size)

    np.testing.assert_array_equal(sensitivity_k_moment,
                                  np.max(np.asarray(sampled_sensitivities)))
    np.testing.assert_array_almost_equal_nulp(
        sensitivity_mean, np.mean(np.asarray(sampled_sensitivities)), nulp=2)


def test_sort_sensitivities_nested_list_of_arrays():
    """Checks that a nested list of arrays of sensitivities is correctly sorted in increasing order.

    In this case, the sorting is done component-wise on the arrays.
    This is useful for instance when a layer-by-layer sensitivity
    estimation in a neural network is desired."""
    sensitivity_sample_size = 5
    num_layers = 10
    sampled_sensitivities = [np.random.rand(num_layers)
                             for _ in range(sensitivity_sample_size)]
    sampler = SensitivitySamplerTest()

    sensitivity_k_moment, sensitivity_mean = \
        sampler.sort_sensitivity(sampled_sensitivities,
                                 k_highest=sensitivity_sample_size)

    for i in range(sensitivity_sample_size):
        layer_sensitivities = np.asarray([sample[i] for sample in sampled_sensitivities])
        np.testing.assert_array_equal(sensitivity_k_moment[i],
                                      np.max(np.asarray(layer_sensitivities)))
        np.testing.assert_array_almost_equal_nulp(
            sensitivity_mean[i], np.mean(np.asarray(layer_sensitivities)), nulp=2)
