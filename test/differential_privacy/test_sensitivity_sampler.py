import numpy as np
import pytest

from shfl.private.query import Mean
# from shfl.private.query import Query
from shfl.differential_privacy.probability_distribution import NormalDistribution
from shfl.differential_privacy import SensitivitySampler
from shfl.differential_privacy import L1SensitivityNorm
from shfl.differential_privacy import L2SensitivityNorm


def test_sample_sensitivity_gamma():
    """Checks sensitivity sampler using gamma input."""
    distribution = NormalDistribution(0, 1)

    sampler = SensitivitySampler()
    _, mean = sampler.sample_sensitivity(Mean(), L1SensitivityNorm(),
                                         distribution, n_data_size=100,
                                         gamma=0.33)

    assert np.abs(mean - 0) < 0.5


def test_sample_sensitivity_m_sample_size():
    """Checks sensitivity sampler using sample size input."""
    distribution = NormalDistribution(0, 1)

    sampler = SensitivitySampler()
    _, mean = sampler.sample_sensitivity(Mean(), L1SensitivityNorm(),
                                         distribution, n_data_size=100,
                                         m_sample_size=285)

    assert np.abs(mean - 0) < 0.5


def test_sample_sensitivity_gamma_m_sample_size():
    """Checks sensitivity sampler using both gamma and sample size inputs."""
    distribution = NormalDistribution(0, 1)

    sampler = SensitivitySampler()
    _, mean = sampler.sample_sensitivity(Mean(), L1SensitivityNorm(),
                                         distribution, n_data_size=100,
                                         m_sample_size=285, gamma=0.33)

    assert np.abs(mean - 0) < 0.5


def test_l2_sensitivity_norm():
    """Checks sensitivity sampler using the L2 norm."""
    distribution = NormalDistribution(0, 1)

    sampler = SensitivitySampler()
    _, mean = sampler.sample_sensitivity(Mean(), L2SensitivityNorm(),
                                         distribution, n_data_size=100,
                                         m_sample_size=285, gamma=0.33)

    assert np.abs(mean - 0) < 0.5


@pytest.mark.parametrize("sensitivity_norm", [L1SensitivityNorm,
                                              L2SensitivityNorm])
def test_sensitivity_norm_list_of_arrays(sensitivity_norm):
    """Checks that the sensitivity norm is computed on each item of a list of arrays."""

    def reshape_to_list(data):
        """Reshapes the input data into a list of arrays."""
        return list(np.reshape(data, (20, 30, 40)))
    query = type('ReshapeToList', (object,), {'get': reshape_to_list})

    distribution = NormalDistribution(0, 1.5)
    sampler = SensitivitySampler()

    max_sensitivity, mean_sensitivity = \
        sampler.sample_sensitivity(query=query,
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

# TODO: need to check the ordering of the highest sensitivity for list of arrays
# def test_sensitivity_norm_nested():
#     """Checks that the sensitivity norm is computed correctly item-wise in a list of arrays.
#
#     The oracle returns a list of arrays. The query returns
#     the item-wise average (still a list of arrays).
#     So the sensitivity is expected to be correctly estimated
#     for each item in the list.
#     """
#     def array_list(size):
#         """Returns a list of arrays of specified length."""
#         arrays = [np.random.rand(size, 3),
#                   np.random.rand(size, 2),
#                   np.random.rand(size, 4)]
#
#         return arrays
#     oracle = type('NestedArrays', (object,), {'sample': array_list})
#
#     def item_average(data):
#         """Returns the item-wise average of a list of arrays."""
#         return [item.mean(axis=0) for item in data]
#     query = type('ItemAverage', (object,), {'get': item_average})
#
#     sensitivity_norm = L1SensitivityNorm()
#     sampler = SensitivitySampler()
#     max_sensitivity, mean_sensitivity = \
#         sampler.sample_sensitivity(query=query,
#                                    sensitivity_norm=sensitivity_norm,
#                                    oracle=oracle,
#                                    n_data_size=4,
#                                    m_sample_size=3,
#                                    gamma=0.33)
#     print("max_sensitivity", max_sensitivity)
#     print("mean_sensitivity", mean_sensitivity)
#     suca


class SensitivitySamplerTest(SensitivitySampler):
    """Allows to access the private members of the parent class."""
    def concatenate(self, x_1, x_2):
        """Calls the protected member of the parent class."""
        return self._concatenate(x_1, x_2)


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
