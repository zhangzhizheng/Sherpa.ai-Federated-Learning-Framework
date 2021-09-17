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

# Using method overloading: only one public method needed
# pylint: disable=too-few-public-methods
import copy
import math
import numpy as np
from scipy import special
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic


class SensitivitySampler:
    """Estimates the sensitivity of a generic query.

    Allows to empirically estimate the sensitivity
    of a query through a sampling procedure using a specified norm.
    This is motivated by the fact that, for a generic query
    (e.g. a function, a model), it can be difficult
    to analytically compute its sensitivity.

    # Example:
        See how to sample the sensitivity of
        a linear regression model in the
        [linear regression notebook](https://github.com/
        sherpaai/Sherpa.ai-Federated-Learning-Framework/blob/master/
        notebooks/federated_models/federated_models_linear_regression.ipynb).
        The same procedure can be applied to estimate the sensitivity
        of any model or function.

    # References:
        [Pain-free random differential privacy with
        sensitivity sampling](https://arxiv.org/pdf/1706.02562.pdf)

        [diffpriv: An R Package for Easy Differential
        Privacy](http://www.bipr.net/diffpriv/articles/diffpriv.pdf)
    """

    def __init__(self):
        self._sort_axis = 0
        self._concatenate_axis = 0

    # Must use all arguments in this case
    # pylint: disable=too-many-arguments
    def sample_sensitivity(self, query, sensitivity_norm, oracle, n_data_size,
                           m_sample_size=None, gamma=None):
        """Samples the sensitivity of a generic query.

        Either `m_sample_size` or `gamma` must be provided. For example small
        sample sizes values (few hundred) reflects limited sampling time.
        On the other hand, small `gamma` (e.g., 0.05) prioritizes privacy.

        # Arguments:
            query: Function to apply on private data.
            sensitivity_norm: Function defining the norm to use
                (see [Norm](../norm)).
            oracle: Probability distribution to sample from.
            n_data_size: The size of the private data set.
            m_sample_size: The sensitivity sample size.
            gamma: The desired privacy confidence level. The resulting
                random differential privacy holds with probability at least 1-gamma.

        # Returns:
            sensitivity: Maximum sampled sensitivity.
            sensitivity_mean: Mean sampled sensitivity.
        """
        optimal_sampling_values = self._get_optimal_values(m_sample_size=m_sample_size,
                                                           gamma=gamma)

        sensitivity, mean = self._compute_sensitivity(
            query=query,
            sensitivity_norm=sensitivity_norm,
            oracle=oracle,
            n_data_size=n_data_size,
            m_sample_size=int(optimal_sampling_values['m_sample_size']),
            k_highest=int(optimal_sampling_values['k_highest']))
        return sensitivity, mean

    def _compute_sensitivity(self, query, sensitivity_norm, oracle,
                             n_data_size, m_sample_size, k_highest):
        """Computes the sensitivity of the samples.

        Each sensitivity computation is as follows:
            1) Two neighbouring databases are created by sampling
                on the original database.
            2) The sensitivity is computed as
                the norm of the difference of the query outcomes
                over the two neighbouring databases.

        # Arguments:
            query: Function to apply on private the data
            (see: [Query](../../private/query)).
            sensitivity_norm: Function defining the norm to use (see: [Norm](../norm)).
            oracle: Probability distribution to sample from.
            n_data_size: Integer representing the size of the data
                to use in each sample.
            m_sample_size: Integer representing the sample size.
            k_highest: k order statistic index in {1,...,m_sample_size}.

        # Returns:
            sensitivity: Maximum sampled sensitivity.
            sensitivity_mean: Mean sampled sensitivity.
        """
        sensitivity_sampled = [np.inf] * m_sample_size

        for i in range(0, m_sample_size):
            data_base_1 = oracle.sample(n_data_size - 1)
            data_base_2 = data_base_1
            data_base_1 = self._concatenate(data_base_1, oracle.sample(1))
            data_base_2 = self._concatenate(data_base_2, oracle.sample(1))
            sensitivity_sampled[i] = \
                self._sensitivity_norm(query, sensitivity_norm,
                                       data_base_1, data_base_2)

        return self._sort_sensitivity(*sensitivity_sampled, k_highest=k_highest)

    @staticmethod
    def _sensitivity_norm(query, sensitivity_norm, data_base_1, data_base_2):
        """Queries two neighbouring databases and computes the norm
        of the difference of the results.

        # Arguments:
            query: Function to apply on private data.
            sensitivity_norm: Function to compute the sensitivity norm
                (see: [Norm](../norm)).
            data_base_1: The database to be queried.
            data_base_2: The database to be queried.

        # Returns:
            The norm of the difference of the queries.
        """

        return sensitivity_norm.compute(query(data_base_1),
                                        query(data_base_2))

    @staticmethod
    def _get_optimal_values(m_sample_size, gamma):
        """Computes the optimal values for m_sample_size, gamma, k_highest and rho.

        # Arguments:
            m_sample_size: Integer representing the sample size.
            gamma: Float representing the privacy confidence level.

        # Returns:
            A dictionary with the optimal values.
        """
        if m_sample_size is None:
            lambert_value = np.real(
                special.lambertw(-gamma / (2 * np.exp(0.5)), 1))
            rho = np.exp(lambert_value + 0.5)
            m_sample_size = np.ceil(
                np.log(1 / rho) / (2 * math.pow((gamma - rho), 2)))
            gamma_lo = rho + np.sqrt(np.log(1 / rho) / (2 * m_sample_size))
            k_highest = np.ceil(m_sample_size * (1 - gamma + gamma_lo))
        else:
            rho = np.exp(
                np.real(special.lambertw(-1 / (4 * m_sample_size), 1)) / 2)
            gamma_lo = rho + np.sqrt(np.log(1 / rho) / (2 * m_sample_size))
            if gamma is None:
                gamma = gamma_lo
                k_highest = m_sample_size
            else:
                k_highest = np.ceil(m_sample_size * (1 - gamma + gamma_lo))

        optimal_values = {'m_sample_size': m_sample_size,
                          'gamma': gamma,
                          'k_highest': k_highest,
                          'rho': rho}

        return optimal_values

    @dispatch(Variadic[(np.ndarray, list, tuple)])
    def _sort_sensitivity(self, *sensitivity_sampled, k_highest):
        """Sorts arrays or lists of arrays.
        """
        sensitivity_sorted = [self._sort_sensitivity(*item, k_highest=k_highest)
                              for item in zip(*sensitivity_sampled)]
        sensitivity_k_moment = [item[0] for item in sensitivity_sorted]
        sensitivity_mean = [item[1] for item in sensitivity_sorted]

        return sensitivity_k_moment, sensitivity_mean

    @dispatch(Variadic[np.ScalarType])
    def _sort_sensitivity(self, *sensitivity_sampled, k_highest):
        """Sorts scalars.
        """
        sensitivity_sorted = np.sort(sensitivity_sampled, axis=self._sort_axis)
        sensitivity_k_moment = sensitivity_sorted[k_highest - 1]
        sensitivity_mean = sensitivity_sorted.mean()

        return sensitivity_k_moment, sensitivity_mean

    @dispatch((np.ScalarType, np.ndarray), (np.ScalarType, np.ndarray))
    def _concatenate(self, x_1, x_2):
        return np.concatenate((x_1, x_2), axis=self._concatenate_axis)

    @dispatch((list, dict), (list, dict))
    def _concatenate(self, x_1, x_2):
        output = copy.deepcopy(x_1)
        for i, j in zip(self._seq_iter(x_1), self._seq_iter(x_2)):
            output[i] = self._concatenate(x_1[i], x_2[j])
        return output

    @staticmethod
    def _seq_iter(obj):
        return obj if isinstance(obj, dict) else range(len(obj))
