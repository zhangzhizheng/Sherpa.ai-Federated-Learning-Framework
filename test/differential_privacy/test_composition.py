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

import pytest
import numpy as np

from shfl.private import DataNode
from shfl.private.utils import unprotected_query
from shfl.differential_privacy.composition import ExceededPrivacyBudgetError
from shfl.differential_privacy.composition import AdaptiveDifferentialPrivacy
from shfl.differential_privacy.mechanism import GaussianMechanism


def test_exception():
    """Checks the exception is correctly thrown."""
    exception = ExceededPrivacyBudgetError(epsilon_delta=1)
    assert str(exception) is not None


def test_get_epsilon_delta():
    """Checks the getter of epsilon and delta parameters in the adaptive mechanism."""
    e_d = (1, 1)
    data_access_definition = AdaptiveDifferentialPrivacy(epsilon_delta=e_d)

    assert data_access_definition.epsilon_delta == e_d


def test_constructor_bad_params():
    """Checks that an exception is thrown with wrong epsilon and delta parameters."""
    with pytest.raises(ValueError):
        AdaptiveDifferentialPrivacy(epsilon_delta=(1, 2, 3))

    with pytest.raises(ValueError):
        AdaptiveDifferentialPrivacy(epsilon_delta=(-1, 2))

    with pytest.raises(ValueError):
        AdaptiveDifferentialPrivacy(epsilon_delta=(1, -2))

    with pytest.raises(ValueError):
        AdaptiveDifferentialPrivacy(epsilon_delta=(1, 1), mechanism=unprotected_query)


def test_configure_data_access_no_mechanism():
    """Checks that an exception is thrown when the data access is not configured properly.

    In this case, a mechanism is not provided."""
    data_access_definition = AdaptiveDifferentialPrivacy(epsilon_delta=(1, 1))
    data_node = DataNode()
    data_node.set_private_data("test", np.array(range(10)))
    data_node.configure_data_access("test", data_access_definition)
    with pytest.raises(ValueError):
        data_node.query("test")


def test_data_access():
    """Checks data is accessed correctly when the access is configured properly.

    We need to define the data access policy and a differentially private mechanism.
    """
    data_access_definition = AdaptiveDifferentialPrivacy(epsilon_delta=(1, 1))
    data_node = DataNode()
    array = np.array(range(10))
    data_node.set_private_data("test", array)

    data_node.configure_data_access("test", data_access_definition)
    query_result = data_node.query("test", mechanism=GaussianMechanism(1, epsilon_delta=(0.1, 1)))

    assert query_result is not None


def test_exception_exceeded_privacy_budget_error_scalar():
    """Checks that the exception is thrown when the privacy budget is exceeded.

    The private data is simply a scalar.
    """
    scalar = 175

    dp_mechanism = GaussianMechanism(1, epsilon_delta=(0.1, 1))
    data_access_definition = \
        AdaptiveDifferentialPrivacy(epsilon_delta=(1, 0), mechanism=dp_mechanism)
    node = DataNode()
    node.set_private_data("scalar", scalar)
    node.configure_data_access("scalar", data_access_definition)

    with pytest.raises(ExceededPrivacyBudgetError):
        node.query("scalar")


def test_exception_exceeded_privacy_budget_error_array():
    """Checks that the exception is thrown when the privacy budget is exceeded.

    The private data is a vector. The mechanism is defined together with the
    data access definition.
    """
    dp_mechanism = GaussianMechanism(1, epsilon_delta=(0.1, 1))
    data_access_definition = AdaptiveDifferentialPrivacy(epsilon_delta=(1, 1),
                                                         mechanism=dp_mechanism)
    data_node = DataNode()
    array = np.array(range(10))
    data_node.set_private_data("test", array)
    data_node.configure_data_access("test", data_access_definition)

    with pytest.raises(ExceededPrivacyBudgetError):
        for _ in range(1, 1000):
            data_node.query("test")


def test_exception_exceeded_privacy_budget_error_array_2():
    """Checks that the exception is thrown when the privacy budget is exceeded.

    The private data is a vector. The mechanism is defined later in query's argument.
        """
    data_access_definition = AdaptiveDifferentialPrivacy(epsilon_delta=(1, 0.001))
    data_node = DataNode()
    array = np.array(range(10))
    data_node.set_private_data("test", array)
    data_node.configure_data_access("test", data_access_definition)

    with pytest.raises(ExceededPrivacyBudgetError):
        for _ in range(1, 1000):
            data_node.query("test", mechanism=GaussianMechanism(1, epsilon_delta=(0.1, 1)))
