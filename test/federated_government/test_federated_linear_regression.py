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

from unittest.mock import patch
import pytest

from shfl.federated_government.federated_linear_regression import FederatedLinearRegression


@patch("shfl.federated_government.FederatedGovernment.__init__")
def test_initialization(fed_gov_init):
    """Checks that the federated linear regression is correctly initialized."""
    database = "CALIFORNIA"
    federated_government = FederatedLinearRegression(database, num_nodes=3, percent=5)
    fed_gov_init.assert_called_once()
    assert hasattr(federated_government, "_test_data")
    assert hasattr(federated_government, "_test_labels")


def test_initialization_wrong_database():
    """Checks that an error is raised when a wrong database is requested."""
    wrong_database = "MNIST"

    with pytest.raises(ValueError):
        FederatedLinearRegression(wrong_database)


@patch("shfl.federated_government.FederatedGovernment.run_rounds")
def test_run_rounds(fed_gov_run_rounds):
    """Checks that the federated round is called correctly."""
    database = 'CALIFORNIA'
    federated_government = FederatedLinearRegression(database,
                                                     num_nodes=3,
                                                     percent=20)
    federated_government.run_rounds(1)
    fed_gov_run_rounds.assert_called_once()
