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

"""Contains fixtures used across the module."""
import pytest


@pytest.fixture(name="global_vars")
def fixture_global_vars():
    """Defines the characteristics of the federated simulation."""
    federated_learning_variables = {"n_features": 23,
                                    "n_classes": 2,
                                    "n_embeddings": 3,
                                    "num_data": 100,
                                    "batch_size": 32,
                                    "n_nodes": 3,
                                    "metrics": [0, 1, 2, 3]}

    return federated_learning_variables


class Helpers:
    """Delivers static helper functions to avoid duplicated code."""

    @staticmethod
    def check_initialization(federated_government):
        """Checks the initialization of high-level federated learning objects."""
        assert hasattr(federated_government, "_nodes_federation")
        assert hasattr(federated_government, "_server")

    @staticmethod
    def check_initialization_high_level(federated_government, fed_gov_init):
        """Checks the initialization of high-level federated learning objects."""
        fed_gov_init.assert_called_once()
        assert hasattr(federated_government, "_test_data")
        assert hasattr(federated_government, "_test_labels")


@pytest.fixture
def helpers():
    """Returns the helpers class."""
    return Helpers
