from unittest.mock import patch
import pytest

from shfl.federated_government.federated_clustering import FederatedClustering
from shfl.data_distribution.data_distribution_non_iid import NonIidDataDistribution


@patch("shfl.federated_government.FederatedGovernment.__init__")
def test_initialization(fed_gov_init, helpers):
    """Checks that the federated round is called correctly."""
    federated_government = FederatedClustering('IRIS', num_nodes=3, percent=20)

    helpers.check_initialization_high_level(federated_government, fed_gov_init)


@patch("shfl.federated_government.FederatedGovernment.__init__")
def test_initialization_non_iid(fed_gov_init, helpers):
    """Checks that the federated round is called correctly in the non-iid case."""
    federated_government = \
        FederatedClustering('IRIS', data_distribution=NonIidDataDistribution,
                            num_nodes=3, percent=20)

    helpers.check_initialization_high_level(federated_government, fed_gov_init)


def test_initialization_wrong_database():
    """Checks that an error is raised when a wrong database is requested."""
    with pytest.raises(ValueError):
        FederatedClustering('MNIST', num_nodes=3, percent=20)


@patch("shfl.federated_government.FederatedGovernment.run_rounds")
def test_run_rounds(fed_gov_run_rounds):
    """Checks that the federated round is called correctly."""
    federated_government = FederatedClustering('IRIS', num_nodes=3, percent=20)
    federated_government.run_rounds(1)
    fed_gov_run_rounds.assert_called_once()
