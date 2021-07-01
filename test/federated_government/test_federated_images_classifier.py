from unittest.mock import patch
import pytest

from shfl.federated_government.federated_images_classifier import FederatedImagesClassifier
from shfl.data_distribution.data_distribution_non_iid import NonIidDataDistribution


@patch("shfl.federated_government.FederatedGovernment.__init__")
def test_initialization(fed_gov_init, helpers):
    """Checks that the federated images classifier is correctly initialized."""
    database = "EMNIST"
    federated_government = FederatedImagesClassifier(database, num_nodes=3, percent=5)

    helpers.check_initialization_high_level(federated_government, fed_gov_init)


@patch("shfl.federated_government.FederatedGovernment.__init__")
def test_initialization_non_iid(fed_gov_init, helpers):
    """Checks that the federated images classifier is correctly initialized
    for the non-iid case."""
    database = "EMNIST"
    federated_government = \
        FederatedImagesClassifier(database, data_distribution=NonIidDataDistribution,
                                  num_nodes=3, percent=5)

    helpers.check_initialization_high_level(federated_government, fed_gov_init)


def test_initialization_wrong_database():
    """Checks that an error is raised when a wrong database is requested."""
    wrong_database = "IRIS"

    with pytest.raises(ValueError):
        FederatedImagesClassifier(wrong_database)


@patch("shfl.federated_government.FederatedGovernment.run_rounds")
def test_run_rounds(fed_gov_run_rounds):
    """Checks that the federated round is called correctly."""
    database = "EMNIST"
    federated_government = FederatedImagesClassifier(database,
                                                     num_nodes=3,
                                                     percent=5)
    federated_government.run_rounds(1)
    fed_gov_run_rounds.assert_called_once()
