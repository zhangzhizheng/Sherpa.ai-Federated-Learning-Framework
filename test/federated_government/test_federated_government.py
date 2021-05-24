from unittest.mock import Mock, patch
import pytest
import numpy as np

from shfl.federated_government.federated_government import FederatedGovernment
from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution_iid import IidDataDistribution


class DataBaseTest(DataBase):
    """Creates a test class for a random data base."""

    def load_data(self):
        self._train_data = np.random.rand(200).reshape([40, 5])
        self._test_data = np.random.rand(200).reshape([40, 5])
        self._train_labels = np.random.randint(0, 10, 40)
        self._test_labels = np.random.randint(0, 10, 40)


@pytest.fixture(name="data_distribution")
def fixture_federated_data(global_vars):
    """Returns the federated data, test data and label."""
    database = DataBaseTest()
    database.load_data()
    data_distribution = IidDataDistribution(database)

    num_nodes = global_vars["n_nodes"]
    federated_data, test_data, test_labels = \
        data_distribution.get_federated_data(num_nodes=num_nodes)

    return federated_data, test_data, test_labels


def test_initialization(data_distribution, helpers):
    """Checks that the vertical federated government is correctly initialized."""
    federated_data, _, _ = data_distribution
    model = Mock()
    aggregator = Mock()
    federated_government = FederatedGovernment(model, federated_data, aggregator)

    helpers.check_initialization(federated_government)


@patch("shfl.private.federated_operation.ServerDataNode")
@patch("shfl.private.federated_operation.FederatedData")
def test_run_rounds(federated_data, server_node, data_distribution):
    """Checks that the federated round is called correctly."""
    _, test_data, test_labels = data_distribution
    model = Mock()
    aggregator = Mock()
    federated_government = FederatedGovernment(model, federated_data,
                                               aggregator, server_node)

    federated_government.evaluate_clients = Mock()
    federated_government.run_rounds(1, test_data, test_labels)

    server_node.deploy_collaborative_model.assert_called_once()
    federated_data.train_model.assert_called_once()
    federated_government.evaluate_clients.assert_called_once_with(test_data, test_labels)
    server_node.aggregate_weights.assert_called_once()
    server_node.evaluate_collaborative_model.assert_called_once_with(test_data, test_labels)


def test_evaluate_clients(global_vars, data_distribution):
    """Checks that all the clients are evaluated correctly.

    Both evaluations on global and local test data are considered."""
    _, test_data, test_labels = data_distribution
    num_nodes = global_vars["n_nodes"]
    federated_data = [Mock() for _ in range(num_nodes)]
    for data_node in federated_data:
        data_node.evaluate.return_value = np.random.rand(2)
    model = Mock()
    aggregator = Mock()

    federated_government = FederatedGovernment(model, federated_data, aggregator)
    federated_government.evaluate_clients(test_data, test_labels)

    for data_node in federated_data:
        assert data_node.evaluate.called_once_with(test_data, test_labels)
