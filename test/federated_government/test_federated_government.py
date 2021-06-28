from unittest.mock import Mock, patch
import pytest
import numpy as np

from shfl.federated_government.federated_government import FederatedGovernment
from shfl.data_base.data_base import LabeledDatabase
from shfl.data_distribution.data_distribution_iid import IidDataDistribution


class DataBaseTest(LabeledDatabase):
    """Creates a test class for a random data base."""

    def load_data(self):
        self._train_data = np.random.rand(200).reshape([40, 5])
        self._test_data = np.random.rand(200).reshape([40, 5])
        self._train_labels = np.random.randint(0, 10, 40)
        self._test_labels = np.random.randint(0, 10, 40)


@pytest.fixture(name="data_distribution")
def fixture_nodes_federation(global_vars):
    """Returns the federated data, test data and label."""
    database = DataBaseTest()
    database.load_data()
    data_distribution = IidDataDistribution(database)

    num_nodes = global_vars["n_nodes"]
    nodes_federation, test_data, test_labels = \
        data_distribution.get_nodes_federation(num_nodes=num_nodes)

    return nodes_federation, test_data, test_labels


def test_initialization(data_distribution, helpers):
    """Checks that the vertical federated government is correctly initialized."""
    nodes_federation, _, _ = data_distribution
    model = Mock()
    aggregator = Mock()
    federated_government = FederatedGovernment(model, nodes_federation, aggregator)

    helpers.check_initialization(federated_government)


def test_error_aggregator_and_server_node_not_provided(data_distribution):
    """Checks that an error is raised if neither the aggregator nor
        the server node are provided."""
    nodes_federation, _, _ = data_distribution
    model = Mock()

    with pytest.raises(AssertionError):
        FederatedGovernment(model, nodes_federation)


@patch("shfl.private.federated_operation.ServerDataNode")
@patch("shfl.private.federated_operation.NodesFederation")
def test_run_rounds(nodes_federation, server_node, data_distribution):
    """Checks that the federated round is called correctly."""
    _, test_data, test_labels = data_distribution
    model = Mock()
    aggregator = Mock()
    federated_government = FederatedGovernment(model, nodes_federation, aggregator, server_node)

    federated_government.evaluate_clients = Mock()
    federated_government.run_rounds(1, test_data, test_labels)

    server_node.deploy_collaborative_model.assert_called_once()
    nodes_federation.train_model.assert_called_once()
    federated_government.evaluate_clients.assert_called_once_with(test_data, test_labels)
    server_node.aggregate_weights.assert_called_once()
    server_node.evaluate_collaborative_model.assert_called_once_with(test_data, test_labels)


def test_evaluate_clients(global_vars, data_distribution):
    """Checks that all the clients are evaluated correctly.

    Both evaluations on global and local test data are considered."""
    _, test_data, test_labels = data_distribution
    num_nodes = global_vars["n_nodes"]
    nodes_federation = [Mock() for _ in range(num_nodes)]
    for data_node in nodes_federation:
        data_node.evaluate.return_value = np.random.rand(2)
    model = Mock()
    aggregator = Mock()

    federated_government = FederatedGovernment(model, nodes_federation, aggregator)
    federated_government.evaluate_clients(test_data, test_labels)

    for data_node in nodes_federation:
        assert data_node.evaluate.called_once_with(test_data, test_labels)
