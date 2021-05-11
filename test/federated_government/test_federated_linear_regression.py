from shfl.federated_government.federated_linear_regression import FederatedLinearRegression
from shfl.federated_aggregator.fedavg_aggregator import FedAvgAggregator
from shfl.model.linear_regression_model import LinearRegressionModel
from unittest.mock import Mock
import pytest


def test_federated_linear_regression():
    database = 'CALIFORNIA'
    federated_linear_regression = FederatedLinearRegression(database, num_nodes=3, percent=20)

    assert federated_linear_regression._test_data is not None
    assert federated_linear_regression._test_labels is not None
    assert isinstance(federated_linear_regression._server._aggregator, FedAvgAggregator)
    assert isinstance(federated_linear_regression._server._model, LinearRegressionModel)
    assert federated_linear_regression._federated_data is not None


def test_federated_linear_regression_wrong_database():
    database = 'MNIST'
    with pytest.raises(ValueError):
        FederatedLinearRegression(database, num_nodes=3, percent=20)


def test_run_rounds():
    database = 'CALIFORNIA'
    federated_linear_regression = FederatedLinearRegression(database, num_nodes=3, percent=20)

    federated_linear_regression._server.deploy_collaborative_model = Mock()
    federated_linear_regression._federated_data.train_model = Mock()
    federated_linear_regression.evaluate_clients = Mock()
    federated_linear_regression._server.aggregate_weights = Mock()
    federated_linear_regression._server.evaluate_collaborative_model = Mock()

    federated_linear_regression.run_rounds(1, )

    federated_linear_regression._server.deploy_collaborative_model.assert_called_once()
    federated_linear_regression._federated_data.train_model.assert_called_once()
    federated_linear_regression.evaluate_clients.assert_called_once_with(
        federated_linear_regression._test_data, federated_linear_regression._test_labels)
    federated_linear_regression._server.aggregate_weights.assert_called_once()
    federated_linear_regression._server.evaluate_collaborative_model.assert_called_once_with(
        federated_linear_regression._test_data, federated_linear_regression._test_labels)
