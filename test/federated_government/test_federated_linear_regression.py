from shfl.federated_government.federated_linear_regression import FederatedLinearRegression, LinearRegressionDataBases
from shfl.federated_aggregator.fedavg_aggregator import FedAvgAggregator
from shfl.model.linear_regression_model import LinearRegressionModel
from unittest.mock import Mock, patch
import pytest


def test_FederatedLinearRegression():
    database = 'CALIFORNIA'
    lrfg = FederatedLinearRegression(database, num_nodes=3, percent=20)

    module = LinearRegressionDataBases.__members__[database].value
    data_base = module()
    train_data, train_labels, test_data, test_labels = data_base.load_data()

    assert lrfg._test_data is not None
    assert lrfg._test_labels is not None
    assert lrfg._num_features == train_data.shape[1]
    assert isinstance(lrfg._server._aggregator, FedAvgAggregator)
    assert isinstance(lrfg._server._model, LinearRegressionModel)
    assert lrfg._federated_data is not None


def test_FederatedLinearRegression_wrong_database():
    database = 'MNIST'
    with pytest.raises(ValueError):
        lrfg = FederatedLinearRegression(database, num_nodes=3, percent=20)



def test_run_rounds():
    database = 'CALIFORNIA'
    lrfg = FederatedLinearRegression(database, num_nodes=3, percent=20)

    lrfg._server.deploy_collaborative_model = Mock()
    lrfg._federated_data.train_model = Mock()
    lrfg.evaluate_clients = Mock()
    lrfg._server.aggregate_weights = Mock()
    lrfg._server.evaluate_collaborative_model = Mock()

    lrfg.run_rounds(1)

    lrfg._server.deploy_collaborative_model.assert_called_once()
    lrfg._federated_data.train_model.assert_called_once()
    lrfg.evaluate_clients.assert_called_once_with(
        lrfg._test_data, lrfg._test_labels)
    lrfg._server.aggregate_weights.assert_called_once()
    lrfg._server.evaluate_collaborative_model.assert_called_once_with(
        lrfg._test_data, lrfg._test_labels)


@patch('shfl.federated_government.federated_linear_regression.LinearRegressionModel')
def test_model_builder(mock_linearegression):
    database = 'CALIFORNIA'
    lrfg = FederatedLinearRegression(database, num_nodes=3, percent=20)

    model = lrfg.model_builder()

    assert isinstance(model, Mock)
    mock_linearegression.assert_called_with(n_features=lrfg._num_features)
