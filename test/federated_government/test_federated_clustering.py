from shfl.model.kmeans_model import KMeansModel
from unittest.mock import Mock, patch
import pytest

from shfl.federated_government.federated_clustering import FederatedClustering
from shfl.federated_aggregator.cluster_fedavg_aggregator import ClusterFedAvgAggregator
from shfl.data_distribution.data_distribution_non_iid import NonIidDataDistribution


def test_federated_clustering():
    database = 'IRIS'
    cfg = FederatedClustering(database, num_nodes=3, percent=20)

    assert cfg._test_data is not None
    assert cfg._test_labels is not None
    assert isinstance(cfg._server._aggregator, ClusterFedAvgAggregator)
    assert isinstance(cfg._server._model, KMeansModel)
    assert cfg._federated_data is not None

    cfg = FederatedClustering(database,
                              data_distribution=NonIidDataDistribution,
                              num_nodes=3, percent=20)

    assert cfg._test_data is not None
    assert cfg._test_labels is not None
    assert isinstance(cfg._server._aggregator, ClusterFedAvgAggregator)
    assert isinstance(cfg._server._model, KMeansModel)
    assert cfg._federated_data is not None


def test_federated_clustering_wrong_database():
    with pytest.raises(ValueError):
        cfg = FederatedClustering('MNIST', num_nodes=3, percent=20)


def test_run_rounds():
    cfg = FederatedClustering('IRIS', num_nodes=3, percent=20)

    cfg._server.deploy_collaborative_model = Mock()
    cfg._federated_data.train_model = Mock()
    cfg.evaluate_clients = Mock()
    cfg._server.aggregate_weights = Mock()
    cfg._server.evaluate_collaborative_model = Mock()

    cfg.run_rounds(1)

    cfg._server.deploy_collaborative_model.assert_called_once()
    cfg._federated_data.train_model.assert_called_once()
    cfg.evaluate_clients.assert_called_once_with(cfg._test_data,
                                                 cfg._test_labels)
    cfg._server.aggregate_weights.assert_called_once()
    cfg._server.evaluate_collaborative_model.assert_called_once_with(
        cfg._test_data, cfg._test_labels)


def test_run_rounds_wrong_database():
    with pytest.raises(ValueError):
        cfg = FederatedClustering('EMNIST', num_nodes=3, percent=20)
