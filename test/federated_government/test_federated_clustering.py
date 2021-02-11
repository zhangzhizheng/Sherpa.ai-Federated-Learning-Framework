from shfl.federated_government.federated_clustering import FederatedClustering, ClusteringDataBases
from shfl.federated_aggregator.cluster_fedavg_aggregator import ClusterFedAvgAggregator
from shfl.model.kmeans_model import KMeansModel
from unittest.mock import Mock, patch
import pytest

import numpy as np


def test_FederatedClustering():
    database = 'IRIS'
    cfg = FederatedClustering(database, iid=True, num_nodes=3, percent=20)

    module = ClusteringDataBases.__members__[database].value
    data_base = module()
    train_data, train_labels, test_data, test_labels = data_base.load_data()

    assert cfg._test_data is not None
    assert cfg._test_labels is not None
    assert cfg._num_clusters == len(np.unique(train_labels))
    assert cfg._num_features == train_data.shape[1]
    assert isinstance(cfg._server._aggregator, ClusterFedAvgAggregator)
    assert isinstance(cfg._server._model, KMeansModel)
    assert cfg._federated_data is not None

    cfg = FederatedClustering(database, iid=False, num_nodes=3, percent=20)

    assert cfg._test_data is not None
    assert cfg._test_labels is not None
    assert cfg._num_clusters == len(np.unique(train_labels))
    assert cfg._num_features == train_data.shape[1]
    assert isinstance(cfg._server._aggregator, ClusterFedAvgAggregator)
    assert isinstance(cfg._server._model, KMeansModel)
    assert cfg._federated_data is not None


def test_FederatedClustering_wrong_database():
    with pytest.raises(ValueError):
        cfg = FederatedClustering('MNIST', iid=True, num_nodes=3, percent=20)


def test_run_rounds():
    cfg = FederatedClustering('IRIS', iid=True, num_nodes=3, percent=20)

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
        cfg = FederatedClustering('EMNIST', iid=True, num_nodes=3, percent=20)


@patch('shfl.federated_government.federated_clustering.KMeansModel')
def test_model_builder(mock_kmeans):
    cfg = FederatedClustering('IRIS', iid=True, num_nodes=3, percent=20)

    model = cfg.model_builder()

    assert isinstance(model, Mock)
    mock_kmeans.assert_called_with(n_clusters=cfg._num_clusters,
                                   n_features=cfg._num_features)
