from unittest.mock import Mock
import random
import string
import pytest

from shfl.federated_government.federated_images_classifier import FederatedImagesClassifier
from shfl.model.deep_learning_model import DeepLearningModel
from shfl.federated_aggregator.federated_aggregator import FederatedAggregator
from shfl.private.federated_operation import FederatedData
from shfl.data_distribution.data_distribution_non_iid import NonIidDataDistribution


def test_images_classifier_iid():
    database = "EMNIST"
    federated_classifier = FederatedImagesClassifier(database,
                                                     num_nodes=3,
                                                     percent=5)

    for node in federated_classifier._federated_data:
        print("model", node._model)
        assert isinstance(node._model, DeepLearningModel)

    assert isinstance(federated_classifier._server._model, DeepLearningModel)
    assert isinstance(federated_classifier._server._aggregator, FederatedAggregator)
    assert isinstance(federated_classifier._federated_data, FederatedData)

    assert federated_classifier._test_data is not None
    assert federated_classifier._test_labels is not None


def test_images_classifier_no_iid():
    database = "EMNIST"
    federated_classifier = FederatedImagesClassifier(database,
                                                     data_distribution=NonIidDataDistribution,
                                                     num_nodes=3,
                                                     percent=5)

    for node in federated_classifier._federated_data:
        assert isinstance(node._model, DeepLearningModel)

    assert isinstance(federated_classifier._server._model, DeepLearningModel)
    assert isinstance(federated_classifier._server._aggregator, FederatedAggregator)
    assert isinstance(federated_classifier._federated_data, FederatedData)

    assert federated_classifier._test_data is not None
    assert federated_classifier._test_labels is not None


def test_images_classifier_wrong_database():
    letters = string.ascii_lowercase
    wrong_database = ''.join(random.choice(letters) for _ in range(10))

    with pytest.raises(ValueError):
        FederatedImagesClassifier(wrong_database)


def test_run_rounds():
    database = "EMNIST"
    federated_classifier = FederatedImagesClassifier(database,
                                                     num_nodes=3,
                                                     percent=5)

    federated_classifier._server.deploy_collaborative_model = Mock()
    federated_classifier._federated_data.train_model = Mock()
    federated_classifier.evaluate_clients = Mock()
    federated_classifier._server.aggregate_weights = Mock()
    federated_classifier._server.evaluate_collaborative_model = Mock()

    federated_classifier.run_rounds(1)

    federated_classifier._server.deploy_collaborative_model.assert_called_once()
    federated_classifier._federated_data.train_model.assert_called_once()
    federated_classifier.evaluate_clients.assert_called_once_with(
        federated_classifier._test_data, federated_classifier._test_labels)
    federated_classifier._server.aggregate_weights.assert_called_once()
    federated_classifier._server.evaluate_collaborative_model.assert_called_once_with(
        federated_classifier._test_data, federated_classifier._test_labels)
