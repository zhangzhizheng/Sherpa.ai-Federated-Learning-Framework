import numpy as np
from unittest.mock import Mock
from unittest.mock import call
import pytest

from shfl.federated_government.vertical_federated_deep_learning import FederatedGovernmentVertical
from shfl.data_base.data_base import LabeledDatabase
from shfl.private.data import LabeledData
from shfl.data_distribution.data_distribution_plain import PlainDataDistribution
from shfl.private.data import DataAccessDefinition
from shfl.private.data import UnprotectedAccess
from shfl.private.federated_operation import VerticalServerDataNode
from shfl.data_base.data_base import vertical_split


class QueryMetaParameters(DataAccessDefinition):
    """Returns embeddings (or their gradients) as computed
                by the local model."""
    def apply(self, model, **kwargs):

        return model.get_meta_params(**kwargs)


class ComputeLoss(DataAccessDefinition):
    """Computes training loss."""
    def apply(self, data, **kwargs):
        embeddings = kwargs.get("embeddings")
        server_model = kwargs.get("server_model")

        return server_model.compute_loss(embeddings, data.label)


@pytest.fixture
def test_vertically_split_database():
    data = np.random.rand(1000).reshape([50, 20])
    labels = np.random.randint(0, 10, 50)
    n_nodes = 3
    train_data, train_labels, test_data, test_labels = \
        vertical_split(data, labels, indices_or_sections=n_nodes)

    labels_nodes = [train_labels for _ in range(n_nodes)]
    data_base = LabeledDatabase(data=train_data,
                                labels=labels_nodes,
                                train_percentage=1.,
                                shuffle=False)
    data_base.load_data()
    federated_data, _, _ = \
        PlainDataDistribution(database=data_base).get_federated_data()

    return federated_data, test_data, test_labels


@pytest.fixture
def test_node_models(test_vertically_split_database):
    federated_data, _, _ = test_vertically_split_database
    models = [Mock() for _ in range(federated_data.num_nodes())]

    return models


@pytest.fixture
def test_server_node(test_vertically_split_database, test_node_models):
    federated_data, _, _ = test_vertically_split_database
    server_node = VerticalServerDataNode(
        federated_data=federated_data,
        model=Mock(),
        aggregator=None,
        data=LabeledData(data=np.random.rand(100).reshape([5, 20]),
                         label=np.random.randint(0, 10, 50)))

    return server_node


@pytest.fixture
def test_vertical_federated_government(test_vertically_split_database,
                                       test_node_models,
                                       test_server_node):

    federated_data, test_data, test_labels = test_vertically_split_database
    vert_fed_gov = FederatedGovernmentVertical(
        models=test_node_models, federated_data=federated_data,
        server_node=test_server_node)

    return vert_fed_gov


def test_train_all_clients(test_vertical_federated_government):

    vert_fed_gov = test_vertical_federated_government

    vert_fed_gov._federated_data.train_model()

    vert_fed_gov._federated_data.configure_data_access(UnprotectedAccess())
    for node in vert_fed_gov._federated_data:
        labeled_data = node.query()
        node._model.train.assert_called_once_with(
            labeled_data.data, labeled_data.label)


def test_aggregate_weights(test_vertical_federated_government):

    vert_fed_gov = test_vertical_federated_government
    vert_fed_gov._federated_data.configure_model_access(QueryMetaParameters())

    for node in vert_fed_gov._federated_data:
        node._model.get_meta_params.return_value = np.random.randint(0, 10, 100)

    vert_fed_gov._server.aggregate_weights()

    for node in vert_fed_gov._federated_data:
        node._model.get_meta_params.assert_called_once()
    vert_fed_gov._server._model.train.assert_called_once()

    vert_fed_gov._server.configure_data_access(UnprotectedAccess())
    server_labeled_data = vert_fed_gov._server.query()
    np.testing.assert_array_equal(
        vert_fed_gov._server._model.train.call_args[0][0],
        server_labeled_data.data)
    np.testing.assert_array_equal(
        vert_fed_gov._server._model.train.call_args[0][1],
        server_labeled_data.label)

    embeddings = [node._model.get_meta_params.return_value
                  for node in vert_fed_gov._federated_data]
    for i_node in range(vert_fed_gov._federated_data.num_nodes()):
        np.testing.assert_array_equal(
            vert_fed_gov._server._model.train.call_args[1]["embeddings"][i_node],
            embeddings[i_node])


def test_train_all_clients_update_stage(test_vertical_federated_government):

    vert_fed_gov = test_vertical_federated_government
    embeddings_grads = np.random.rand(50)
    vert_fed_gov._federated_data.train_model(embeddings_grads=embeddings_grads)

    vert_fed_gov._federated_data.configure_data_access(UnprotectedAccess())
    for node in vert_fed_gov._federated_data:
        labeled_data = node.query()
        node._model.train.assert_called_once_with(
            labeled_data.data, labeled_data.label,
            embeddings_grads=embeddings_grads)


def test_compute_loss(test_vertical_federated_government):

    vert_fed_gov = test_vertical_federated_government
    vert_fed_gov._server.configure_data_access(ComputeLoss())
    vert_fed_gov._federated_data.configure_model_access(QueryMetaParameters())

    vert_fed_gov._server._model.compute_loss.return_value = np.random.random()
    for node in vert_fed_gov._federated_data:
        node._model.get_meta_params.return_value = np.random.randint(0, 10, 100)

    vert_fed_gov._server.compute_loss()

    embeddings = [node._model.get_meta_params.return_value
                  for node in vert_fed_gov._federated_data]
    for node in vert_fed_gov._federated_data:
        node._model.get_meta_params.assert_called_once()
    vert_fed_gov._server._model.compute_loss.assert_called_once()

    for i_node in range(vert_fed_gov._federated_data.num_nodes()):
        np.testing.assert_array_equal(
            vert_fed_gov._server._model.compute_loss.call_args[0][0][i_node],
            embeddings[i_node])
    vert_fed_gov._server.configure_data_access(UnprotectedAccess())
    server_labeled_data = vert_fed_gov._server.query()
    np.testing.assert_array_equal(
        vert_fed_gov._server._model.compute_loss.call_args[0][1],
        server_labeled_data.label)


def test_evaluate_collaborative_model(test_vertical_federated_government,
                                      test_vertically_split_database):

    federated_data, test_data, test_labels = test_vertically_split_database
    vert_fed_gov = test_vertical_federated_government

    for node in federated_data:
        node._model.predict.return_value = np.random.randint(0, 10, 100)
    vert_fed_gov._server._model.predict.return_value = \
        np.random.randint(0, 10, 100)

    vert_fed_gov._server.evaluate_collaborative_model(test_data, test_labels)

    for node, client_data in zip(federated_data, test_data):
        node._model.predict.assert_called_once()
        np.testing.assert_array_equal(
            node._model.predict.call_args[0][0], client_data)

    vert_fed_gov._server._model.predict.assert_called_once()
    embeddings = [node._model.predict.return_value for node in federated_data]
    for i_node in range(federated_data.num_nodes()):
        np.testing.assert_array_equal(
            vert_fed_gov._server._model.predict.call_args[0][0][i_node],
            embeddings[i_node])


def test_run_rounds(test_vertical_federated_government,
                    test_vertically_split_database):

    _, test_data, test_labels = test_vertically_split_database
    vert_fed_gov = test_vertical_federated_government

    vert_fed_gov._federated_data.train_model = Mock()
    vert_fed_gov._server.aggregate_weights = Mock()
    vert_fed_gov._server.query_model = Mock()
    vert_fed_gov._server.query_model.return_value = np.random.rand(50)
    vert_fed_gov._server.compute_loss = Mock()
    vert_fed_gov._server.evaluate_collaborative_model = Mock()

    vert_fed_gov.run_rounds(n=1, test_data=test_data, test_label=test_labels)

    train_model_calls = \
        [call(),
         call(embeddings_grads=vert_fed_gov._server.query_model.return_value)]
    vert_fed_gov._federated_data.train_model.assert_has_calls(train_model_calls)
    vert_fed_gov._server.aggregate_weights.assert_called_once()
    vert_fed_gov._server.query_model.assert_called_once()
    vert_fed_gov._server.compute_loss.assert_called_once()
    vert_fed_gov._server.evaluate_collaborative_model.assert_called_once()

