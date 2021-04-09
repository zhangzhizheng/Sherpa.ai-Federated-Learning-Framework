import numpy as np
from unittest.mock import Mock
from unittest.mock import call
import pytest

from shfl.federated_government.vertical_federated_government import VerticalFederatedGovernment
from shfl.data_base.data_base import LabeledDatabase
from shfl.private.data import LabeledData
from shfl.data_distribution.data_distribution_plain import PlainDataDistribution
from shfl.private.data import DataAccessDefinition
from shfl.private.data import UnprotectedAccess
from shfl.private.federated_operation import VerticalServerDataNode
from shfl.data_base.data_base import vertical_split


class TrainEvaluation(DataAccessDefinition):
    """Evaluate collaborative model on batch train data."""

    def apply(self, data, **kwargs):
        server_model = kwargs.get("server_model")
        embeddings, embeddings_indices = kwargs.get("meta_params")
        embeddings = np.sum(embeddings, axis=0)
        labels = data.label[embeddings_indices]

        evaluation = server_model.evaluate(embeddings, labels)

        return evaluation


class QueryMetaParameters(DataAccessDefinition):
    """
    Returns embeddings (or their gradients) as computed
    by the local model.
    """
    def apply(self, model, **kwargs):
        return model.get_meta_params(**kwargs)


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
def test_federated_data(test_vertically_split_database):
    federated_data, _, _ = test_vertically_split_database
    federated_data.configure_model_access(QueryMetaParameters())

    return federated_data


@pytest.fixture
def test_server_node(test_federated_data, test_node_models):
    server_node = VerticalServerDataNode(
        federated_data=test_federated_data,
        model=Mock(),
        data=LabeledData(data=np.random.rand(100).reshape([5, 20]),
                         label=np.random.randint(0, 10, 50)))
    server_node.configure_data_access(TrainEvaluation())
    server_node.configure_model_access(QueryMetaParameters())

    return server_node


@pytest.fixture
def test_vertical_federated_government(test_federated_data,
                                       test_node_models,
                                       test_server_node):
    vert_fed_gov = VerticalFederatedGovernment(
        models=test_node_models, federated_data=test_federated_data,
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

    embeddings_indices = np.random.randint(0, 10, 50)
    for node in vert_fed_gov._federated_data:
        embeddings = np.random.rand(50)
        node._model.get_meta_params.return_value = (embeddings,
                                                    embeddings_indices)

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

    meta_params = [node._model.get_meta_params.return_value
                   for node in vert_fed_gov._federated_data]
    embeddings = [item[0] for item in meta_params]
    embeddings_indices = [item[1] for item in meta_params]
    for i_node in range(vert_fed_gov._federated_data.num_nodes()):
        np.testing.assert_array_equal(
            vert_fed_gov._server._model.train.call_args[1]["meta_params"][0][i_node],
            embeddings[i_node])
    np.testing.assert_array_equal(
        vert_fed_gov._server._model.train.call_args[1]["meta_params"][1],
        embeddings_indices[0])


def test_aggregate_weights_non_matching_clients_indices(test_vertical_federated_government):

    vert_fed_gov = test_vertical_federated_government

    for node in vert_fed_gov._federated_data:
        embeddings_indices = np.random.randint(0, 10, 50)
        embeddings = np.random.rand(50)
        node._model.get_meta_params.return_value = (embeddings,
                                                    embeddings_indices)

    with pytest.raises(AssertionError):
        vert_fed_gov._server.aggregate_weights()


def test_train_all_clients_update_stage(test_vertical_federated_government):

    vert_fed_gov = test_vertical_federated_government
    embeddings_indices = np.random.randint(0, 10, 50)
    embeddings_grads = np.random.rand(50)
    meta_params = (embeddings_grads, embeddings_indices)

    vert_fed_gov._federated_data.train_model(meta_params=meta_params)

    vert_fed_gov._federated_data.configure_data_access(UnprotectedAccess())
    for node in vert_fed_gov._federated_data:
        labeled_data = node.query()
        node._model.train.assert_called_once_with(
            labeled_data.data, labeled_data.label,
            meta_params=meta_params)


def test_evaluate_collaborative_model_train_data(
        test_vertical_federated_government,
        test_vertically_split_database):

    federated_data, test_data, test_labels = test_vertically_split_database
    vert_fed_gov = test_vertical_federated_government

    embeddings_indices = np.random.randint(0, 10, 100)
    for node in federated_data:
        embeddings = np.random.rand(100)
        meta_params = (embeddings, embeddings_indices)
        node._model.get_meta_params.return_value = meta_params
    vert_fed_gov._server._model.predict.return_value = \
        np.random.randint(0, 10, 100)

    vert_fed_gov._server.evaluate_collaborative_model()

    for node, client_data in zip(federated_data, test_data):
        node._model.predict.assert_called_once()
        np.testing.assert_array_equal(
            node._model.predict.call_args[0][0], client_data)

    vert_fed_gov._server._model.evaluate.assert_called_once()
    embeddings = [node._model.predict.return_value for node in federated_data]
    np.testing.assert_array_equal(
        vert_fed_gov._server._model.evaluate.call_args[0][0],
        np.sum(embeddings, axis=0))


def test_evaluate_collaborative_model_test_data(
        test_vertical_federated_government,
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

    vert_fed_gov._server._model.evaluate.assert_called_once()
    embeddings = [node._model.predict.return_value for node in federated_data]
    np.testing.assert_array_equal(
        vert_fed_gov._server._model.evaluate.call_args[0][0],
        np.sum(embeddings, axis=0))


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
         call(meta_params=vert_fed_gov._server.query_model.return_value)]
    test_evaluate_collaborative_model_calls = \
        [call(), call(test_data, test_labels)]
    vert_fed_gov._federated_data.train_model.assert_has_calls(train_model_calls)
    vert_fed_gov._server.aggregate_weights.assert_called_once()
    vert_fed_gov._server.query_model.assert_called_once()
    vert_fed_gov._server.evaluate_collaborative_model.assert_has_calls(
        test_evaluate_collaborative_model_calls)

