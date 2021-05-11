import numpy as np
from unittest.mock import Mock
from unittest.mock import call
import pytest

from shfl.federated_government.vertical_federated_government import VerticalFederatedGovernment
from shfl.private.data import LabeledData
from shfl.private.federated_operation import federate_list
from shfl.private.data import DataAccessDefinition
from shfl.private.data import UnprotectedAccess
from shfl.private.federated_operation import VerticalServerDataNode
from shfl.data_base.data_base import vertical_split


class TrainEvaluation(DataAccessDefinition):
    """Evaluate collaborative model on batch train data."""

    def apply(self, data, **kwargs):
        server_model = kwargs.get("server_model")
        embeddings, embeddings_indices = kwargs.get("meta_params")
        labels = data.label[embeddings_indices]

        evaluation = server_model.evaluate(embeddings, labels)

        return evaluation


class QueryMetaParameters(DataAccessDefinition):
    """Returns embeddings (or their gradients) as computed by the local model."""

    def apply(self, model, **kwargs):
        return model.get_meta_params(**kwargs)


@pytest.fixture
def global_vars():
    global_vars = {"n_features": 23,
                   "n_classes": 2,
                   "n_embeddings": 3,
                   "num_data": 100,
                   "batch_size": 32,
                   "n_nodes": 3,
                   "metrics": [0, 1, 2, 3]}

    return global_vars


@pytest.fixture
def vertically_split_database(global_vars):
    data = np.random.rand(global_vars["num_data"],
                          global_vars["n_features"])
    labels = np.random.randint(low=0,
                               high=global_vars["n_classes"],
                               size=(global_vars["num_data"], 1))

    train_data, train_labels, test_data, test_labels = \
        vertical_split(data, labels, indices_or_sections=global_vars["n_nodes"])

    labels_nodes = [train_labels for _ in range(global_vars["n_nodes"])]
    federated_data = federate_list(train_data, labels_nodes)

    return federated_data, test_data, test_labels


@pytest.fixture
def node_models(vertically_split_database):
    federated_data, _, _ = vertically_split_database
    models = [Mock() for _ in range(federated_data.num_nodes())]

    return models


@pytest.fixture
def federated_data(vertically_split_database):
    federated_data, _, _ = vertically_split_database
    federated_data.configure_model_access(QueryMetaParameters())

    return federated_data


@pytest.fixture
def server_node(federated_data, node_models, global_vars):
    server_data = LabeledData(
        data=np.random.rand(global_vars["num_data"],
                            global_vars["n_features"]),
        label=np.random.randint(low=0,
                                high=global_vars["n_classes"],
                                size=(global_vars["num_data"], 1)))

    server_node = VerticalServerDataNode(
        federated_data=federated_data,
        model=Mock(),
        aggregator=Mock(),
        data=server_data)

    server_node.configure_data_access(TrainEvaluation())
    server_node.configure_model_access(QueryMetaParameters())

    return server_node


@pytest.fixture
def vertical_federated_government(federated_data,
                                  node_models,
                                  server_node):
    vert_fed_gov = VerticalFederatedGovernment(
        models=node_models,
        federated_data=federated_data,
        server_node=server_node)

    return vert_fed_gov


def test_train_all_clients(vertical_federated_government):
    vert_fed_gov = vertical_federated_government

    vert_fed_gov._federated_data.train_model()

    vert_fed_gov._federated_data.configure_data_access(UnprotectedAccess())
    for node in vert_fed_gov._federated_data:
        labeled_data = node.query()
        node._model.train.assert_called_once_with(
            labeled_data.data, labeled_data.label)


def test_aggregate_weights(vertical_federated_government, global_vars):
    vert_fed_gov = vertical_federated_government

    embeddings_indices = np.random.choice(a=global_vars["num_data"],
                                          size=global_vars["batch_size"],
                                          replace=False)
    aggregated_embeddings = np.zeros((global_vars["batch_size"],
                                      global_vars["n_embeddings"]))
    for node in vert_fed_gov._federated_data:
        embeddings = np.random.rand(global_vars["batch_size"],
                                    global_vars["n_embeddings"])
        node._model.get_meta_params.return_value = (embeddings,
                                                    embeddings_indices)
        aggregated_embeddings += embeddings

    vert_fed_gov._server._aggregator.aggregate_weights.return_value = \
        aggregated_embeddings
    aggregated_meta_params, matching_indices = \
        vert_fed_gov._server.aggregate_weights()

    for node in vert_fed_gov._federated_data:
        node._model.get_meta_params.assert_called_once()
    vert_fed_gov._server._aggregator.aggregate_weights.assert_called_once()

    clients_meta_params = [node._model.get_meta_params()
                           for node in vert_fed_gov._federated_data]
    embeddings = [client[param]
                  for client in clients_meta_params
                  for param in range(len(client) - 1)]

    for i_node in range(vert_fed_gov._federated_data.num_nodes()):
        np.testing.assert_array_equal(
            vert_fed_gov._server._aggregator.aggregate_weights.call_args[0][0][i_node],
            embeddings[i_node])
    np.testing.assert_array_equal(aggregated_meta_params, aggregated_embeddings)
    np.testing.assert_array_equal(matching_indices, embeddings_indices)


def test_aggregate_weights_non_matching_clients_indices(
        vertical_federated_government, global_vars):

    vert_fed_gov = vertical_federated_government

    for node in vert_fed_gov._federated_data:
        embeddings_indices = np.random.choice(a=global_vars["num_data"],
                                              size=global_vars["batch_size"],
                                              replace=False)
        embeddings = np.random.rand(global_vars["batch_size"],
                                    global_vars["n_embeddings"])
        node._model.get_meta_params.return_value = (embeddings,
                                                    embeddings_indices)

    with pytest.raises(AssertionError):
        vert_fed_gov._server.aggregate_weights()


def test_server_node_train(vertical_federated_government, global_vars):
    vert_fed_gov = vertical_federated_government

    embeddings_indices = np.random.choice(a=global_vars["num_data"],
                                          size=global_vars["batch_size"],
                                          replace=False)
    aggregated_embeddings = np.random.rand(global_vars["batch_size"],
                                           global_vars["n_embeddings"])

    clients_meta_params = (aggregated_embeddings, embeddings_indices)
    vert_fed_gov._server.train_model(meta_params=clients_meta_params)

    vert_fed_gov._server._model.train.assert_called_once()

    vert_fed_gov._server.configure_data_access(UnprotectedAccess())
    server_labeled_data = vert_fed_gov._server.query()
    np.testing.assert_array_equal(
        vert_fed_gov._server._model.train.call_args[0][0],
        server_labeled_data.data)
    np.testing.assert_array_equal(
        vert_fed_gov._server._model.train.call_args[0][1],
        server_labeled_data.label)

    np.testing.assert_array_equal(
        vert_fed_gov._server._model.train.call_args[1]["meta_params"][0],
        aggregated_embeddings)
    np.testing.assert_array_equal(
        vert_fed_gov._server._model.train.call_args[1]["meta_params"][1],
        embeddings_indices)


def test_train_all_clients_update_stage(vertical_federated_government,
                                        global_vars):
    vert_fed_gov = vertical_federated_government
    embeddings_indices = np.random.choice(a=global_vars["num_data"],
                                          size=global_vars["batch_size"],
                                          replace=False)
    embeddings_grads = np.random.rand(global_vars["batch_size"],
                                      global_vars["n_embeddings"])
    server_meta_params = (embeddings_grads, embeddings_indices)
    vert_fed_gov._federated_data.train_model(meta_params=server_meta_params)

    vert_fed_gov._federated_data.configure_data_access(UnprotectedAccess())
    for node in vert_fed_gov._federated_data:
        labeled_data = node.query()
        node._model.train.assert_called_once()
        np.testing.assert_array_equal(node._model.train.call_args[0][0],
                                      labeled_data.data)
        np.testing.assert_array_equal(node._model.train.call_args[0][1],
                                      labeled_data.label)
        np.testing.assert_array_equal(node._model.train.call_args[1]["meta_params"][0],
                                      embeddings_grads)
        np.testing.assert_array_equal(node._model.train.call_args[1]["meta_params"][1],
                                      embeddings_indices)


def test_evaluate_collaborative_model_on_train_data(
        vertical_federated_government, vertically_split_database, global_vars):

    federated_data, _, _ = vertically_split_database
    vert_fed_gov = vertical_federated_government

    embeddings_indices = np.random.choice(a=global_vars["num_data"],
                                          size=global_vars["batch_size"],
                                          replace=False)
    aggregated_embeddings = np.zeros((global_vars["batch_size"],
                                      global_vars["n_embeddings"]))
    for node in federated_data:
        embeddings = np.random.rand(global_vars["batch_size"],
                                    global_vars["n_embeddings"])
        meta_params = (embeddings, embeddings_indices)
        node._model.get_meta_params.return_value = meta_params
        aggregated_embeddings += embeddings

    vert_fed_gov._server._aggregator.aggregate_weights.return_value = \
        aggregated_embeddings
    vert_fed_gov._server._model.evaluate.return_value = \
        np.random.rand(len(global_vars["metrics"]))

    vert_fed_gov._server.evaluate_collaborative_model()

    vert_fed_gov._server.configure_data_access(UnprotectedAccess())
    server_labeled_data = vert_fed_gov._server.query()

    vert_fed_gov._server._aggregator.aggregate_weights.assert_called_once()
    vert_fed_gov._server._model.evaluate.assert_called_once()
    for node in federated_data:
        node._model.get_meta_params.assert_called_once()

    np.testing.assert_array_equal(
        vert_fed_gov._server._model.evaluate.call_args[0][0],
        aggregated_embeddings)
    np.testing.assert_array_equal(
        vert_fed_gov._server._model.evaluate.call_args[0][1],
        server_labeled_data.label[embeddings_indices])


def test_evaluate_collaborative_model_on_test_data(
        vertical_federated_government, vertically_split_database, global_vars):

    federated_data, test_data, test_labels = vertically_split_database
    vert_fed_gov = vertical_federated_government

    aggregated_embeddings = np.zeros((len(test_data[0]),
                                      global_vars["n_embeddings"]))
    for node in federated_data:
        embeddings = np.random.rand(len(test_data[0]),
                                    global_vars["n_embeddings"])
        node._model.predict.return_value = embeddings
        aggregated_embeddings += embeddings
    vert_fed_gov._server._aggregator.aggregate_weights.return_value = \
        aggregated_embeddings
    vert_fed_gov._server._model.evaluate.return_value = \
        np.random.rand(len(global_vars["metrics"]))

    vert_fed_gov._server.evaluate_collaborative_model(test_data, test_labels)

    for node, client_data in zip(federated_data, test_data):
        node._model.predict.assert_called_once()
        np.testing.assert_array_equal(
            node._model.predict.call_args[0][0], client_data)

    vert_fed_gov._server._aggregator.aggregate_weights.assert_called_once()
    vert_fed_gov._server._model.evaluate.assert_called_once()
    np.testing.assert_array_equal(
        vert_fed_gov._server._model.evaluate.call_args[0][0],
        aggregated_embeddings)
    np.testing.assert_array_equal(
        vert_fed_gov._server._model.evaluate.call_args[0][1],
        test_labels)


def test_predict_collaborative_model(
        vertical_federated_government, vertically_split_database, global_vars):

    federated_data, input_data, _ = vertically_split_database
    vert_fed_gov = vertical_federated_government

    aggregated_embeddings = np.zeros((len(input_data[0]),
                                      global_vars["n_embeddings"]))
    for node in federated_data:
        embeddings = np.random.rand(len(input_data[0]),
                                    global_vars["n_embeddings"])
        node._model.predict.return_value = embeddings
        aggregated_embeddings += embeddings
    vert_fed_gov._server._aggregator.aggregate_weights.return_value = \
        aggregated_embeddings
    vert_fed_gov._server._model.predict.return_value = \
        np.random.rand(len(input_data[0]),
                       global_vars["n_classes"])

    vert_fed_gov._server.predict_collaborative_model(input_data)

    for node, client_data in zip(federated_data, input_data):
        node._model.predict.assert_called_once()
        np.testing.assert_array_equal(
            node._model.predict.call_args[0][0], client_data)

    vert_fed_gov._server._aggregator.aggregate_weights.assert_called_once()
    vert_fed_gov._server._model.predict.assert_called_once()
    np.testing.assert_array_equal(
        vert_fed_gov._server._model.predict.call_args[0][0],
        aggregated_embeddings)


def test_run_rounds(vertical_federated_government,
                    vertically_split_database,
                    global_vars):

    _, test_data, test_labels = vertically_split_database
    vert_fed_gov = vertical_federated_government

    embeddings_indices = np.random.choice(a=global_vars["num_data"],
                                          size=global_vars["batch_size"],
                                          replace=False)
    aggregated_embeddings = np.random.rand(global_vars["batch_size"],
                                           global_vars["n_embeddings"])

    clients_meta_params = (aggregated_embeddings, embeddings_indices)
    embeddings_grads = np.random.rand(global_vars["batch_size"],
                                      global_vars["n_embeddings"])
    server_meta_params = (embeddings_grads, embeddings_indices)

    vert_fed_gov._federated_data.train_model = Mock()
    vert_fed_gov._server.aggregate_weights = Mock()
    vert_fed_gov._server.aggregate_weights.return_value = clients_meta_params
    vert_fed_gov._server.train_model = Mock()
    vert_fed_gov._server.query_model = Mock()
    vert_fed_gov._server.query_model.return_value = server_meta_params
    vert_fed_gov._server.evaluate_collaborative_model = Mock()

    vert_fed_gov.run_rounds(n_rounds=1, test_data=test_data, test_label=test_labels)

    train_model_calls = \
        [call(),
         call(meta_params=server_meta_params)]
    test_evaluate_collaborative_model_calls = \
        [call(), call(test_data, test_labels)]
    vert_fed_gov._federated_data.train_model.assert_has_calls(train_model_calls)
    vert_fed_gov._server.aggregate_weights.assert_called_once()
    vert_fed_gov._server.train_model.assert_called_once()
    vert_fed_gov._server.query_model.assert_called_once()
    vert_fed_gov._server.evaluate_collaborative_model.assert_has_calls(
        test_evaluate_collaborative_model_calls)

