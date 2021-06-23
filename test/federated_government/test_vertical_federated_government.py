from unittest.mock import Mock, patch
from unittest.mock import call
import pytest
import numpy as np

from shfl.federated_government.vertical_federated_government import VerticalFederatedGovernment
from shfl.private.federated_operation import federate_list
from shfl.data_base.data_base import vertical_split


@pytest.fixture(name="vertically_split_database")
def fixture_vertically_split_database(global_vars):
    """Returns a vertically split random data base."""
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


@pytest.fixture(name="node_models")
def fixture_node_models(vertically_split_database):
    """Returns the set of models to be assigned to client nodes."""
    federated_data, _, _ = vertically_split_database
    models = [Mock() for _ in range(federated_data.num_nodes())]

    return models


@pytest.fixture(name="federated_data")
def fixture_federated_data(vertically_split_database):
    """Returns the set of federated client nodes."""
    federated_data, _, _ = vertically_split_database

    return federated_data


@patch("shfl.private.federated_operation.VerticalServerDataNode")
def test_initialization(server_node, federated_data, node_models, helpers):
    """Checks that the vertical federated government is correctly initialized."""
    federated_government = VerticalFederatedGovernment(
        models=node_models,
        federated_data=federated_data,
        server_node=server_node)

    helpers.check_initialization(federated_government)


@patch("shfl.private.federated_operation.VerticalServerDataNode")
@patch("shfl.private.federated_operation.NodesFederation")
def test_run_rounds(federated_data,
                    server_node,
                    node_models,
                    vertically_split_database,
                    global_vars):
    """Checks that the federated round is called correctly."""

    embeddings_indices = np.random.choice(a=global_vars["num_data"],
                                          size=global_vars["batch_size"],
                                          replace=False)
    aggregated_embeddings = np.random.rand(global_vars["batch_size"],
                                           global_vars["n_embeddings"])

    embeddings_grads = np.random.rand(global_vars["batch_size"],
                                      global_vars["n_embeddings"])

    server_meta_params = (embeddings_grads, embeddings_indices)

    server_node.aggregate_weights.return_value = (aggregated_embeddings, embeddings_indices)
    server_node.query_model.return_value = server_meta_params

    federated_government = VerticalFederatedGovernment(
        models=node_models,
        federated_data=federated_data,
        server_node=server_node)

    _, test_data, test_labels = vertically_split_database
    federated_government.run_rounds(n_rounds=1,
                                    test_data=test_data,
                                    test_label=test_labels)

    train_model_calls = [call(), call(meta_params=server_meta_params)]
    test_evaluate_collaborative_model_calls = [call(), call(test_data, test_labels)]

    federated_data.train_model.assert_has_calls(train_model_calls)
    server_node.aggregate_weights.assert_called_once()
    server_node.train_model.assert_called_once()
    server_node.query_model.assert_called_once()
    server_node.evaluate_collaborative_model.assert_has_calls(
        test_evaluate_collaborative_model_calls)
