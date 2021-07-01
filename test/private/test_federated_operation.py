from unittest.mock import Mock, patch
import pytest
import numpy as np

import shfl.private.federated_operation
from shfl.private.federated_operation import NodesFederation
from shfl.private.federated_operation import FederatedDataNode
from shfl.private.federated_operation import ServerDataNode
from shfl.private.federated_operation import VerticalServerDataNode
from shfl.private.federated_operation import federate_array
from shfl.private.federated_operation import federate_list
from shfl.private.data import LabeledData
from shfl.private.utils import unprotected_query


@pytest.fixture(name="federated_array")
def fixture_federated_array():
    """Returns a set of federated data nodes using an input random vector."""
    num_data = 30
    input_array = np.random.rand(num_data)
    nodes_federation = federate_array(input_array, num_data)
    nodes_federation.configure_data_access(unprotected_query)

    return nodes_federation, input_array


@pytest.fixture(name="nodes_federation_single_node")
def fixture_nodes_federation_single_node():
    """Returns a set of federated data nodes with only one node."""
    data_size = 10
    nodes_federation = NodesFederation()
    input_data = np.random.rand(data_size)
    nodes_federation.append_data_node(input_data)
    nodes_federation.configure_data_access(unprotected_query)

    return nodes_federation, input_data


@pytest.fixture(name="nodes_federation_multiple_nodes")
def fixture_nodes_federation():
    """Returns a set of federated data nodes using random data and labels as input."""
    num_nodes = 15
    data_shape = (10, 5)
    input_data = np.random.rand(num_nodes, *data_shape)
    input_labels = np.random.randint(low=0, high=2, size=(num_nodes, data_shape[0]))
    nodes_federation = NodesFederation()
    for data, labels in zip(input_data, input_labels):
        nodes_federation.append_data_node(LabeledData(data, labels))
    nodes_federation.configure_data_access(unprotected_query)

    return nodes_federation, input_data, input_labels


@pytest.fixture(name="horizontal_server_node")
def fixture_horizontal_server_node():
    """Returns a mocked horizontal server node."""
    nodes_federation = Mock()
    model = Mock()
    aggregator = Mock()
    server_node = ServerDataNode(nodes_federation, model, aggregator)
    server_node.configure_data_access(unprotected_query)

    return server_node, nodes_federation, model, aggregator


@pytest.fixture(name="vertical_server_node")
def fixture_vertical_server_node():
    """Returns a mocked vertical server node."""
    num_nodes = 5
    nodes_federation = [Mock() for _ in range(num_nodes)]
    clients_embeddings = []
    for node in nodes_federation:
        node_embeddings = np.random.rand(5, 9)
        node.predict.return_value = node_embeddings
        clients_embeddings.append(node_embeddings)
    model = Mock()
    aggregator = Mock()
    server_node = VerticalServerDataNode(nodes_federation, model, aggregator)
    server_node.configure_data_access(unprotected_query)

    return server_node, nodes_federation, model, aggregator, clients_embeddings


def normalize_query(data, mean, std):
    """Applies a normalization over the input data.
    # Arguments:
        data: Input data.
        mean: Mean used for the normalization.
        std: Standard deviation used for the normalization.
    """
    data.data = (data.data - mean) / std


def test_nodes_federation_initialization():
    """Checks that the set of federated nodes is correctly initialized.
    """
    nodes_federation = NodesFederation()

    assert nodes_federation.num_nodes() == 0


def test_nodes_federation_append_node(nodes_federation_single_node):
    """Checks that a data node is correctly appended to the federated set."""
    nodes_federation, input_data = nodes_federation_single_node

    assert nodes_federation.num_nodes() == 1
    np.testing.assert_array_equal(nodes_federation[0].query(), input_data)


def test_federated_wrong_data_identifier(nodes_federation_single_node):
    """Checks that the an error is raised when a wrong data identifier is used in
    a query to a node."""
    nodes_federation = nodes_federation_single_node[0]

    with pytest.raises(ValueError):
        nodes_federation[0].query("bad_identifier_nodes_federation")


def test_split_train_test(nodes_federation_multiple_nodes):
    """Checks that the set of federated nodes correctly splits private data
    into train and test."""
    nodes_federation, input_data, input_labels = nodes_federation_multiple_nodes
    train_proportion = 0.8

    nodes_federation.split_train_test(train_proportion=train_proportion)

    for i, data_node in enumerate(nodes_federation):
        train_size = round(len(input_data[i]) * train_proportion)
        np.testing.assert_array_equal(data_node.query().data, input_data[i][:train_size])
        np.testing.assert_array_equal(data_node.query().label, input_labels[i][:train_size])


@patch("shfl.private.federated_operation.DataNode.evaluate")
@patch("shfl.private.federated_operation.DataNode.local_evaluate")
def test_evaluate(mock_super_local_evaluate, mock_super_evaluate):
    """Checks that the set of federated data nodes evaluates correctly."""
    data = np.random.rand(15)
    test = np.random.rand(5)

    identifier = 'id'
    nodes_federation = FederatedDataNode(identifier)

    mock_super_evaluate.return_value = 10
    mock_super_local_evaluate.return_value = 15

    evaluate, local_evaluate = nodes_federation.evaluate(data, test)

    assert evaluate == 10
    assert local_evaluate == 15
    mock_super_evaluate.assert_called_once_with(data, test)
    mock_super_local_evaluate.assert_called_once_with(identifier)


@patch("shfl.private.federated_operation.DataNode.train_model")
def test_train_model(mock_super_train_model):
    """Checks that the set of federated data nodes evaluates correctly."""
    identifier = 'id'
    nodes_federation = FederatedDataNode(identifier)
    mock_super_train_model.return_value = None

    nodes_federation.train_model(optional_train_params="train_params")

    mock_super_train_model.assert_called_once_with(identifier,
                                                   optional_train_params="train_params")


@pytest.mark.parametrize("server_type", ["horizontal_server_node", "vertical_server_node"])
def test_server_initialization(server_type, request):
    """Checks that the horizontal server data node is properly initialized."""
    server_node = request.getfixturevalue(server_type)[0]

    assert hasattr(server_node, "_nodes_federation")
    assert hasattr(server_node, "_model")
    assert hasattr(server_node, "_aggregator")
    assert hasattr(server_node, "_nodes_federation_identifier")


def test_horizontal_server_deploy_collaborative_model(horizontal_server_node):
    """Checks that the horizontal server deploy correctly the collaborative model
    among the associated set of federated nodes."""
    server_node, nodes_federation, _, _ = horizontal_server_node
    server_node.query_model_params = Mock()
    params = np.random.rand(10, 20)
    server_node.query_model_params.return_value = params

    server_node.deploy_collaborative_model()

    nodes_federation.set_model_params.assert_called_once_with(params)


def test_horizontal_server_evaluate_collaborative_model(horizontal_server_node):
    """Checks that the horizontal server performs global and local evaluation correctly."""
    server_node = horizontal_server_node[0]
    server_node.evaluate = Mock()
    server_node.evaluate.return_value = (np.random.rand(4), np.random.rand(4))
    global_data, global_labels = (np.random.rand(100, 9),
                                  np.random.randint(low=0, high=2, size=100))

    server_node.evaluate_collaborative_model(data=global_data, labels=global_labels)

    server_node.evaluate.assert_called_once_with(global_data, global_labels)


def test_horizontal_server_aggregate_weights(horizontal_server_node):
    """Checks that the horizontal server model correctly aggregates the model parameters
    from the associated set of federated nodes."""
    server_node, nodes_federation, _, aggregator = horizontal_server_node
    nodes_federation.query_model_params = Mock()
    clients_params = np.random.rand(10, 20)
    nodes_federation.query_model_params.return_value = clients_params
    aggregated_weights = np.random.rand(5, 20)
    aggregator.return_value = aggregated_weights

    server_node.aggregate_weights()

    nodes_federation.query_model_params.assert_called_once()
    aggregator.assert_called_once_with(clients_params)


def test_vertical_server_predict_collaborative_model(vertical_server_node):
    """Checks that the vertical server node orchestrates correctly the prediction
    using the distributed model."""
    server_node, nodes_federation, _, aggregator, clients_embeddings = vertical_server_node
    server_node.predict_clients = Mock()
    server_node.predict_clients.return_value = clients_embeddings
    clients_embeddings_aggregated = np.random.rand(10, 20)
    aggregator.return_value = clients_embeddings_aggregated
    server_output = np.random.randint(low=0, high=2, size=10)
    server_node.predict = Mock()
    server_node.predict.return_value = server_output
    data = [np.random.rand(5, 9) for _ in range(len(nodes_federation))]

    prediction = server_node.predict_collaborative_model(data)

    server_node.predict_clients.assert_called_once_with(data)
    for i, node_embeddings in enumerate(clients_embeddings):
        np.testing.assert_array_equal(aggregator.call_args[0][0][i],
                                      node_embeddings)
    server_node.predict.assert_called_once_with(clients_embeddings_aggregated)
    np.testing.assert_array_equal(prediction, server_output)


def test_vertical_server_predict_clients(vertical_server_node):
    """Checks that the vertical server node calls correctly the prediction
    from the clients' side."""
    server_node, nodes_federation, _, _, clients_embeddings = vertical_server_node
    data = [np.random.rand(3, 4) for _ in range(len(nodes_federation))]

    output_clients_embeddings = server_node.predict_clients(data)

    for node, data_chunk in zip(nodes_federation, data):
        node.predict.assert_called_once_with(data_chunk)
    for true_embeddings, output_embeddings in zip(clients_embeddings,
                                                  output_clients_embeddings):
        np.testing.assert_array_equal(true_embeddings, output_embeddings)


def test_vertical_server_evaluate_collaborative_model(vertical_server_node):
    """Checks that the vertical server correctly orchestrates the evaluation on input data
    using the distributed model."""
    server_node, nodes_federation, _, aggregator, clients_embeddings = vertical_server_node
    data = [np.random.rand(3, 4) for _ in range(len(nodes_federation))]
    server_node.predict_clients = Mock()
    server_node.predict_clients.return_value = clients_embeddings
    clients_embeddings_aggregated = np.random.rand(10, 20)
    aggregator.return_value = clients_embeddings_aggregated
    server_node.evaluate = Mock()

    server_node.evaluate_collaborative_model(data)

    server_node.predict_clients.assert_called_once_with(data)
    for i, node_embeddings in enumerate(clients_embeddings):
        np.testing.assert_array_equal(aggregator.call_args[0][0][i],
                                      node_embeddings)
    server_node.evaluate.assert_called_once_with(clients_embeddings_aggregated, None)


def test_vertical_server_evaluate_local(vertical_server_node):
    """Checks that the vertical server correctly orchestrates the local evaluation.

    When no global data is provided as input, the collaborative model
    is evaluated on the current batch."""
    server_node = vertical_server_node[0]
    server_node.query = Mock()
    server_node.aggregate_weights = Mock()

    server_node.evaluate_collaborative_model()

    server_node.query.assert_called_once()


def test_vertical_server_aggregate_weights():
    """Checks that the vertical server correctly aggregates clients' embeddings."""
    model = Mock()
    aggregator = Mock()
    true_aggregated_meta_params = np.random.rand(10, 2)
    aggregator.return_value = true_aggregated_meta_params
    num_nodes = 5
    samples_indices = np.random.randint(low=0, high=100, size=10)
    clients_meta_params = [(np.random.rand(10, 2), samples_indices)
                           for _ in range(num_nodes)]
    nodes_federation = Mock()
    nodes_federation.query_model.return_value = clients_meta_params
    server_node = VerticalServerDataNode(nodes_federation, model, aggregator)

    output_aggregated_meta_params, output_indices = server_node.aggregate_weights()

    np.testing.assert_array_equal(output_aggregated_meta_params,
                                  true_aggregated_meta_params)
    np.testing.assert_array_equal(output_indices, samples_indices)


def test_vertical_server_wrong_samples_indices():
    """Checks that an error is raised if the samples indices of the mini-batch
    do not match among clients."""
    model = Mock()
    aggregator = Mock()
    num_nodes = 5
    samples_indices = np.random.randint(low=0, high=100, size=10)
    clients_meta_params = [(np.random.rand(10, 2), samples_indices)
                           for _ in range(num_nodes)]
    wrong_samples_indices = np.random.randint(low=0, high=100, size=10)
    clients_meta_params[0] = (np.random.rand(10, 2), wrong_samples_indices)
    nodes_federation = Mock()
    nodes_federation.query_model.return_value = clients_meta_params
    server_node = VerticalServerDataNode(nodes_federation, model, aggregator)

    with pytest.raises(AssertionError):
        server_node.aggregate_weights()


def test_federate_transformation(federated_array):
    """Checks that the federated transformation is correctly applied on a set
    of federated data nodes."""
    def transformation_query(data):
        """Some data transformation."""
        data += 1
    nodes_federation, input_array = federated_array

    nodes_federation.apply_data_transformation(transformation_query)

    for i, data_node in enumerate(nodes_federation):
        assert data_node.query() == input_array[i] + 1


def test_federated_normalization(nodes_federation_multiple_nodes):
    """Checks that the federated normalization performs correctly."""
    nodes_federation, input_data, _ = nodes_federation_multiple_nodes
    standard_deviation = 0.2
    mean = 0.4

    nodes_federation.apply_data_transformation(normalize_query, mean=mean, std=standard_deviation)

    for i, data_node in enumerate(nodes_federation):
        normalized_data = (input_data[i] - mean) / standard_deviation
        np.testing.assert_array_equal(data_node.query().data, normalized_data)


def test_federate_array():
    """Checks that the function to federate an array properly splits an array
    among a set of federated nodes.

    The number of nodes is tested."""
    data_size = 10000
    num_clients = 1000
    array = np.random.rand(data_size)

    nodes_federation = federate_array(array, num_clients)

    assert nodes_federation.num_nodes() == num_clients


def test_federate_array_size_private_data():
    """Checks that the function to federate an array properly splits an array
    among a set of federated nodes.

    The size of the private data is tested."""
    data_size = 10000
    num_clients = 10
    array = np.random.rand(data_size)

    federated_array = federate_array(array, num_clients)
    federated_array.configure_data_access(unprotected_query)

    for data_node in federated_array:
        assert len(data_node.query()) == data_size / num_clients


def test_query_federate_data(federated_array):
    """Checks that the set of federated data correctly answers to a query."""
    nodes_federation, input_array = federated_array

    federated_answer = nodes_federation.query()

    for i, node_answer in enumerate(federated_answer):
        assert node_answer == input_array[i]


def test_federate_list():
    """Checks that a list is properly converted to a set of federated nodes.

    The data is already partitioned, i.e., each entry of the list
    contains the data of a single node."""
    distributed_data = [np.random.rand(50).reshape([10, -1])
                        for _ in range(5)]
    distributed_labels = [np.random.randint(0, 2, size=(10, ))
                          for _ in range(5)]

    nodes_federation = federate_list(distributed_data, distributed_labels)
    nodes_federation.configure_data_access(unprotected_query)

    for i, node in enumerate(nodes_federation):
        np.testing.assert_array_equal(node.query().data, distributed_data[i])
        np.testing.assert_array_equal(node.query().label, distributed_labels[i])


def test_federate_list_no_labels():
    """Checks that a list is properly converted to a set of federated nodes.

    If no labels are provided as input, the "label" field is assigned to None."""
    distributed_data = [np.random.rand(50).reshape([10, -1])
                        for _ in range(5)]

    nodes_federation = shfl.private.federated_operation.federate_list(distributed_data)
    nodes_federation.configure_data_access(unprotected_query)

    for i, node in enumerate(nodes_federation):
        np.testing.assert_array_equal(node.query().data,
                                      distributed_data[i])
        assert node.query().label is None
