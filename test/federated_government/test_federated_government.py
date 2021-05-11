import numpy as np
from unittest.mock import Mock

from shfl.federated_government.federated_government import FederatedGovernment
from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution_iid import IidDataDistribution
from shfl.private.data import UnprotectedAccess
from shfl.private.federated_operation import ServerDataNode


class FederatedGovernmentTest(FederatedGovernment):
    def __init__(self, model, federated_data, aggregator, server_node=None):
        super(FederatedGovernmentTest, self).__init__(model,
                                                      federated_data,
                                                      aggregator,
                                                      server_node)

    def run_rounds(self, n_rounds, test_data, test_label, eval_freq=1):
        pass


class DataBaseTest(DataBase):
    def __init__(self):
        super(DataBaseTest, self).__init__()

    def load_data(self):
        self._train_data = np.random.rand(200).reshape([40, 5])
        self._test_data = np.random.rand(200).reshape([40, 5])
        self._train_labels = np.random.randint(0, 10, 40)
        self._test_labels = np.random.randint(0, 10, 40)


def test_evaluate_collaborative_model():
    model_builder = Mock()
    aggregator = Mock()
    database = DataBaseTest()
    database.load_data()
    data_distribution = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = \
        data_distribution.get_federated_data(num_nodes=num_nodes)

    federated_government = FederatedGovernment(model_builder, federated_data, aggregator)
    federated_government._server._model.evaluate.return_value = np.random.randint(0, 10, 40)

    federated_government._server.evaluate_collaborative_model(test_data, test_labels)
    federated_government._server._model.evaluate.assert_called_once_with(test_data, test_labels)


def test_evaluate_collaborative_model_local_test():
    model_builder = Mock()
    aggregator = Mock()
    database = DataBaseTest()
    database.load_data()
    data_distribution = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = \
        data_distribution.get_federated_data(num_nodes=num_nodes)

    federated_government = FederatedGovernment(model_builder, federated_data, aggregator)
    federated_government._server.evaluate = Mock()
    federated_government._server.evaluate.return_value = [np.random.randint(0, 10, 40),
                                                          np.random.randint(0, 10, 30)]

    federated_government._server.evaluate_collaborative_model(test_data, test_labels)
    federated_government._server.evaluate.assert_called_once_with(test_data, test_labels)


copy_mock = Mock()


def test_deploy_central_model():
    model_builder = Mock()
    aggregator = Mock()
    database = DataBaseTest()
    database.load_data()
    data_distribution = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = \
        data_distribution.get_federated_data(num_nodes=num_nodes)

    federated_government = FederatedGovernment(model_builder, federated_data, aggregator)
    array_params = np.random.rand(30)
    federated_government._server._model.get_model_params.return_value = array_params

    federated_government._server.deploy_collaborative_model()

    for node in federated_government._federated_data:
        node._model.set_model_params.assert_called_once()


def test_evaluate_clients_global():
    model_builder = Mock()
    aggregator = Mock()
    database = DataBaseTest()
    database.load_data()
    data_distribution = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = \
        data_distribution.get_federated_data(num_nodes=num_nodes)

    federated_government = FederatedGovernment(model_builder, federated_data, aggregator)

    for node in federated_government._federated_data:
        node.evaluate = Mock()
        node.evaluate.return_value = [np.random.randint(0, 10, 40), None]

    federated_government.evaluate_clients(test_data, test_labels)

    for node in federated_government._federated_data:
        node.evaluate.assert_called_once_with(test_data, test_labels)


def test_evaluate_clients_local():
    model_builder = Mock()
    aggregator = Mock()
    database = DataBaseTest()
    database.load_data()
    data_distribution = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = \
        data_distribution.get_federated_data(num_nodes=num_nodes)

    federated_government = FederatedGovernment(model_builder, federated_data, aggregator)

    for node in federated_government._federated_data:
        node.evaluate = Mock()
        node.evaluate.return_value = [np.random.randint(0, 10, 40),
                                      np.random.randint(0, 10, 40)]

    federated_government.evaluate_clients(test_data, test_labels)

    for node in federated_government._federated_data:
        node.evaluate.assert_called_once_with(test_data, test_labels)


def test_train_all_clients():
    model_builder = Mock()
    aggregator = Mock()
    database = DataBaseTest()
    database.load_data()
    data_distribution = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = \
        data_distribution.get_federated_data(num_nodes=num_nodes)

    federated_government = FederatedGovernment(model_builder, federated_data, aggregator)

    federated_government._federated_data.train_model()

    federated_government._federated_data.configure_data_access(UnprotectedAccess())
    for node in federated_government._federated_data:
        labeled_data = node.query()
        node._model.train.assert_called_once_with(
            labeled_data.data, labeled_data.label)


def test_aggregate_weights():
    model_builder = Mock()
    aggregator = Mock()
    database = DataBaseTest()
    database.load_data()
    data_distribution = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = \
        data_distribution.get_federated_data(num_nodes=num_nodes)

    federated_government = FederatedGovernment(model_builder, federated_data, aggregator)

    weights = np.random.rand(64, 32)
    federated_government._server._aggregator.aggregate_weights.return_value = weights

    federated_government._server.aggregate_weights()

    federated_government._server._model.set_model_params.assert_called_once_with(weights)


def test_federated_government_private_data():
    model_builder = Mock()
    aggregator = Mock()
    database = DataBaseTest()
    database.load_data()
    data_distribution = IidDataDistribution(database)
    federated_data, test_data, test_labels = data_distribution.get_federated_data(num_nodes=3)
    federated_data.configure_data_access(UnprotectedAccess())

    la = FederatedGovernmentTest(model_builder, federated_data, aggregator)

    for node in la._federated_data:
        assert isinstance(node._model, type(model_builder))

    assert isinstance(la._server._model, type(model_builder))
    assert aggregator.id == la._server._aggregator.id


def test_federated_government_server():
    model_builder = Mock()
    aggregator = Mock()
    database = DataBaseTest()
    database.load_data()
    data_distribution = IidDataDistribution(database)
    federated_data, test_data, test_labels = data_distribution.get_federated_data(num_nodes=3)
    server = ServerDataNode(federated_data, model_builder, aggregator)

    federated_government = FederatedGovernmentTest(model_builder, federated_data,
                                                   aggregator=None, server_node=server)

    assert isinstance(federated_government._server._model, type(model_builder))
    assert isinstance(federated_government._server._federated_data, type(federated_data))
    assert aggregator.id == federated_government._server._aggregator.id
    assert id(federated_data) == id(federated_government._server._federated_data)


def test_run_rounds():
    model_builder = Mock()
    aggregator = Mock()
    database = DataBaseTest()
    database.load_data()
    data_distribution = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = \
        data_distribution.get_federated_data(num_nodes=num_nodes)

    federated_government = FederatedGovernment(model_builder, federated_data, aggregator)

    federated_government._server.deploy_collaborative_model = Mock()
    federated_government._federated_data.train_model = Mock()
    federated_government.evaluate_clients = Mock()
    federated_government._server.aggregate_weights = Mock()
    federated_government._server.evaluate_collaborative_model = Mock()

    federated_government.run_rounds(1, test_data, test_labels)

    federated_government._server.deploy_collaborative_model.assert_called_once()
    federated_government._federated_data.train_model.assert_called_once()
    federated_government.evaluate_clients.assert_called_once_with(test_data, test_labels)
    federated_government._server.aggregate_weights.assert_called_once()
    federated_government._server.evaluate_collaborative_model.assert_called_once_with(
        test_data, test_labels)


def test_run_rounds_local_tests():
    model_builder = Mock()
    aggregator = Mock()
    database = DataBaseTest()
    database.load_data()
    data_distribution = IidDataDistribution(database)

    num_nodes = 3
    federated_data, test_data, test_labels = \
        data_distribution.get_federated_data(num_nodes=num_nodes)

    federated_data.split_train_test()

    federated_government = FederatedGovernment(model_builder, federated_data, aggregator)

    federated_government._server.deploy_collaborative_model = Mock()
    federated_government._federated_data.train_model = Mock()
    federated_government.evaluate_clients = Mock()
    federated_government._server.aggregate_weights = Mock()
    federated_government._server.evaluate_collaborative_model = Mock()

    federated_government.run_rounds(1, test_data, test_labels)

    federated_government._server.deploy_collaborative_model.assert_called_once()
    federated_government._federated_data.train_model.assert_called_once()
    federated_government.evaluate_clients.assert_called_once_with(test_data, test_labels)
    federated_government._server.aggregate_weights.assert_called_once()
    federated_government._server.evaluate_collaborative_model.assert_called_once_with(
        test_data, test_labels)
