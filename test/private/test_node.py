import unittest.mock
from unittest.mock import Mock
import pytest
import numpy as np

from shfl.private.node import DataNode
from shfl.private.data import LabeledData
from shfl.private.utils import unprotected_query


def test_private_data(data_and_labels):
    """Checks that the private data is not returned."""
    data_node = DataNode()
    data_node.set_private_data("random_array", data_and_labels)

    assert data_node.private_data is None


def test_private_test_data(data_and_labels):
    """Checks that the private data test is not returned."""
    data_node = DataNode()
    data_node.set_private_test_data("random_array_test", data_and_labels)

    assert data_node.private_test_data is None


def test_query_private_data(data_and_labels):
    """Checks that the node correctly returns the private data if the data access
    is set to unprotected."""
    data_node = DataNode()
    data_node.set_private_data("random_data", data_and_labels)
    data_node.configure_data_access("random_data", unprotected_query)

    output_data = data_node.query("random_data")

    np.testing.assert_array_equal(output_data[0], data_and_labels[0])
    np.testing.assert_array_equal(output_data[1], data_and_labels[1])


def test_query_model_params():
    """Checks that the node correctly returns the model's parameters.

    The access to the model's parameters is unprotected by default."""
    random_params = np.random.rand(30, 20)
    data_node = DataNode()
    model_mock = Mock()
    model_mock.get_model_params.return_value = random_params
    data_node.set_model(model_mock)

    model_params = data_node.query_model_params()

    np.testing.assert_array_equal(model_params, random_params)


def test_query_model():
    """Checks that the node correctly queries its model."""
    data_node = DataNode()
    model_mock = Mock()
    data_node.set_model(model_mock)
    data_node.configure_model_access(unprotected_query)

    model = data_node.query_model()

    assert isinstance(model, type(model_mock))


def test_query_model_access_not_configured():
    """Checks that the node raises an error if the model is queried without first
    configuring the access."""
    data_node = DataNode()
    model_mock = Mock()
    data_node.set_model(model_mock)
    with pytest.raises(ValueError):
        data_node.query_model()


def test_train_model_wrong_data(data_and_labels):
    """Checks that the node raises an error if wrong input is used for the model.

    The private input data must have "data" and "labels" attributes."""
    labeled_data = LabeledData(*data_and_labels)
    delattr(labeled_data, "_label")
    data_node = DataNode()
    model_mock = Mock()
    data_node.set_model(model_mock)
    data_node.set_private_data("invalid_data", labeled_data)
    with pytest.raises(ValueError):
        data_node.train_model("invalid_data")


copy_mock = Mock()


@unittest.mock.patch("copy.deepcopy", unittest.mock.MagicMock(return_value=copy_mock))
def test_train_model_data(data_and_labels):
    """Checks that the node trains correctly its local model.

    The model of each client is deep-copied. Thus the original model
    instance must not be called."""
    labeled_data = LabeledData(*data_and_labels)
    data_node = DataNode()
    model_mock = Mock()
    data_node.set_model(model_mock)
    data_node.set_private_data("random_data", labeled_data)

    data_node.train_model("random_data")

    model_mock.train.assert_not_called()
    copy_mock.train.assert_called_once()


def test_get_model():
    """Checks that the node's model is not returned."""
    model_mock = Mock()
    data_node = DataNode()
    data_node.set_model(model_mock)

    assert data_node.model is None


@unittest.mock.patch("copy.deepcopy", unittest.mock.MagicMock(return_value=copy_mock))
def test_predict(data_and_labels):
    """Checks that the node correctly predicts on input data using its local model.

    The model of each client is deep-copied. Thus the original model
    instance must not be called."""
    model_mock = Mock()
    data_node = DataNode()
    data_node.set_model(model_mock)

    data_node.predict(data_and_labels[0])

    model_mock.predict.assert_not_called()
    copy_mock.predict.assert_called_once_with(data_and_labels[0])


@unittest.mock.patch("copy.deepcopy", unittest.mock.MagicMock(return_value=copy_mock))
def test_set_params():
    """Checks that the node correctly sets the model's parameters.

    The model of each client is deep-copied. Thus the original model
    instance must not be called."""
    random_array = np.random.rand(30)
    model_mock = Mock()
    data_node = DataNode()
    data_node.set_model(model_mock)

    data_node.set_model_params(random_array)

    model_mock.set_model_params.assert_not_called()
    copy_mock.set_model_params.assert_called_once_with(copy_mock)


@unittest.mock.patch("copy.deepcopy", unittest.mock.MagicMock(return_value=copy_mock))
def test_evaluate(data_and_labels):
    """Checks that the node correctly evaluates the local model.

    The model of each client is deep-copied. Thus the original model
    instance must not be called."""
    model_mock = Mock()
    data_node = DataNode()
    data_node.set_model(model_mock)

    data_node.evaluate(*data_and_labels)

    model_mock.evaluate.assert_not_called()
    copy_mock.evaluate.assert_called_once_with(*data_and_labels)


@unittest.mock.patch("copy.deepcopy", unittest.mock.MagicMock(return_value=copy_mock))
def test_local_evaluate(data_and_labels):
    """Checks that the node correctly evaluates the model on the local test data."""
    data_node = DataNode()
    data_key = 'EMNIST'
    data_node.set_private_test_data(data_key, LabeledData(*data_and_labels))

    model_mock = Mock()
    data_node.set_model(model_mock)
    copy_mock.evaluate.return_value = 0.8

    evaluation = data_node.local_evaluate(data_key)

    assert evaluation == 0.8


def test_local_evaluate_wrong():
    """Checks that the local evaluation is None when local test data are not present."""
    data_node = DataNode()
    data_node.self_private_test_data = 0

    evaluation = data_node.local_evaluate('some_non_existent_id')

    assert evaluation is None


@unittest.mock.patch("copy.deepcopy", unittest.mock.MagicMock(return_value=copy_mock))
def test_performance(data_and_labels):
    """Checks that the node correctly calls the model's performance."""
    data_node = DataNode()
    model_mock = Mock()
    data_node.set_model(model_mock)
    copy_mock.performance.return_value = 0.8

    performance = data_node.performance(*data_and_labels)

    copy_mock.performance.assert_called_once_with(*data_and_labels)
    assert performance == 0.8
