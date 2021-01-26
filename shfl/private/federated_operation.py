import abc
from shfl.private.node import DataNode
from shfl.private.data import LabeledData


class FederatedDataNode(DataNode):
    """
    This class represents a [DataNode](../data_node) in a FederatedData. Extends DataNode allowing
    calls to methods without explicit private data identifier, assuming access to the federated data.

    It supports Adaptive Differential Privacy through Privacy Filters

    # Arguments:
        federated_data_identifier: identifier to use in private data

    When you iterate over [FederatedData](./#federateddata-class) the kind of DataNode that you obtain is a \
    FederatedDataNode.

    # Example:

    ```python
        # Definition of federated data from dataset
        database = shfl.data_base.Emnist()
        iid_distribution = shfl.data_distribution.IidDataDistribution(database)
        federated_data, test_data, test_labels = iid_distribution.get_federated_data(num_nodes=20, percent=10)

        # Data access definition and query node 0
        federated_data.configure_data_access(UnprotectedAccess())
        federated_data[0].query()
    ```
    """
    def __init__(self, federated_data_identifier):
        super().__init__()
        self._federated_data_identifier = federated_data_identifier

    def query(self, private_property=None, **kwargs):
        """
        Queries private data previously configured. If the access didn't configured this method will raise exception

        # Arguments:
            private_property: String with the key identifier for the data
        """
        if private_property is None:
            private_property = self._federated_data_identifier
        return super().query(private_property, **kwargs)

    def configure_data_access(self, data_access_definition):
        """
        Adds a DataAccessDefinition for some concrete private data.

        # Arguments:
            data_access_definition: Policy to access data (see: [DataAccessDefinition](../data/#dataaccessdefinition-class))
        """
        super().configure_data_access(self._federated_data_identifier, data_access_definition)

    def set_private_data(self, data):
        """
        Creates copy of data in private memory using name as key. If there is a previous value with this key the
        data will be overridden.

        # Arguments:
            data: Data to be stored in the private memory of the DataNode
        """
        super().set_private_data(self._federated_data_identifier, data)

    def set_private_test_data(self, data):
        """
        Creates copy of test data in private memory using name as key. If there is a previous value with this key the
        data will be override.

        # Arguments:
            data: Data to be stored in the private memory of the DataNode
        """
        super().set_private_test_data(self._federated_data_identifier, data)

    def train_model(self, **kwargs):
        """
        Train the model that has been previously set in the data node
        """
        super().train_model(self._federated_data_identifier, **kwargs)

    def apply_data_transformation(self, federated_transformation):
        """
        Executes FederatedTransformation (see: [Federated Operation](../federated_operation)) over private data.

        # Arguments:
            federated_transformation: Operation to execute (see: [Federated Operation](../federated_operation))
        """
        super().apply_data_transformation(self._federated_data_identifier, federated_transformation)

    def evaluate(self, data, test):
        """
        Evaluates the performance of the model

        # Arguments:
            data: Data to predict
            test: True values of data

        # Returns:
            metrics: array with metrics values for predictions for data argument.
        """
        return super().evaluate(data, test), super().local_evaluate(self._federated_data_identifier)

    def split_train_test(self, test_split=0.2):
        """
        Splits private_data in train and test sets

        # Arguments:
            test_split: percentage of test split
        """
        labeled_data = self._private_data.get(self._federated_data_identifier)
        length = len(labeled_data.data)
        train_data = labeled_data.data[int(test_split * length):]
        train_label = labeled_data.label[int(test_split * length):]
        test_data = labeled_data.data[:int(test_split * length)]
        test_label = labeled_data.label[:int(test_split * length)]

        self.set_private_data(LabeledData(train_data, train_label))
        self.set_private_test_data(LabeledData(test_data, test_label))


class ServerDataNode(FederatedDataNode):
    """
        This class represents a type Server [DataNode](../data_node) in a FederatedData.
        Extends DataNode allowing calls to methods without explicit private data identifier,
        assuming access to the Server's data (if any).

        It supports Adaptive Differential Privacy through Privacy Filters

        # Arguments:
            federated_data: the set of client nodes
            model: python object representing the model of the server node
            aggregator: python object representing the type of aggregator to use
            data: optional, server's private data
        """

    def __init__(self, federated_data, model, aggregator, data=None):
        super().__init__(federated_data_identifier=str(id(federated_data)))
        self._federated_data = federated_data
        self.model = model
        self._aggregator = aggregator
        self.set_private_data(data)

    def deploy_collaborative_model(self):
        """
        Deployment of the collaborative learning model from server node to
        each client node.
        """
        for data_node in self._federated_data:
            data_node.set_model_params(self.query_model_params())

    def evaluate_collaborative_model(self, data_test, label_test):
        """
        Evaluation of the performance of the collaborative model.

        # Arguments:
            test_data: test dataset
            test_label: corresponding labels to test dataset
        """
        evaluation, local_evaluation = \
            self.evaluate(data_test, label_test)

        print("Collaborative model test performance : " + str(evaluation))
        if local_evaluation is not None:
            print("Collaborative model server local test performance : "
                  + str(local_evaluation))

    def aggregate_weights(self):
        """
        Aggregate weights from all data nodes in the server model and
        updates the server
        """
        weights = [data_node.query_model_params()
                   for data_node in self._federated_data]
        aggregated_weights = self._aggregator.aggregate_weights(weights)
        self._model.set_model_params(aggregated_weights)


class FederatedData:
    """
    Class representing data across different data nodes.

    This object is iterable over different data nodes.
    """

    def __init__(self):
        self._data_nodes = []
        node_methods_list = [func for func in dir(FederatedDataNode)
                             if callable(getattr(FederatedDataNode, func))
                             and not func.startswith("__")]
        for method in node_methods_list:
            setattr(self, method, self._create_apply_method(method))

    def __getitem__(self, item):
        return self._data_nodes[item]

    def __iter__(self):
        return iter(self._data_nodes)

    def add_data_node(self, data):
        """
        This method adds a new node containing data to the federated data

        # Arguments:
            data: Data to add to this node
        """
        node = FederatedDataNode(str(id(self)))
        node.set_private_data(data)
        self._data_nodes.append(node)

    def num_nodes(self):
        """
        # Returns:
            num_nodes: The number of nodes in this federated data.
        """
        return len(self._data_nodes)

    def _create_apply_method(self, method):
        """
        Create a function that loops on all the FederatedData nodes and
        calls a node's method.

        # Arguments:
            method: string corresponding to a node's method

        # Returns:
            node_method: function that loops the method on all the nodes
        """

        def apply_method(*args, **kwargs):
            """
            Apply a method on the FederatedData nodes.

            # Returns:
                output: List containing responses for every node (if any)
            """
            output = [getattr(data_node, method)(*args, **kwargs)
                      for data_node in self._data_nodes]
            return output

        return apply_method


class FederatedTransformation(abc.ABC):
    """
    Interface defining the method for applying an operation over [FederatedData](./#federateddata-class)
    """
    @abc.abstractmethod
    def apply(self, data):
        """
        This method receives data to be modified and performs the required modifications over it.

        # Arguments:
            data: The object that has to be modified
        """


def federate_array(array, num_data_nodes):
    """
    Creates [FederatedData](./#federateddata-class) from an indexable array.

    The array will be divided using the first dimension.

    It supports Adaptive Differential Privacy through Privacy Filters

    # Arguments:
        array: Indexable array with any number of dimensions
        num_data_nodes: Number of nodes to use

    # Returns:
        federated_array: [FederatedData](./#federateddata-class) with an array of size len(array)/num_data_nodes \
        in every node
    """
    split_size = len(array) / float(num_data_nodes)
    last = 0.0
    federated_array = FederatedData()
    while last < len(array):
        print("first and last ", int(last), int(last + split_size))
        federated_array.add_data_node(array[int(last):int(last + split_size)])
        last = last + split_size

    return federated_array


def apply_federated_transformation(federated_data, federated_transformation):
    """
    Applies the federated transformation over this federated data.

    Original federated data will be modified.

    # Arguments:
        federated_data: [FederatedData](./#federateddata-class) to use in the transformation
        federated_transformation: [FederatedTransformation](./#federatedtransformation-class) that will be applied \
        over this data
    """
    for data_node in federated_data:
        data_node.apply_data_transformation(federated_transformation)


def split_train_test(federated_data, test_split=0.2):
    """
    Splits all data nodes in train and test sets

    # Arguments:
        federated_data: [FederatedData](./#federateddata-class)
        test_split: percentage of test split
    """
    for data_node in federated_data:
        data_node.split_train_test(test_split)


class Normalize(FederatedTransformation):
    """
    Normalization class of federated data [FederatedData](./#federateddata-class). It implements \
    [FederatedTransformation](./#federatedtransformation-class).

    # Arguments:
        mean: mean used for normalization.
        std: std used for normalization.
    """
    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def apply(self, labeled_data):
        labeled_data.data = (labeled_data.data - self.__mean) / self.__std
