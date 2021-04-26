import abc
import numpy as np
from shfl.private.node import DataNode
from shfl.private.data import LabeledData


class FederatedData:
    """
    Class representing data across different data nodes.
    This object overrides dynamically the callable methods of class
    FederatedDataNode to make them iterable over different data nodes.
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
        """Creates generic apply methods.

        Creates a function that loops on all the federated data nodes and
        calls a node's method.

        # Arguments:
            method: String corresponding to a node's method

        # Returns:
            node_method: Function that loops the desired method on all the nodes
        """

        def apply_method(*args, **kwargs):
            """Applies a method on the FederatedData nodes.

            # Returns:
                output: List containing method's output for every node.
                    If the method does not have an explicit return value,
                    this is a list of None.
            """
            output = [getattr(data_node, method)(*args, **kwargs)
                      for data_node in self._data_nodes]

            return output

        return apply_method


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
        This class represents a type Server [DataNode](../data_node)
        in a FederatedData. It extends DataNode allowing calls to methods
        without explicit private data identifier,
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
        self._federated_data.set_model_params(self.query_model_params())

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

        weights = self._federated_data.query_model_params()
        aggregated_weights = self._aggregator.aggregate_weights(weights)
        self._model.set_model_params(aggregated_weights)


class VerticalServerDataNode(FederatedDataNode):
    """
    This class represents a Server data node [DataNode](../data_node) in
    the Vertical Federated Learning setting.
    It extends DataNode allowing calls to methods without explicit private
    data identifier, assuming access to the Server's data (if any).
    It also Aggregates weights from all data nodes in the server model and
    trains the server's model.

    # Arguments:
        federated_data: Object of class [FederatedData](./federated_data)
            representing the set of client nodes
        model: Object representing the model of the server node
        data: Optional server's private data
    """

    def __init__(self, federated_data, model, aggregator, data=None):
        super().__init__(federated_data_identifier=str(id(federated_data)))
        self._federated_data = federated_data
        self.model = model
        self._aggregator = aggregator
        self.set_private_data(data)

    def predict_collaborative_model(self, data):
        """
        Make a prediction using the collaborative model.

        # Arguments:
            data: List, each item representing the global test
                dataset for a single client.
        """

        clients_embeddings = [node.predict(data)
                              for node, data in
                              zip(self._federated_data, data)]
        clients_embeddings_aggregated = \
            self._aggregator.aggregate_weights(clients_embeddings)
        prediction = self.predict(clients_embeddings_aggregated)

        return prediction

    def evaluate_collaborative_model(self, test_data=None, test_label=None):
        """Evaluates the collaborative model.

        If the global test_data or test_label are not provided,
        the evaluation is made on the batch of train data and labels
        available at the present iteration.

        # Arguments:
            test_data: List, each item representing the global test
                dataset for a single client.
            test_label: Array representing the global labels (the
                same for all clients)
        """

        if test_data is not None and test_label is not None:

            clients_embeddings = [node.predict(data)
                                  for node, data in
                                  zip(self._federated_data, test_data)]
            clients_embeddings_aggregated = \
                self._aggregator.aggregate_weights(clients_embeddings)
            evaluation = self.evaluate(clients_embeddings_aggregated,
                                       test_label)
            print("Collaborative model test evaluation (global, local): " +
                  str(evaluation))

        else:

            evaluation = self.query(server_model=self._model,
                                    meta_params=self.aggregate_weights())
            print("Collaborative model train batch evaluation: " +
                  str(evaluation))

    def aggregate_weights(self):
        """Aggregates the parameters from all clients.

        It is assumed that the last item of each client's
        parameters is constituted by samples' indices (id).
        The latter are not aggregated and must match among all clients.
        """

        clients_meta_params = self._federated_data.query_model()
        params = [client[param] for client in clients_meta_params
                  for param in range(len(client) - 1)]
        samples_indices = [item[-1] for item in clients_meta_params]
        matching_indices = self._check_indices_matching(samples_indices)
        aggregated_meta_params = self._aggregator.aggregate_weights(params)

        return aggregated_meta_params, matching_indices

    @staticmethod
    def _check_indices_matching(sample_indices):
        """Checks all indices are matching.

        Checks that all the nodes' indices received by the
        vertical server are the same. If not, an error is raised.

        # Arguments:
            indices_samples: List of multi-dimensional integer arrays.
                Each entry in the list contains the one client's sample indices.

        # Returns
            matching_indices: Array containing samples' indices.
        """

        if all(np.array_equal(sample_indices[0], item)
               for item in sample_indices):
            return sample_indices[0]
        else:
            raise AssertionError("Clients samples' indices do not match.")
    

class FederatedTransformation(abc.ABC):
    """Applies a federated transformation over the federated data.
    """
    @abc.abstractmethod
    def apply(self, data):
        """Performs an arbitrary transformation on a node's data.

        # Arguments:
            data: The node's data that has to be transformed
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
        federated_array.add_data_node(array[int(last):int(last + split_size)])
        last = last + split_size

    return federated_array


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
