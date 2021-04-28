import abc
import numpy as np

from shfl.private.node import DataNode
from shfl.private.data import LabeledData
from shfl.data_base.data_base import split_train_test


class FederatedData:
    """Represents the set of federated nodes.

    This object contains the data nodes of the federated experiment.
    All node's callable methods (see callable methods in class
    [FederatedDataNode](./#federateddatanode-class)) are also
    available at "federated level", meaning that when invoking a node's
    method from this object, the method gets executed on all the member nodes.

    # Example:

    ```python
        from shfl.data_base import Emnist
        from shfl.data_distribution import IidDataDistribution
        from shfl.private.data import UnprotectedAccess


        # Create federated data from a dataset:
        database = Emnist()
        database.load_data()
        iid_distribution = IidDataDistribution(database)
        federated_data, test_data, test_labels = \\
            iid_distribution.get_federated_data(num_nodes=5, percent=10)

        # Data access definition at "node level" (only node 0):
        federated_data[0].configure_data_access(UnprotectedAccess())
        print(federated_data[0].query())
        # print(federated_data[1].query())  # Raises an exception

        # Data access definition at "federate level" (all nodes at once):
        federated_data.configure_data_access(UnprotectedAccess())
        print(federated_data.query())
    ```
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

    def append_data_node(self, data):
        """Appends a new node to the set of federated nodes.

        # Arguments:
            data: Private data of the node to append.
        """
        node = FederatedDataNode(str(id(self)))
        node.set_private_data(data)
        self._data_nodes.append(node)

    def num_nodes(self):
        """Returns the total number of nodes in the set of federated nodes.
        """
        return len(self._data_nodes)

    def _create_apply_method(self, method):
        """Applies a method on all the set of federated nodes.

        Dynamically creates a loop ever all the set of federated nodes
        to call a node's method (see callable methods in class
        [FederatedDataNode](./#federateddatanode-class)).

        # Arguments:
            method: String corresponding to a node's method.

        # Returns:
            node_method: Function that loops the desired method
                on all the nodes.
        """

        def apply_method(*args, **kwargs):
            """Applies a method on the set of federated nodes.

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
    """Represents a data node as a member of a set of federated nodes.

    Extends the class [DataNode](../data_node/#datanode-class).
    The main difference with respect its base class is that this
    class allows calls to node's methods without having to
    explicitly specify a private identifier, since the latter
    is being set at initialization time.

    As for an individual data node, the access to the private data must
    be configured with an access policy before querying it or an
    exception will be raised (see example in the class
    [FederatedData](./#federateddata-class)).

    # Arguments:
        federated_data_identifier: String identifying the set
            of federated nodes.
    """
    def __init__(self, federated_data_identifier):
        super().__init__()
        self._federated_data_identifier = federated_data_identifier

    def query(self, private_property=None, **kwargs):
        """See base class.
        """
        if private_property is None:
            private_property = self._federated_data_identifier
        return super().query(private_property, **kwargs)

    def configure_data_access(self, data_access_definition):
        """See base class.
        """
        super().configure_data_access(self._federated_data_identifier,
                                      data_access_definition)

    def set_private_data(self, data):
        """See base class.
        """
        super().set_private_data(self._federated_data_identifier, data)

    def set_private_test_data(self, data):
        """See base class.
        """
        super().set_private_test_data(self._federated_data_identifier, data)

    def train_model(self, **kwargs):
        """See base class.
        """
        super().train_model(self._federated_data_identifier, **kwargs)

    def apply_data_transformation(self, federated_transformation):
        """See base class.
        """
        super().apply_data_transformation(self._federated_data_identifier,
                                          federated_transformation)

    def evaluate(self, data, labels):
        """Evaluates the performance of the model.

        The node's model is evaluated on both input data, and, if present,
        on node's local test data.

        # Arguments:
            data: The data on which to make the evaluation.
            labels: The true labels.

        # Returns:
            metrics: Metrics for the evaluation.
        """
        return super().evaluate(data, labels), \
            super().local_evaluate(self._federated_data_identifier)

    def split_train_test(self, train_proportion=0.8):
        """Splits node's private data into train and test sets.

        # Arguments:
            train_proportion: Optional; Float between 0 and 1 proportional to the
                amount of data to dedicate to train. If 1 is provided, all data is
                assigned to train (default is 0.8).
        """
        labeled_data = self._private_data.get(self._federated_data_identifier)
        train_data, train_labels, test_data, test_labels = \
            split_train_test(labeled_data.data, labeled_data.label,
                             train_proportion=train_proportion)

        self.set_private_data(LabeledData(train_data, train_labels))
        self.set_private_test_data(LabeledData(test_data, test_labels))


class ServerDataNode(FederatedDataNode):
    """Represents a server node in the horizontal federated learning setting.

    Extends the class [FederatedDataNode](./#federateddatanode-class).
    The server node is in charge of querying the clients
    (i.e. the set of federated nodes).
    In the horizontal federated learning setting, typical queries
    are to deploy (update) the collaborative model over
    the client nodes, and to aggregate the clients' models
    into the collaborative one, held by the server.

    # Arguments:
        federated_data: The clients (i.e. the set of federated nodes,
            see class [FederatedData](./#federateddata-class)).
        model: Object representing the collaborative model.
        aggregator: Object representing the aggregator to use (see class
            [FederatedAggregator](../../federated_aggregator/#federatedaggregator-class)).
        data: Optional; Object of class [LabeledData](../data/#labeleddata-class)
            containing the server's private data.
    """

    def __init__(self, federated_data, model, aggregator, data=None):
        super().__init__(federated_data_identifier=str(id(federated_data)))
        self._federated_data = federated_data
        self.model = model
        self._aggregator = aggregator
        self.set_private_data(data)

    def deploy_collaborative_model(self):
        """Sends the server's model to each client node.
        """
        self._federated_data.set_model_params(self.query_model_params())

    def evaluate_collaborative_model(self, data, labels):
        """Evaluates the performance of the collaborative model.

        # Arguments:
            data: The data on which to make the evaluation.
            labels: The true labels.
        """
        evaluation, local_evaluation = \
            self.evaluate(data, labels)

        print("Collaborative model test performance : " + str(evaluation))
        if local_evaluation is not None:
            print("Collaborative model server local test performance : "
                  + str(local_evaluation))

    def aggregate_weights(self):
        """Aggregate model's parameters from client nodes.

        After aggregation, updates the collaborative model.
        """

        weights = self._federated_data.query_model_params()
        aggregated_weights = self._aggregator.aggregate_weights(weights)
        self._model.set_model_params(aggregated_weights)


class VerticalServerDataNode(FederatedDataNode):
    """Represents a server node in the vertical federated learning setting.

    Extends the class [FederatedDataNode](./#federateddatanode-class).
    As opposed to the horizontal setting, in the vertical
    federated learning setting the collaborative model is typically
    distributed between the clients and the server.
    As a result, neither the clients nor the server hold the entire model,
    but only part of it. The server node is still in charge
    of querying the clients (i.e. the set of federated nodes).
    Typical queries in this case are communication of meta-parameters
    (e.g. embeddings, gradients etc.) with the clients,
    as well as prediction and evaluation of the collaborative model.

    # Arguments:
        federated_data: The clients (i.e. the set of federated nodes,
            see class [FederatedData](./#federateddata-class)).
        model: Object representing the collaborative model.
        aggregator: Object representing the aggregator to use (see class
            [FederatedAggregator](../../federated_aggregator/#federatedaggregator-class)).
        data: Optional; Object of class [LabeledData](../data/#labeleddata-class)
            containing the server's private data.
    """

    def __init__(self, federated_data, model, aggregator, data=None):
        super().__init__(federated_data_identifier=str(id(federated_data)))
        self._federated_data = federated_data
        self.model = model
        self._aggregator = aggregator
        self.set_private_data(data)

    def predict_collaborative_model(self, data):
        """"Makes a prediction on input data using the collaborative model.

        # Arguments:
            data: List, each item represents the input data
                on which to make the prediction for a single client.

        # Returns:
            prediction: The prediction using the collaborative model.
        """
        clients_embeddings = [node.predict(data)
                              for node, data in
                              zip(self._federated_data, data)]
        clients_embeddings_aggregated = \
            self._aggregator.aggregate_weights(clients_embeddings)
        prediction = self.predict(clients_embeddings_aggregated)

        return prediction

    def evaluate_collaborative_model(self, data=None, labels=None):
        """"Evaluates the performance of the collaborative model.

        If the global test_data or test_label are not provided,
        the evaluation is made on the batch of train data and labels
        available at the present iteration.

        # Arguments:
            data: Optional; List, each item representing the global test
                dataset for a single client.
            label: Optional; Array representing the global labels (the
                same for all clients).
        """

        if data is not None and labels is not None:

            clients_embeddings = [node.predict(data)
                                  for node, data in
                                  zip(self._federated_data, data)]
            clients_embeddings_aggregated = \
                self._aggregator.aggregate_weights(clients_embeddings)
            evaluation = self.evaluate(clients_embeddings_aggregated,
                                       labels)
            print("Collaborative model test evaluation (global, local): " +
                  str(evaluation))

        else:

            evaluation = self.query(server_model=self._model,
                                    meta_params=self.aggregate_weights())
            print("Collaborative model train batch evaluation: " +
                  str(evaluation))

    def aggregate_weights(self):
        """Aggregate model's meta-parameters from client nodes.

        It is assumed that the last item of each client's
        parameters is constituted by samples' indices (id).
        The latter are not aggregated and must match among all clients,
        otherwise an exception will be raised.

        # Returns
            aggregated_meta_params: Array-like object containing the
                aggregated meta-parameters.
            matching_indices: Array-like object containing a single copy
                of samples' indices.
        """

        clients_meta_params = self._federated_data.query_model()
        params = [client[param] for client in clients_meta_params
                  for param in range(len(client) - 1)]
        samples_indices = [item[-1] for item in clients_meta_params]
        self._check_indices_matching(samples_indices)
        aggregated_meta_params = self._aggregator.aggregate_weights(params)

        return aggregated_meta_params, samples_indices[0]

    @staticmethod
    def _check_indices_matching(sample_indices):
        """Checks that all samples' indices match.

        Checks that all the samples' indices received by the
        server are the same. If not, an exception is raised.

        # Arguments:
            sample_indices: List, each entry contains
                one client's sample indices.
        """

        if not all(np.array_equal(sample_indices[0], item)
                   for item in sample_indices):
            raise AssertionError("Clients samples' indices do not match.")


class FederatedTransformation(abc.ABC):
    """Applies a federated transformation over the federated data.
    """

    @abc.abstractmethod
    def apply(self, data):
        """Applies an arbitrary transformation on the node's private data.

        This method must be implemented in order to define how
        to transform the node's private data.

        # Arguments:
            data: The node's private data to be transformed.

        # Example:
            See implementation of class [Normalize](./#normalize-class).
        """


class Normalize(FederatedTransformation):
    """Applies a normalization over a set of federated nodes.

    It implements class
    [FederatedTransformation](./#federatedtransformation-class).

    # Arguments:
        mean: Mean used for the normalization.
        std: Standard deviation used for the normalization.
    """
    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def apply(self, data):
        data.data = (data.data - self.__mean) / self.__std


def federate_array(data, num_nodes):
    """Creates a set of federated nodes from an array-like object.

    The data in the input array is evenly split among
    a specified number of nodes using the array's first dimension.

    # Arguments:
        array: Array-like object with any number of dimensions.
        num_nodes: Number of nodes for splitting.

    # Returns:
        federated_data: Object of class [FederatedData](./#federateddata-class).
    """
    split_size = len(data) / float(num_nodes)
    last = 0.0
    federated_data = FederatedData()
    while last < len(data):
        federated_data.append_data_node(data[int(last):int(last + split_size)])
        last = last + split_size

    return federated_data


def federate_list(data, labels=None):
    """Converts a list to a set of federated nodes.

    # Arguments:
        data: List, each element contains one client's private data.
        labels: Optional; List, each element contains one client's
            target labels.

    # Returns:
        federated_data: Object of class [FederatedData](./#federateddata-class).
    """
    if labels is None:
        labels = [None] * len(data)

    federated_data = FederatedData()
    for node_data, node_labels in zip(data, labels):
        node_labeled_data = LabeledData(node_data, node_labels)
        federated_data.append_data_node(node_labeled_data)

    return federated_data
