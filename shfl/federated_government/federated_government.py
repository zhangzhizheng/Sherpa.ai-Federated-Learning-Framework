from shfl.private.federated_operation import ServerDataNode


class FederatedGovernment:
    """Defines the horizontal federated learning algorithm.

    Coordinates the sequence of operations to be performed
    by the server and the clients nodes.
    This class can be overridden to define new custom federated algorithms.

    # Arguments:
        model: Object representing a trainable model
            (see class [Model](../model)).
        federated_data: Object of class
            [NodesFederation](../private/federated_operation/#nodesfederation-class),
            the set of federated nodes.
        aggregator: Optional; The aggregator to use
            (see class [FederatedAggregator](../federated_aggregator)).
            If not specified as argument, the argument `server_node` must be provided.
        server_node: Optional; Object of class
            [FederatedDataNode](../private/federated_operation/#federateddatanode-class),
            the server node. Default is None, in which case a server node is
            created using the `model`, `federated_data` and `aggregator` provided.
    """

    def __init__(self, model, federated_data,
                 aggregator=None, server_node=None):

        if aggregator is None and server_node is None:
            raise AssertionError("Either the aggregator or the server node "
                                 "must be provided.")
        self._federated_data = federated_data
        for data_node in self._federated_data:
            data_node.set_model(model)

        if server_node is not None:
            self._server = server_node
        else:
            self._server = ServerDataNode(
                federated_data,
                model,
                aggregator)

    def run_rounds(self, n_rounds, test_data, test_label, eval_freq=1):
        """Runs the federated learning rounds.

        It starts in the actual state, testing on global test data
        and, if present, on local test data too.

        # Arguments:
            n_rounds: The number of federated learning rounds to perform.
            test_data: The global test data for evaluation in between rounds.
            test_label: Global test target labels for evaluation
                in between rounds.
            eval_freq: Frequency for evaluation on global test data.
        """

        for i in range(0, n_rounds):

            self._federated_data.train_model()

            if i % eval_freq == 0:
                print("Round " + str(i))
                self.evaluate_clients(test_data, test_label)

            self._server.aggregate_weights()
            self._server.deploy_collaborative_model()

            if i % eval_freq == 0:
                self._server.evaluate_collaborative_model(
                    test_data, test_label)
                print("\n")

    def evaluate_clients(self, data, labels):
        """Evaluates the clients' models using a global dataset.

        # Arguments:
            data: The global test data.
            labels: The global target labels.
        """

        for data_node in self._federated_data:
            evaluation, local_evaluation = \
                data_node.evaluate(data, labels)

            print("Performance client " + str(data_node) +
                  ": Global test: " + str(evaluation)
                  + ", Local test: " + str(local_evaluation))