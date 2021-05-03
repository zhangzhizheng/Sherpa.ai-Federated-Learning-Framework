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
            [FederatedData](../private/federated_operation/#federateddata-class),
            the set of federated nodes.
        aggregator: The aggregator to use
            (see class [FederatedAggregator](../federated_aggregator)).
        server_node: Optional; Object of class
            [FederatedDataNode](../private/federated_operation/#federateddatanode-class),
            the server node. Default is None, in which case a server node is
            created.
    """

    def __init__(self, model, federated_data,
                 aggregator, server_node=None):

        self._federated_data = federated_data
        for data_node in self._federated_data:
            data_node.model = model

        if server_node is not None:
            self._server = server_node
        else:
            self._server = ServerDataNode(
                federated_data,
                model,
                aggregator)

    def evaluate_clients(self, data, labels):
        """Evaluates the clients' models using a global dataset.

        # Arguments:
            data: The global test data.
            labels: The global target labels.
        """
        for data_node in self._federated_data:
            evaluation, local_evaluation = data_node.evaluate(data,
                                                              labels)
            if local_evaluation is not None:
                print("Performance client " + str(data_node) +
                      ": Global test: " + str(evaluation)
                      + ", Local test: " + str(local_evaluation))
            else:
                print("Test performance client " +
                      str(data_node) + ": " + str(evaluation))

    def run_rounds(self, n, test_data, test_label, eval_freq=1):
        """Runs the federated learning rounds.
        
        It starts in the actual state, testing on global test data and, 
        if present, on local test data too.

        # Arguments:
            n: The number of federated learning rounds to perform.
            test_data: The global test data for evaluation in between rounds.
            test_label: Global test target labels for evaluation
                in between rounds.
            eval_freq: Frequency for evaluation on global test data.
        """
        for i in range(0, n):

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
