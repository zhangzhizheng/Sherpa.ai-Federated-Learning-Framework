from shfl.private.federated_operation import ServerDataNode


class FederatedGovernment:
    """
    Class used to represent the central class FederatedGovernment.

    # Arguments:
       model: Object representing a trainable model
        (see: [Model](../model))
       federated_data: Federated data to use
        (see: [FederatedData](../private/federated_operation/#federateddata-class))
       aggregator: Federated aggregator function
        (see: [Federated Aggregator](../federated_aggregator))
       server_node: Optional server node of class FederatedDataNode
        (see: [FederatedDataNode](../private/federated_operation/#federateddatanode-class))
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

    def evaluate_clients(self, test_data, test_label):
        """
        Evaluation of local learning models over global test dataset.

        # Arguments:
            test_data: test dataset
            test_label: corresponding labels to test dataset
        """
        for data_node in self._federated_data:
            # Predict local model in test
            evaluation, local_evaluation = data_node.evaluate(test_data,
                                                              test_label)
            if local_evaluation is not None:
                print("Performance client " + str(data_node) +
                      ": Global test: " + str(evaluation)
                      + ", Local test: " + str(local_evaluation))
            else:
                print("Test performance client " +
                      str(data_node) + ": " + str(evaluation))

    def run_rounds(self, n, test_data, test_label, eval_freq=1):
        """
        Run federated learning rounds beginning in the actual state,
        testing on global test data and local federated_local_test (if any).

        # Arguments:
            n: Number of federated learning rounds
            test_data: Global test data for evaluation between rounds
            test_label: Global test labels for evaluation between rounds
            eval_freq: Frequency for evaluation and print on global test data
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
