from shfl.private.federated_operation import ServerDataNode


class FederatedGovernment:
    """
    Class used to represent the central class FederatedGovernment.

    # Arguments:
       model_builder: Function that return a trainable model
        (see: [Model](../model))
       federated_data: Federated data to use
        (see: [FederatedData](../private/federated_operation/#federateddata-class))
       aggregator: Federated aggregator function
        (see: [Federated Aggregator](../federated_aggregator))
       model_param_access: Policy to access model's parameters,
        by default non-protected
        (see: [DataAccessDefinition](../private/data/#dataaccessdefinition-class))
       server_node: Optional server node of class FederatedDataNode
        (see: [FederatedDataNode](../private/federated_operation/#federateddatanode-class))
    """

    def __init__(self,
                 model_builder,
                 federated_data,
                 aggregator,
                 server_node=None):

        self._federated_data = federated_data
        for data_node in self._federated_data:
            data_node.model = model_builder()

        if server_node is not None:
            self._server = server_node
        else:
            self._server = ServerDataNode(
                federated_data,
                model_builder(),
                aggregator)

    def evaluate_collaborative_model(self, data_test, label_test):
        """
        Evaluation of the performance of the collaborative model
        (contained in the server node).

        # Arguments:
            test_data: test dataset
            test_label: corresponding labels to test dataset
        """
        evaluation, local_evaluation = \
            self._server.evaluate(data_test, label_test)

        print("Collaborative model test performance : " + str(evaluation))
        if local_evaluation is not None:
            print("Collaborative model server local test performance : "
                  + str(local_evaluation))

    def deploy_collaborative_model(self):
        """
        Deployment of the collaborative learning model from server node to
        each client node.
        """
        self._server.deploy_collaborative_model()

    def evaluate_clients(self, data_test, label_test):
        """
        Evaluation of local learning models over global test dataset.

        # Arguments:
            test_data: test dataset
            test_label: corresponding labels to test dataset
        """
        for data_node in self._federated_data:
            # Predict local model in test
            evaluation, local_evaluation = data_node.evaluate(data_test,
                                                              label_test)
            if local_evaluation is not None:
                print("Performance client " + str(data_node) +
                      ": Global test: " + str(evaluation)
                      + ", Local test: " + str(local_evaluation))
            else:
                print("Test performance client " +
                      str(data_node) + ": " + str(evaluation))

    def train_all_clients(self):
        """
        Train all the clients
        """
        for data_node in self._federated_data:
            data_node.train_model()

    def aggregate_weights(self):
        """
        Aggregate weights from all data nodes and update parameters of
        server's model.
        """
        self._server.aggregate_weights()

    def run_rounds(self, n, test_data, test_label, eval_freq=1):
        """
        Run federated learning rounds beginning in the actual state,
        testing on global test data and local federated_local_test (if any).

        # Arguments:
            n: Number of federated learning rounds
            test_data: Global test data for evaluation between rounds
            test_label: Global test label for evaluation between rounds
            eval_freq: Round frequency for evaluation
        """
        for i in range(0, n):

            self.train_all_clients()

            if i % eval_freq == 0:
                print("Round " + str(i))
                self.evaluate_clients(test_data, test_label)

            self.aggregate_weights()
            self.deploy_collaborative_model()

            if i % eval_freq == 0:
                self.evaluate_collaborative_model(test_data, test_label)
                print("\n")



