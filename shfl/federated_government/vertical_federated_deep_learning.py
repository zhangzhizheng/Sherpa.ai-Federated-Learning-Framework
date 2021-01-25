class FederatedGovernmentVertical:
    """
    Class used to represent the central class FederatedGovernment.

    # Arguments:
       model_builder: Function that returns a trainable model
        (see: [Model](../model))
       federated_data: Federated data to use.
        (see: [FederatedData](../private/federated_operation/#federateddata-class))
       aggregator: Federated aggregator function
        (see: [Federated Aggregator](../federated_aggregator))
       model_param_access: Policy to access model's parameters,
        by default non-protected
        (see: [DataAccessDefinition](../private/data/#dataaccessdefinition-class))

    # Properties:
        global_model: Return the global model.
    """

    def __init__(self,
                 model_builder,
                 federated_data,
                 server_node):

        self._federated_data = federated_data
        for data_node, model_node in zip(self._federated_data, model_builder):
            data_node.model = model_node

        self._server = server_node

    def train_all_clients(self, **kwargs):
        """
        Train all the clients
        """
        for data_node in self._federated_data:
            data_node.train_model(**kwargs)

    def run_rounds(self, n, test_data, test_label, print_freq=1000):
        """
        Run federated learning rounds beginning in the actual state.

        # Arguments:
            n: Number of rounds
            test_data: Test data for evaluation between rounds
            test_label: Test label for evaluation between rounds

        """
        for i in range(0, n):

            self.train_all_clients()
            self._server.aggregate_weights()
            self.train_all_clients(embeddings_grads=self._server.query_model())

            if i % print_freq == 0:
                print("Round " + str(i))
                self._server.compute_loss()
                self._server.evaluate_collaborative_model(test_data, test_label)
                print("\n")
