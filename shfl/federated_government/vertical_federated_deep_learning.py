class FederatedGovernmentVertical:
    """
    Class used to represent the central class FederatedGovernment.

    # Arguments:
       models: List containing nodes' trainable models
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

    def __init__(self, models, federated_data, server_node):

        self._federated_data = federated_data
        for data_node, model_node in zip(self._federated_data, models):
            data_node.model = model_node

        self._server = server_node

    def run_rounds(self, n, test_data, test_label, print_freq=1000):
        """
        Run federated learning rounds beginning in the actual state.

        # Arguments:
            n: Number of rounds
            test_data: Test data for evaluation between rounds
            test_label: Test label for evaluation between rounds
            print_freq: frequency for evaluation and print
        """
        for i in range(0, n):

            self._federated_data.train_model()
            self._server.aggregate_weights()
            self._federated_data.train_model(
                meta_params=self._server.query_model())

            if i % print_freq == 0:
                print("Round " + str(i))
                self._server.compute_loss()
                self._server.evaluate_collaborative_model(test_data, test_label)
                print("\n")

