class VerticalFederatedGovernment:
    """Defines the vertical federated learning algorithm.

    In the vertical federated learning setting,
    the collaborative training might take place using different
    features in each node, but the same training samples are used (i.e. each
    node possesses a vertical chunk of the features matrix).
    The key difference with respect horizontal FL is that clients'
    models might differ from client to client (e.g. different model type
    and/or architecture), and some nodes might not even possess the target labels.

    # Arguments:
        models: List containing the nodes' trainable models (see: [Model](../model)).
        nodes_federation: Object of class
            [NodesFederation](../private/federated_operation/#nodesfederation-class),
            the set of federated nodes.
        server_node: Object of class
            [VerticalServerDataNode](../private/federated_operation/#verticalserverdatanode-class).
    """

    def __init__(self, models, nodes_federation, server_node):

        self._nodes_federation = nodes_federation
        for data_node, model_node in zip(self._nodes_federation, models):
            data_node.set_model(model_node)

        self._server = server_node

    def run_rounds(self, n_rounds, test_data, test_label, eval_freq=1000):
        """
        Runs the federated learning rounds.

        It starts in the actual state, testing on global test data and, if present, on local test data too.

        # Arguments:
            n_rounds: The number of federated learning rounds to perform.
            test_data: The global test data for evaluation in between rounds.
            test_label: The global test target labels for evaluation in between rounds.
            eval_freq: The frequency for evaluation on global test data.
        """
        for i in range(0, n_rounds):

            self._nodes_federation.train_model()
            clients_meta_params = self._server.aggregate_weights()
            self._server.train_model(meta_params=clients_meta_params)
            server_meta_params = self._server.query_model()
            self._nodes_federation.train_model(meta_params=server_meta_params)

            if i % eval_freq == 0:
                self.evaluate_collaborative_model(i, test_data, test_label)

    def evaluate_collaborative_model(self, iteration, test_data, test_label):
        """
        Evaluates the collaborative model at the current iteration.
        """

        print("Round " + str(iteration))
        self._server.evaluate_collaborative_model()
        self._server.evaluate_collaborative_model(test_data, test_label)
        print("\n")
