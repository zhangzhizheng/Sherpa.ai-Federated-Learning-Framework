class VerticalFederatedGovernment:
    """
    Class used to represent the central class FederatedGovernment in a
    Vertical Federated Learning (FL) setting.
    Essentially, the collaborative training might take place using different
    features in each node, but the same training samples are used (i.e. each
    node possesses a vertical chunk of the features matrix).
    The key difference with respect Horizontal FL is that clients'
    models might differ from client to client (e.g. different model type
    and/or architecture), and some nodes might not even possess the target labels.

    # Arguments:
       models: List containing nodes' trainable models
        (see: [Model](../model))
       federated_data: Federated data to use.
        (see: [FederatedData](../private/federated_operation/#federateddata-class))
       server_node: Server node of class FederatedDataNode
        (see: [FederatedDataNode](../private/federated_operation/#federateddatanode-class))
    """

    def __init__(self, models, federated_data, server_node):

        self._federated_data = federated_data
        for data_node, model_node in zip(self._federated_data, models):
            data_node.model = model_node

        self._server = server_node

    def run_rounds(self, n_rounds, test_data, test_label, eval_freq=1000):
        """
        Run federated learning rounds beginning in the actual state.

        # Arguments:
            n_rounds: Number of federated learning rounds
            test_data: Global test data for evaluation between rounds
            test_label: Global test labels for evaluation between rounds
            eval_freq: Frequency for evaluation and print on global test data
        """
        for i in range(0, n_rounds):

            self._federated_data.train_model()
            clients_meta_params = self._server.aggregate_weights()
            self._server.train_model(meta_params=clients_meta_params)
            server_meta_params = self._server.query_model()
            self._federated_data.train_model(meta_params=server_meta_params)

            if i % eval_freq == 0:
                print("Round " + str(i))
                self._server.evaluate_collaborative_model()
                self._server.evaluate_collaborative_model(test_data, test_label)
                print("\n")
