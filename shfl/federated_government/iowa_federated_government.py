# TODO: Need to refactor this whole code.
# Read the article and comply with Pylint.
import numpy as np

from shfl.federated_government.federated_government import FederatedGovernment
from shfl.federated_aggregator.iowa_federated_aggregator import IowaFederatedAggregator


class IowaFederatedGovernment(FederatedGovernment):
    """Defines the IOWA federated learning algorithm.

    It overrides the class [FederatedGovernment](./#federatedgovernment-class).

    See base class.

    # Arguments:
        model: Object representing a trainable model
            (see class [Model](../model)).
        federated_data: Object of class
            [NodesFederation](../private/federated_operation/#federateddata-class),
            the set of federated nodes.
        dynamic: Optional; Boolean indicating whether we use the dynamic
            or static version (default is True).
        a: Optional; First argument of linguistic quantifier (default is 0).
        b: Optional; Second argument of linguistic quantifier (default is 0.2).
        c: Optional; Third argument of linguistic quantifier (default is 0.8).
        y_b: Optional; Fourth argument of linguistic quantifier (default is 0.4).
        k_highest: Optional; Distance param of the dynamic version (default is 3/4).

    # References:
        [Dynamic federated learning model for identifying
        adversarial clients](https://arxiv.org/abs/2007.15030)
    """

    def __init__(self, model, federated_data, dynamic=True, a=0,
                 b=0.2, c=0.8, y_b=0.4, k=3/4):
        super().__init__(model, federated_data, IowaFederatedAggregator())

        self._a = a
        self._b = b
        self._c = c
        self._y_b = y_b
        self._k = k
        self._dynamic = dynamic

    def performance_clients(self, data_val, label_val):
        """Evaluates clients' models over a global validation dataset.

        # Arguments:
            val_data: The global validation dataset.
            val_label: The global target labels.

        # Returns:
            client_performance: Performance for each client.
        """
        client_performance = []
        for data_node in self._federated_data:
            local_performance = data_node.performance(data_val, label_val)
            client_performance.append(local_performance)

        return np.array(client_performance)

    def run_rounds(self, n_rounds, test_data, test_label):
        """
        See base class.
        """
        randomize = np.arange(len(test_label))
        np.random.shuffle(randomize)
        test_data = test_data[randomize, ]
        test_label = test_label[randomize]

        # Split between validation and test
        validation_data = test_data[:int(0.15*len(test_label)), ]
        validation_label = test_label[:int(0.15*len(test_label))]

        test_data = test_data[int(0.15 * len(test_label)):, ]
        test_label = test_label[int(0.15 * len(test_label)):]

        for i in range(0, n_rounds):
            print("Accuracy round " + str(i))
            self._server.deploy_collaborative_model()
            self._federated_data.train_model()
            self.evaluate_clients(test_data, test_label)
            client_performance = self.performance_clients(
                validation_data, validation_label)
            self._server._aggregator.set_ponderation(
                client_performance, self._dynamic, self._a,
                self._b, self._c, self._y_b, self._k)
            self._server.aggregate_weights()
            self._server.evaluate_collaborative_model(test_data, test_label)
            print("\n\n")
