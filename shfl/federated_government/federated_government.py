"""
This file is part of Sherpa Federated Learning Framework.

Sherpa Federated Learning Framework is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3.

Sherpa Federated Learning Framework is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
"""

from shfl.private.federated_operation import ServerDataNode


class FederatedGovernment:
    """Defines the horizontal federated learning algorithm.

    Coordinates the sequence of operations to be performed
    by the server and the clients nodes.
    This class can be overridden to define new custom federated algorithms.

    # Arguments:
        model: Object representing a trainable model
            (see class [Model](../model)).
        nodes_federation: Object of class
            [NodesFederation](../private/federated_operation/#nodesfederation-class),
            the set of federated nodes.
        aggregator: Optional; The aggregator to use
            (see class [FederatedAggregator](../federated_aggregator)).
            If not specified as argument, the argument `server_node` must be provided.
        server_node: Optional; Object of class
            [ServerDataNode](../private/federated_operation/#serverdatanode-class),
            the server node. Default is None, in which case a server node is
            created using the `model`, `nodes_federation` and `aggregator` provided.
    """

    def __init__(self, model, nodes_federation, aggregator=None, server_node=None):

        if aggregator is None and server_node is None:
            raise AssertionError("Either the aggregator or the server node "
                                 "must be provided.")
        self._nodes_federation = nodes_federation
        for data_node in self._nodes_federation:
            data_node.set_model(model)

        if server_node is not None:
            self._server = server_node
        else:
            self._server = ServerDataNode(
                nodes_federation,
                model,
                aggregator)

    def run_rounds(self, n_rounds, test_data, test_label, eval_freq=1):
        """Runs the federated learning rounds.

        It starts in the actual state, testing on global test data
        and, if present, on local test data too.

        # Arguments:
            n_rounds: The number of federated learning rounds to perform.
            test_data: The global test data for evaluation in between rounds.
            test_label: The global test target labels for evaluation
                in between rounds.
            eval_freq: The frequency for evaluation on global test data.
        """

        for i in range(0, n_rounds):

            self._nodes_federation.train_model()

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

        results = self._nodes_federation.evaluate(data, labels)

        for result in results:
            evaluation, local_evaluation = result

            print(" Global test: " + str(evaluation)
                  + ", Local test: " + str(local_evaluation))
