from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class FederatedAggregatorTest(FederatedAggregator):
    """Creates a dummy class for the federated aggregator."""
    @property
    def axis(self):
        """Returns the percentage."""
        return self._axis

    def aggregate_weights(self, clients_params):
        pass


def test_federated_aggregator_private_data():
    """Checks that the percentage attribute is correctly assigned."""
    axis = 0
    federated_aggregator = FederatedAggregatorTest(axis)

    assert federated_aggregator.axis == axis
