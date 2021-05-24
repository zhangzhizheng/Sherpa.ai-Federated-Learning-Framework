from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class FederatedAggregatorTest(FederatedAggregator):
    """Creates a dummy class for the federated aggregator."""
    @property
    def percentage(self):
        """Returns the percentage."""
        return self._percentage

    def aggregate_weights(self, clients_params):
        pass


def test_federated_aggregator_private_data():
    """Checks that the percentage attribute is correctly assigned."""
    percentage = 100
    federated_aggregator = FederatedAggregatorTest(percentage)

    assert federated_aggregator.percentage == percentage
