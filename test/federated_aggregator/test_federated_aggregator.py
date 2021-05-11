from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class FederatedAggregatorTest(FederatedAggregator):
    def aggregate_weights(self, clients_params):
        pass


def test_federated_aggregator_private_data():
    percentage = 100
    fa = FederatedAggregatorTest(percentage)

    assert fa._percentage == percentage

