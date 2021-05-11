import numpy as np
from sklearn.cluster import KMeans

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class ClusterFedAvgAggregator(FederatedAggregator):
    """Performs a cluster aggregation.

    It implements the class
    [FederatedAggregator](./#federatedaggregator-class).

    It performs a k_highest-means round to find the minimum distance
    of cluster centroids coming from each node.

    # References:
        [sklearn.cluster.KMeans](https://scikit-learn.org/stable/
        modules/generated/sklearn.cluster.KMeans.html)
    """

    def aggregate_weights(self, clients_params):
        """
        See base class.
        """
        clients_params_array = np.concatenate(clients_params)

        n_clusters = clients_params[0].shape[0]
        model_aggregator = KMeans(n_clusters=n_clusters, init='k-means++')
        model_aggregator.fit(clients_params_array)
        aggregated_weights = np.array(model_aggregator.cluster_centers_)
        return aggregated_weights
