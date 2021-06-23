import numpy as np
from sklearn.cluster import KMeans


def cluster_fed_avg_aggregator(clients_params):
    """Performs a cluster aggregation.

    It performs a k_highest-means round to find the minimum distance
    of cluster centroids coming from each node.

    # Arguments:
        clients_params: List where each item contains one client's parameters.
            One client's parameters can be a (nested) list or tuples of
            array-like objects.

        # Returns:
            aggregated_params: The aggregated clients' parameters.

    # References:
        [sklearn.cluster.KMeans](https://scikit-learn.org/stable/
        modules/generated/sklearn.cluster.KMeans.html)
    """
    clients_params_array = np.concatenate(clients_params)

    n_clusters = clients_params[0].shape[0]
    model_aggregator = KMeans(n_clusters=n_clusters, init='k-means++')
    model_aggregator.fit(clients_params_array)
    aggregated_weights = np.array(model_aggregator.cluster_centers_)

    return aggregated_weights
