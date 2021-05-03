from enum import Enum
import numpy as np

from shfl.federated_government.federated_government import FederatedGovernment
from shfl.data_distribution.data_distribution_iid import IidDataDistribution
from shfl.data_distribution.data_distribution_non_iid import NonIidDataDistribution
from shfl.federated_aggregator.cluster_fedavg_aggregator import ClusterFedAvgAggregator
from shfl.model.kmeans_model import KMeansModel
from shfl.data_base.iris import Iris


class FederatedClustering(FederatedGovernment):
    """Runs a federated clustering with minimal user input.

    It overrides the class [FederatedGovernment](./#federatedgovernment-class).

    Runs a clustering federated learning experiment
    with predefined values. This way, it suffices to just specify
    which dataset to use.

    # Arguments:
        data_base_name_key: Key of a valid data base (see possibilities
            in class [ClusteringDataBases](./#clusteringdatabases-class)).
        iid: Optional; Boolean specifying whether the data distribution IID or
            non-IID. By default set to `iid=True`.
        num_nodes: Optional; number of client nodes (default is 20).
        percent: Optional; Percentage of the database to distribute
            among nodes (by default set to 100, in which case
            all the available data is used).
    """

    def __init__(self, data_base_name_key, iid=True, num_nodes=20, percent=100):
        if data_base_name_key in ClusteringDataBases.__members__.keys():
            data_base = ClusteringDataBases[data_base_name_key].value()
            train_data, train_labels, \
                test_data, test_labels = data_base.load_data()

            self._num_clusters = len(np.unique(train_labels))
            self._num_features = train_data.shape[1]

            if iid:
                distribution = IidDataDistribution(data_base)
            else:
                distribution = NonIidDataDistribution(data_base)

            federated_data, self._test_data, self._test_labels = \
                distribution.get_federated_data(num_nodes=num_nodes,
                                                percent=percent)

            aggregator = ClusterFedAvgAggregator()

            super().__init__(self.model_builder(), federated_data, aggregator)

        else:
            raise ValueError(
                "The data base " + data_base_name_key +
                " is not included. Try with: " +
                str(", ".join([e.name for e in ClusteringDataBases])))

    def run_rounds(self, n=5, **kwargs):
        """See base class.
        """
        super().run_rounds(n, self._test_data, self._test_labels, **kwargs)

    def model_builder(self):
        """Creates a k-means model.

        # Returns:
            model: Object of class [KMeansModel](../model/unsupervised/#kmeansmodel),
                the k-means model to use.
        """
        model = KMeansModel(n_clusters=self._num_clusters,
                            n_features=self._num_features)
        return model


class ClusteringDataBases(Enum):
    """Enumerates the available databases for clustering.

    Options are: `"IRIS"`.
    """
    IRIS = Iris
