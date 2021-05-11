from enum import Enum
import numpy as np

from shfl.federated_government.federated_government import FederatedGovernment
from shfl.data_distribution.data_distribution_iid import IidDataDistribution
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
        data_distribution: Optional; Reference to the object defining the data sampling.
            Options are
            [IidDataDistribution](../data_distribution/#iiddatadistribution-class) (default)
            and [NonIidDataDistribution](../data_distribution/#noniiddatadistribution-class).
        num_nodes: Optional; number of client nodes (default is 20).
        percent: Optional; Percentage of the database to distribute
            among nodes (by default set to 100, in which case
            all the available data is used).
    """

    def __init__(self, data_base_name_key,
                 data_distribution=IidDataDistribution,
                 num_nodes=20, percent=100):
        if data_base_name_key in ClusteringDataBases.__members__.keys():
            data_base = ClusteringDataBases[data_base_name_key].value()
            data_base.load_data()
            train_data, train_labels = data_base.train

            n_clusters = len(np.unique(train_labels))
            n_features = train_data.shape[1]
            model = KMeansModel(n_clusters=n_clusters,
                                n_features=n_features)

            federated_data, self._test_data, self._test_labels = \
                data_distribution(data_base).get_federated_data(
                    num_nodes=num_nodes,
                    percent=percent)

            aggregator = ClusterFedAvgAggregator()

            super().__init__(model, federated_data, aggregator)

        else:
            raise ValueError(
                "The data base " + data_base_name_key +
                " is not included. Try with: " +
                str(", ".join([e.name for e in ClusteringDataBases])))

    def run_rounds(self, n_rounds=5, **kwargs):
        """See base class.
        """
        super().run_rounds(n_rounds, self._test_data, self._test_labels, **kwargs)


class ClusteringDataBases(Enum):
    """Enumerates the available databases for clustering.

    Options are: `"IRIS"`.
    """
    IRIS = Iris
