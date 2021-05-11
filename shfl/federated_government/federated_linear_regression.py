from enum import Enum

from shfl.federated_government.federated_government import FederatedGovernment
from shfl.data_distribution.data_distribution_iid import IidDataDistribution
from shfl.federated_aggregator.fedavg_aggregator import FedAvgAggregator
from shfl.model.linear_regression_model import LinearRegressionModel
from shfl.data_base.california_housing import CaliforniaHousing


class FederatedLinearRegression(FederatedGovernment):
    """Runs a federated linear regression with minimal user input.

    It overrides the class [FederatedGovernment](./#federatedgovernment-class).

    Runs a linear regression federated learning experiment
    with predefined values. This way, it suffices to just specify
    which dataset to use.

    # Arguments:
        data_base_name_key: Key of a valid data base (see possibilities in class
            [LinearRegressionDatabases](./#linearregressiondatabases-class)).
        num_nodes: Optional; number of client nodes (default is 20).
        percent: Optional; Percentage of the database to distribute
            among nodes (by default set to 100, in which case
            all the available data is used).
    """

    def __init__(self, data_base_name_key, num_nodes=20, percent=100):
        if data_base_name_key in LinearRegressionDataBases.__members__.keys():

            data_base = LinearRegressionDataBases[data_base_name_key].value()
            data_base.load_data()
            train_data, train_label = data_base.train

            n_features = train_data.shape[1]
            try:
                n_targets = train_label.shape[1]
            except IndexError:
                n_targets = 1
            model = LinearRegressionModel(n_features=n_features,
                                          n_targets=n_targets)

            distribution = IidDataDistribution(data_base)
            federated_data, self._test_data, self._test_labels = \
                distribution.get_federated_data(num_nodes=num_nodes,
                                                percent=percent)
            aggregator = FedAvgAggregator()

            super().__init__(model, federated_data, aggregator)

        else:
            raise ValueError(
                "The data base " + data_base_name_key +
                " is not included. Try with: " +
                str(", ".join([e.name for e in LinearRegressionDataBases])))

    def run_rounds(self, n_rounds=5, **kwargs):
        """See base class.
        """
        super().run_rounds(n_rounds, self._test_data, self._test_labels, **kwargs)


class LinearRegressionDataBases(Enum):
    """Enumerates the available databases for linear regression.

    Options are: `"CALIFORNIA"`.
    """
    CALIFORNIA = CaliforniaHousing
