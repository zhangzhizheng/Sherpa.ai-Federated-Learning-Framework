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
            train_data, train_labels, test_data, test_labels = \
                data_base.load_data()

            self._num_features = train_data.shape[1]

            distribution = IidDataDistribution(data_base)

            federated_data, self._test_data, self._test_labels = \
                distribution.get_federated_data(num_nodes=num_nodes,
                                                percent=percent)
            aggregator = FedAvgAggregator()

            super().__init__(self.model_builder(), federated_data, aggregator)

        else:
            raise ValueError(
                "The data base " + data_base_name_key +
                " is not included. Try with: " +
                str(", ".join([e.name for e in LinearRegressionDataBases])))

    def run_rounds(self, n=5, **kwargs):
        """See base class.
        """
        super().run_rounds(n, self._test_data, self._test_labels, **kwargs)

    def model_builder(self):
        """Creates a linear regression model.

        # Returns:
            model: Object of class
                [LinearRegressionModel](../model/supervised/#linearregressionmodel),
                the linear regression model to use.
        """
        model = LinearRegressionModel(n_features=self._num_features)
        return model


class LinearRegressionDataBases(Enum):
    """Enumerates the available databases for linear regression.

    Options are: `"CALIFORNIA"`.
    """
    CALIFORNIA = CaliforniaHousing
