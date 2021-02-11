from shfl.federated_government.federated_government import FederatedGovernment
from shfl.data_distribution.data_distribution_iid import IidDataDistribution
from shfl.federated_aggregator.fedavg_aggregator import FedAvgAggregator
from shfl.model.linear_regression_model import LinearRegressionModel
from shfl.data_base.california_housing import CaliforniaHousing

from enum import Enum


class LinearRegressionDataBases(Enum):
    """
    Enumeration of possible databases for linear regression.
    """
    CALIFORNIA = CaliforniaHousing


class FederatedLinearRegression(FederatedGovernment):
    """
    Class used to represent a high-level federated linear regression
    (see: [FederatedGoverment](../federated_government/#federatedgovernment-class)).

    # Arguments:
        data_base_name_key: key of the enumeration of valid data bases (see: [LinearRegressionDatabases](../federated_government/#linearregressiondatabases-class))
        iid: boolean which specifies if the distribution if IID (True) or non-IID (False) (True by default)
        num_nodes: number of clients.
        percent: percentage of the database to distribute among nodes.
    """

    def __init__(self, data_base_name_key, num_nodes=20, percent=100):
        if data_base_name_key in LinearRegressionDataBases.__members__.keys():
            module = LinearRegressionDataBases.__members__[data_base_name_key].value
            data_base = module()
            train_data, train_labels, test_data, test_labels = data_base.load_data()

            self._num_features = train_data.shape[1]

            distribution = IidDataDistribution(data_base)

            federated_data, self._test_data, self._test_labels = \
                distribution.get_federated_data(num_nodes=num_nodes,
                                                percent=percent)
            aggregator = FedAvgAggregator()

            super().__init__(self.model_builder(), federated_data, aggregator)

        else:
            raise ValueError("The data base " + data_base_name_key +
                             " is not included. Try with: " +
                             str(", ".join([e.name for e in LinearRegressionDataBases])))

    def run_rounds(self, n=5):
        """
        Overriding of the method of run_rounds of [FederatedGovernment](../federated_government/#federatedgovernment-class)).

        Run one more round beginning in the actual state testing in test data and federated_local_test.

        # Arguments:
            n: Number of rounds (5 by default)
        """
        super().run_rounds(n, self._test_data, self._test_labels)

    def model_builder(self):
        """
        Create a Linear Regression Model.

        # Returns:
            model: Linear Regression Model [LinearRegressionModel](../../model/#linearregressionmodel-class)).
        """
        model = LinearRegressionModel(n_features=self._num_features)
        return model
