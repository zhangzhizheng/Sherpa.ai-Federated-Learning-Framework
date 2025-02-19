"""
This file is part of Sherpa Federated Learning Framework.

Sherpa Federated Learning Framework is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3.

Sherpa Federated Learning Framework is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
"""

import abc

from shfl.private.federated_operation import federate_list


class DataDistribution(abc.ABC):
    """Distributes a centralized database to a set of federated nodes.

    The input is an object of class
    [Database](../databases/#database-class), whose
    train data and labels are distributed to a set of nodes.
    Instead, the test data and labels are left unaltered since
    they are meant to be used as a centralized (global) test set.
    The way the data is distributed must be specified in the
    abstract method `make_data_federated` of this class.

    # Arguments:
        database: Object of class [Database](../databases/#database-class),
            the centralized database to be distributed among nodes.
    """

    def __init__(self, database):
        self._database = database

    def get_nodes_federation(self, **kwargs):
        """Gets the set of federated nodes.

        Assigns to each node the corresponding data partition as
        defined by the abstract method
        [make_data_federated](./#make_data_federated).

        # Arguments:
            **kwargs: Optional named arguments.
                These are passed to the call of the abstract method
                [make_data_federated](./#make_data_federated).

        # Returns:
            nodes_federation: Object of class
                [NodesFederation](../private/federated_operation/#federateddata-class),
                the set of federated nodes containing the distributed train data.
            test_data: The centralized (global) test data.
            test_label: The centralized (global) target labels.
        """

        train_data, train_label = self._database.train
        test_data, test_label = self._database.test

        federated_train_data, federated_train_label = \
            self.make_data_federated(train_data, train_label, **kwargs)

        federated_data = federate_list(federated_train_data,
                                       federated_train_label)

        return federated_data, test_data, test_label

    @abc.abstractmethod
    def make_data_federated(self, data, labels, **kwargs):
        """Creates the data partition for each client.

        Abstract method.

        Defines how the centralized data is distributed among the client nodes.

        # Arguments:
            data: The train data to be distributed among
                a set of federated nodes.
            labels: The target labels.
            **kwargs: Optional named arguments. These can be passed
                when invoking the class method
                [get_nodes_federation](./#get_nodes_federation).

        # Returns:
            nodes_federation: A list-like object containing the data
                for each client.
            federated_label: A list-like object containing the target labels
                for each client.

        # Example:
            See implementation of the
            [IID data distribution class](./#iiddatadistribution-class).
        """
