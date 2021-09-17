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

# Disable too many arguments and locals: needed in this case
# pylint: disable=too-many-arguments, too-many-locals
import random
import numpy as np
import tensorflow as tf

from shfl.data_base.data_base import shuffle_rows
from shfl.data_distribution.data_distribution_sampling import SamplingDataDistribution


class NonIidDataDistribution(SamplingDataDistribution):
    """Creates a set of federated nodes from a centralized database.

    Implements the class
    [Data Distribution](../data_distribution/#datadistribution-class).

    A non-independent and identically distribution is used.
    In the present scenario the clients posses only a part of
    the total classes.

    Note: This distribution only works with classification problems.
    """

    def make_data_federated(self, data, labels, percent=100, num_nodes=1,
                            weights=None, sampling="with_replacement"):
        """See base class.
        """

        if weights is None:
            weights = np.full(num_nodes, 1/num_nodes)

        # Check label's format
        if labels.ndim == 1:
            one_hot = False
            labels = tf.keras.utils.to_categorical(labels)
        else:
            one_hot = True

        data, labels = shuffle_rows(data, labels)

        # Select percent
        data = data[0:int(percent * len(data) / 100)]
        labels = labels[0:int(percent * len(labels) / 100)]

        num_data = len(data)

        # We generate random classes for each client
        total_labels = np.unique(labels.argmax(axis=-1))
        random_classes = self._choose_labels(num_nodes, len(total_labels))

        federated_data = []
        federated_label = []

        if sampling == "with_replacement":
            for i in range(0, num_nodes):
                labels_to_use = random_classes[i]

                idx = np.array([i in labels_to_use
                                for i in labels.argmax(axis=-1)])
                data_aux = data[idx]
                labels_aux = labels[idx]

                data_aux, labels_aux = shuffle_rows(data_aux, labels_aux)

                percent_per_client = min(int(weights[i]*num_data),
                                         len(data_aux))

                federated_data.append(
                    np.array(data_aux[0:percent_per_client, ]))
                federated_label.append(
                    np.array(labels_aux[0:percent_per_client, ]))

        else:
            if sum(weights) > 1:
                weights = np.array([float(i) / sum(weights) for i in weights])

            for i in range(0, num_nodes):
                labels_to_use = random_classes[i]

                idx = np.array([i in labels_to_use
                                for i in labels.argmax(axis=-1)])
                data_aux = data[idx]
                rest_data = data[~idx]
                labels_aux = labels[idx]
                rest_labels = labels[~idx]

                data_aux, labels_aux = shuffle_rows(data_aux, labels_aux)

                percent_per_client = min(int(weights[i] * num_data),
                                         len(data_aux))

                federated_data.append(
                    np.array(data_aux[0:percent_per_client, ]))
                rest_data = np.append(
                    rest_data, data_aux[percent_per_client:, ], axis=0)
                federated_label.append(
                    np.array(labels_aux[0:percent_per_client, ]))
                rest_labels = np.append(
                    rest_labels, labels_aux[percent_per_client:, ], axis=0)

                data = rest_data
                labels = rest_labels

        if not one_hot:
            federated_label = [np.argmax(node, 1)
                               for node in federated_label]

        return federated_data, federated_label

    @staticmethod
    def _choose_labels(num_nodes, total_labels):
        """Randomly chooses the labels used for each client.

        # Arguments:
            num_nodes: Number of nodes.
            total_labels: Number of labels.

        # Returns:
            labels_to_use: The labels to use for each node.
        """

        random_labels = []

        for _ in range(0, num_nodes):
            num_labels = random.randint(2, total_labels)
            labels_to_use = []

            for _ in range(num_labels):
                label = random.randint(0, total_labels - 1)
                if label not in labels_to_use:
                    labels_to_use.append(label)
                else:
                    while label in labels_to_use:
                        label = random.randint(0, total_labels - 1)
                    labels_to_use.append(label)

            random_labels.append(labels_to_use)

        return random_labels
