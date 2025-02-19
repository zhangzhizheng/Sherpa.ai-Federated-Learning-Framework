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
import numpy as np


class LabeledData:
    """Represents labeled data.

    # Arguments:
        data: Object containing data.
        labels: Object containing target labels.

    # Properties:
        data: Getter and setter for data.
        label: Getter and setter for the target labels.
    """
    def __init__(self, data, label):
        self._data = data
        self._label = label

    @property
    def data(self):
        """Returns data."""
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def label(self):
        """Returns target labels."""
        return self._label

    @label.setter
    def label(self, label):
        self._label = label


class DPDataAccessDefinition:
    """Interface defining a differentially private query
        to the node's private properties.

    In order to define a differential privacy mechanism, this class has to
    be implemented.

    # Examples:
        See the implementation of the
        [available differential privacy mechanisms](../../differential_privacy/mechanisms/).
    """

    @property
    @abc.abstractmethod
    def epsilon_delta(self):
        """Every differentially private mechanism must implement this property.

        Abstract method.

        # Returns:
            epsilon_delta: Privacy budget spent each time this
                differentially private mechanism is used.
        """

    @abc.abstractmethod
    def __call__(self, data, **kwargs):
        """Applies an arbitrary query on the node's private property.

        Abstract method.

        It must be implemented in order to define how
        to query a node's private property.

        # Arguments:
            data: Node's private data to be accessed.

        # Returns:
            result_data: Result from the query on the node's private data.
        """

    @staticmethod
    def _check_epsilon_delta(epsilon_delta):
        """Checks whether epsilon and delta are valid.

        If the check fails, a ValueError exception is raised
        with the appropriate message.

        # Arguments:
            epsilon_delta: A tuple of values corresponding to the
                epsilon and delta parameters the differential privacy mechanism.
        """
        if len(epsilon_delta) != 2:
            raise ValueError("epsilon_delta parameter should be a tuple with "
                             "two elements, but {} were given."
                             .format(len(epsilon_delta)))
        if epsilon_delta[0] < 0:
            raise ValueError("Epsilon has to be greater than zero.")
        if epsilon_delta[1] < 0:
            raise ValueError("Delta has to be greater than 0 and less than 1.")

    @staticmethod
    def _check_binary_data(data):
        """Checks whether data is binary.

        If the check fails, a ValueError exception is raised
        with the appropriate message.

        # Arguments:
            data: Input data.

        """
        if not np.array_equal(data, data.astype(bool)):
            raise ValueError(
                "This mechanism works with binary data, "
                "but input is not binary.")

    @staticmethod
    def _check_sensitivity_positive(sensitivity):
        """Checks whether given sensitivity is strictly positive.

        If the check fails, a ValueError exception is raised
        with the appropriate message.

        # Arguments:
            sensitivity: Array-like object containing sensitivity values.
        """
        if isinstance(sensitivity, (np.ScalarType, np.ndarray)):
            sensitivity = np.asarray(sensitivity)
            if (sensitivity < 0).any():
                raise ValueError(
                    "Sensitivity of the query cannot be negative.")

    @staticmethod
    def _check_sensitivity_shape(sensitivity, query_result):
        """Checks whether given sensitivity shape matches the query result.

        If the check fails, a ValueError exception is raised
        with the appropriate message.

        # Arguments:
            sensitivity: Array-like object containing sensitivity values.
            query_result: Array-like object containing the query's result.
        """
        if sensitivity.size > 1:
            if sensitivity.size > query_result.size:
                raise ValueError(
                    "Provided more sensitivity values than query outputs.")
            if not all((m == n) for m, n in zip(sensitivity.shape[::-1],
                                                query_result.shape[::-1])):
                raise ValueError("Sensitivity array dimension " +
                                 str(sensitivity.shape) +
                                 " cannot broadcast to query result dimension " +
                                 str(query_result.shape) + ".")
