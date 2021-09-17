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


class TrainableModel(abc.ABC):
    """Wraps a machine learning model.

    Allows an arbitrary model to interact with the
    federated learning framework. It is useful when
    you want to use a custom model that is not already
    provided by the framework.

    # Example:
        See the [notebook on wrapping a custom model
        in the Framework](https://github.com/sherpaai/
        Sherpa.ai-Federated-Learning-Framework/
        blob/master/notebooks/federated_models/
        federated_models_custom_model.ipynb).
    """

    @abc.abstractmethod
    def train(self, data, labels, **kwargs):
        """Trains the model.

        Abstract method.

        # Arguments:
            data: Data to train the model.
            labels: Target labels.
            **kwargs: Optional named parameters.
        """

    @abc.abstractmethod
    def predict(self, data):
        """Makes a prediction on input data.

        Abstract method.

        # Arguments:
            data: The input data on which to make the prediction.

        # Returns:
            prediction: Model's prediction using the input data.
        """

    @abc.abstractmethod
    def evaluate(self, data, labels):
        """Evaluates the performance of the model.

        Abstract method.

        # Arguments:
            data: The data on which to make the evaluation.
            labels: The true labels.

        # Returns:
            metrics: Metrics for the evaluation.
        """

    @abc.abstractmethod
    def get_model_params(self):
        """Gets model's parameters.

        Abstract method.

        # Returns:
            params: Parameters defining the model.
        """

    @abc.abstractmethod
    def set_model_params(self, params):
        """Sets model's parameters.

        Abstract method.

        # Arguments:
            params: Parameters defining the model.
        """

    @abc.abstractmethod
    def performance(self, data, labels):
        """Evaluates the performance of the model using
            the most representative metrics.

        Abstract method.

        # Arguments:
            data: The data on which to make the evaluation.
            labels: The true labels.

        # Returns:
            metrics: Most representative metrics for the evaluation.
        """
