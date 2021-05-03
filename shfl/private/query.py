import abc
import numpy as np


class Query(abc.ABC):
    """Queries the private data.

    Abstract method.

    This interface exposes a method that receives the node's private data
    and must return a result based on this input.
    """

    @abc.abstractmethod
    def get(self, data):
        """Receives data and applies and arbitrary query on it.

        # Arguments:
            data: Data to process.

        # Returns:
            answer: Result from applying the query over the input data.
        """


class IdentityFunction(Query):
    """Returns the data.
    """
    def get(self, data):
        return data


class Mean(Query):
    """Computes the mean over data.
    """
    def get(self, data):
        return np.mean(data)
