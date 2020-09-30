import numpy as np
import abc


class Query(abc.ABC):
    """
    This class represents a query over private data. This interface exposes a method receiving
    data and must return a result based on this input.
    """

    @abc.abstractmethod
    def get(self, data):
        """
        Receives data and apply some function to answer it.

        # Arguments:
            data: Data to process

        # Returns:
            answer: Result of apply query over data
        """


class IdentityFunction(Query):
    """
    This function doesn't transform data. The answer is the data.
    """
    def get(self, data):
        return data


class Mean(Query):
    """
    Implements mean over data array.
    """
    def get(self, data):
        return np.mean(data)

    
class CheckDataType(Query):
    """
    It assesses and returns the type of data: either int/float/ndarray or a list of them.  
    If the data is in a different format, it throws a ValueError exception with the appropiate message

    # Arguments:
        data: input data

    # Returns:
        is_scalar: True if data is a scalar int/float
        is_array: True if data is ndarray
        is_list: True if data is a list (should contain int/float/ndarray)
    """
    def get(self, data):
        is_scalar, is_array, is_list = False, False, False
        
        if isinstance(data, (int, float)):
            is_scalar = True
        elif isinstance(data, np.ndarray):
            is_array = True
        elif isinstance(data, list):
            is_list = True
        else:
            raise ValueError(
                    "Data must be either int/float/ndarray or a list of them.")
        return is_scalar, is_array, is_list