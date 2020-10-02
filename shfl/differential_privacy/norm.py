import numpy as np
import abc
from shfl.private.query import CheckDataType


class SensitivityNorm(abc.ABC):
    """
    This class defines the interface that must be implemented to compute
    the sensitivity norm between two values in a normed space.
    
    # Arguments:
        axis: direction. Options are axis=None that considers all elements 
            and thus returns a scalar value for each array (default). 
            Instead, axis=0 operates along vertical axis and thus returns a vector of 
            size equal to the number of columns of each array
            (see [numpy.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html))
    """
    def __init__(self, axis=None):
        self._axis = axis
        
    def compute(self, x_1, x_2):
        """
        The compute method receives the result of apply a certain function over private data and
        returns the norm of the responses. The inputs can be scalars, arrays, or lists of arrays.

        # Arguments:
            x_1: response from a concrete query over database 1
            x_2: response from the same query over database 2
        """
        is_scalar, is_array, is_list = CheckDataType.get(x_1)
        if is_list:
            x = [self.norm(xi_1, xi_2) 
                 for xi_1, xi_2 in zip(x_1, x_2)]
        else: 
            x = self.norm(x_1, x_2)
            
        return x
    
    @abc.abstractmethod
    def norm(self, x_1, x_2):
        """
        Implements the norm of the difference between x_1 and x_2.

        # Arguments:
            x_1: scalar or array
            x_2: scalar or array
        """

        
class L1SensitivityNorm(SensitivityNorm):
    """
    Implements the L1 norm of the difference between x_1 and x_2
    """
    def norm(self, x_1, x_2):
        """L1 norm over scalars and arrays."""
        return np.sum(np.abs(x_1 - x_2), axis=self._axis)


class L2SensitivityNorm(SensitivityNorm):
    """
    Implements the L2 norm of the difference between x_1 and x_2
    """
    def norm(self, x_1, x_2):
        """L2 norm over scalars and arrays."""
        return np.sqrt(np.sum((x_1 - x_2)**2, axis=self._axis))
