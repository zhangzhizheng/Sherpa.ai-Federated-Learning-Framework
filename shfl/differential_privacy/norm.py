import numpy as np
import abc


class SensitivityNorm(abc.ABC):
    """
    This class defines the interface that must be implemented to compute the sensitivity norm between
    two values in a normed space.
    """

    @abc.abstractmethod
    def compute(self, x_1, x_2):
        """
        The compute method receives the result of apply a certain function over private data and
        returns the norm of the responses

        # Arguments:
            x_1: array response from a concrete query over database 1
            x_2: array response from the same query over database 2
        """


class L1SensitivityNorm(SensitivityNorm):
    """
    Implements the L1 norm of the difference between x_1 and x_2
    """
    def compute(self, x_1, x_2):
        x = x_1 - x_2
        return np.sum(np.abs(x))


class L2SensitivityNorm(SensitivityNorm):
    """
    Implements the L2 norm of the difference between x_1 and x_2
    """
    def compute(self, x_1, x_2):
        x = x_1 - x_2
        return np.sqrt(np.sum(x**2))

    
class L1SensitivityNormList(SensitivityNorm):
    """
    Implements the L1 norm of the difference between x_1 and x_2 for lists of arrays
    
    # Arguments:
        axis: direction. Options are axis=None that considers all elements 
            and thus returns a scalar value for each array in the list (default). 
            Instead, axis=0 operates along vertical axis and thus returns a vector of 
            size equal to the number of columns of each array in the list 
            (see [numpy.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html))
    """
    def __init__(self, axis=None):
        self._axis = axis
        
    def compute(self, x_1, x_2):
        x = [np.sum(np.abs(xi_1 - xi_2), axis=self._axis) 
             for xi_1, xi_2 in zip(x_1, x_2)]
        return x
    
    
class L2SensitivityNormList(SensitivityNorm):
    """
    Implements the L2 norm of the difference between x_1 and x_2 for lists of arrays
    
    # Arguments:
        axis: direction. Options are axis=None that considers all elements 
            and thus returns a scalar value for each array in the list (default). 
            Instead, axis=0 operates along vertical axis and thus returns a vector of 
            size equal to the number of columns of each array in the list 
            (see [numpy.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html))
    """
    def __init__(self, axis=None):
        self._axis = axis
        
    def compute(self, x_1, x_2):
        x = [np.sqrt(np.sum((xi_1 - xi_2)**2, axis=self._axis))
             for xi_1, xi_2 in zip(x_1, x_2)]
        return x