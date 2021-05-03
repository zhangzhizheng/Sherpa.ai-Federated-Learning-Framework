import numpy as np
import abc
from multipledispatch import dispatch


class SensitivityNorm(abc.ABC):
    """Implements the norm of the difference between two inputs.

    This interface must be implemented in order to define
    a norm of the difference between two inputs in a normed space.

    # Arguments:
        axis: Optional; Direction on which to compute the norm.
            Options are axis=None that considers all elements
            and thus returns a scalar value for each array (default).
            Instead, axis=0 operates along vertical axis and thus
            returns a vector of size equal to the number of columns of each array
            (see [numpy.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)).
    """

    def __init__(self, axis=None):
        self._axis = axis

    def compute(self, x_1, x_2):
        """Computes the norm of the difference between two inputs.

        Abstract method.

        # Arguments:
            x_1: Response from a specific query over database 1.
            x_2: Response from a specific query over database 2.
        """


class L1SensitivityNorm(SensitivityNorm):
    """Implements the L1 norm of the difference between x_1 and x_2.

    It implements the class [SensitivityNorm](./#sensitivitynorm-class).
    """
    @dispatch((np.ScalarType, np.ndarray), (np.ScalarType, np.ndarray))
    def compute(self, x_1, x_2):
        """L1 norm of the difference between arrays."""
        x = np.sum(np.abs(x_1 - x_2), axis=self._axis)
        return x

    @dispatch(list, list)
    def compute(self, x_1, x_2):
        """L1 norm of the difference between (nested) lists of arrays."""
        x = [self.compute(xi_1, xi_2)
             for xi_1, xi_2 in zip(x_1, x_2)]
        return x


class L2SensitivityNorm(SensitivityNorm):
    """Implements the L2 norm of the difference between x_1 and x_2.

    It implements the class [SensitivityNorm](./#sensitivitynorm-class).
    """

    @dispatch((np.ScalarType, np.ndarray), (np.ScalarType, np.ndarray))
    def compute(self, x_1, x_2):
        """"L2 norm of the difference between arrays."""
        x = np.sqrt(np.sum((x_1 - x_2)**2, axis=self._axis))
        return x

    @dispatch(list, list)
    def compute(self, x_1, x_2):
        """"L2 norm of the difference between (nested) lists of arrays."""
        x = [self.compute(xi_1, xi_2)
             for xi_1, xi_2 in zip(x_1, x_2)]
        return x
