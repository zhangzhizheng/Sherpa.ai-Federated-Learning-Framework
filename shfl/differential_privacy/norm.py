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

# In this case, only one method is needed
# pylint: disable=too-few-public-methods
import abc
import numpy as np
from multipledispatch import dispatch


class SensitivityNorm(abc.ABC):
    """Implements the norm of the difference between two inputs.

    This interface must be implemented in order to define
    a norm of the difference between two inputs in a normed space.

    # Arguments:
        axis: Optional; Direction on which to compute the norm.
            Options are axis=None (default) that considers all elements
            and thus returns a scalar value for each array (default).
            Instead, axis=0 operates along vertical axis and thus
            returns a vector of size equal to the number of columns of each array
            (see [numpy.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)).
    """

    def __init__(self, axis=None):
        self._axis = axis

    @abc.abstractmethod
    def compute(self, x_1, x_2):
        """Computes the norm of the difference between two inputs.

        Abstract method.

        # Arguments:
            x_1: Response from a specific query over database 1.
            x_2: Response from a specific query over database 2.
        """

    @dispatch(list, list)
    def _norm(self, x_1, x_2):
        """L1 norm of the difference between (nested) lists of arrays."""
        norm = [self.compute(xi_1, xi_2)
                for xi_1, xi_2 in zip(x_1, x_2)]
        return norm

    @dispatch(tuple, tuple)
    def _norm(self, x_1, x_2):
        """L1 norm of the difference between (nested) tuples of arrays."""
        norm = tuple(self.compute(xi_1, xi_2)
                     for xi_1, xi_2 in zip(x_1, x_2))
        return norm


class L1SensitivityNorm(SensitivityNorm):
    """Computes the L1 norm of the difference between x_1 and x_2.

    It implements the class [SensitivityNorm](./#sensitivitynorm-class).
    """

    def compute(self, x_1, x_2):
        """See base class.
        """
        return self._norm(x_1, x_2)

    @dispatch((np.ScalarType, np.ndarray), (np.ScalarType, np.ndarray))
    def _norm(self, x_1, x_2):
        """L1 norm of the difference between arrays."""
        norm = np.sum(np.abs(x_1 - x_2), axis=self._axis)
        return norm

    @dispatch((list, tuple), (list, tuple))
    def _norm(self, x_1, x_2):
        return super()._norm(x_1, x_2)


class L2SensitivityNorm(SensitivityNorm):
    """Computes the L2 norm of the difference between x_1 and x_2.

    It implements the class [SensitivityNorm](./#sensitivitynorm-class).
    """

    def compute(self, x_1, x_2):
        """See base class.
        """
        return self._norm(x_1, x_2)

    @dispatch((np.ScalarType, np.ndarray), (np.ScalarType, np.ndarray))
    def _norm(self, x_1, x_2):
        """L2 norm of difference between arrays"""
        return np.sqrt(np.sum((x_1 - x_2) ** 2, axis=self._axis))

    @dispatch((list, tuple), (list, tuple))
    def _norm(self, x_1, x_2):
        return super()._norm(x_1, x_2)
