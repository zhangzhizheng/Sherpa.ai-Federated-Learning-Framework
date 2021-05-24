import abc
from math import log
from math import exp
from functools import reduce
from operator import mul
import numpy as np

from shfl.private.data import DPDataAccessDefinition


class Sampler(DPDataAccessDefinition):
    """Defines sub-sampling methods to amplify differential privacy.

    It implements the class
    [DPDataAccessDefinition](../../private/data/#dpdataaccessdefinition-class).

    # Arguments:
        dp_mechanism: The differential privacy mechanism to apply.

    # References:
        [Privacy amplification by subsampling:
        Tight analyses via couplings and divergences](https://arxiv.org/abs/1807.01647)
    """

    def __init__(self, dp_mechanism):
        self._dp_mechanism = dp_mechanism

    def apply(self, data, **kwargs):
        """Samples the input data and applies the differential privacy mechanism.

        # Arguments:
            data: Input data which to be accessed with differential privacy.

        # Returns:
            result: Array-type object of length equal to the sample size
                containing the differentially-private randomized data.
        """
        sampled_data = self.sample(data)
        return self._dp_mechanism.apply(sampled_data)

    @property
    def epsilon_delta(self):
        return self.epsilon_delta_reduction(self._dp_mechanism.epsilon_delta)

    @abc.abstractmethod
    def epsilon_delta_reduction(self, epsilon_delta):
        """Computes the new epsilon and delta.

         Abstract method.

        It receives epsilon and delta parameters from a
        differential privacy mechanism and computes the new
        (hopefully reduced) epsilon and delta.

        # Arguments:
            epsilon_delta: Privacy budget provided by a
                differential privacy mechanism.

        # Returns:
            new_epsilon_delta: New epsilon delta values.
        """

    @abc.abstractmethod
    def sample(self, data):
        """Samples over the input data.

         Abstract method.

        # Arguments:
            data: Raw data that to be sampled.

        # Returns:
            sampled_data: Sampled data.
        """


class SampleWithoutReplacement(Sampler):
    """Amplifies privacy by sampling without replacement.

    It implements the class [Sampler](./#sampler-class).

    See Theorem 9 in the references.
    Note that it samples the first dimension of an array-like object.

    # Arguments:
        dp_mechanism: The differential privacy mechanism to apply.
        sample_size: One dimensional size of the sample.
        data_shape: Shape of the input data.
    """

    def __init__(self, dp_mechanism, sample_size, data_shape):
        super().__init__(dp_mechanism)
        check_sample_size(sample_size, data_shape)
        self._dp_mechanism = dp_mechanism
        self._data_shape = data_shape
        self._sample_size = sample_size
        if len(self._data_shape) > 1:
            # Data with more than one dimension
            self._actual_sample_size = self._sample_size * \
                prod(self._data_shape[1:])
            self._data_shape = prod(self._data_shape)
        else:
            # One dimensional data
            self._actual_sample_size = self._sample_size
            self._data_shape = self._data_shape[0]

    def sample(self, data):
        """See base class.
        """
        return array_sampler.choice(data,
                                    size=self._sample_size,
                                    replace=False)

    def epsilon_delta_reduction(self, epsilon_delta):
        """See base class.
        """
        proportion = self._actual_sample_size / self._data_shape
        epsilon, delta = epsilon_delta

        new_epsilon = log(1 + proportion * (exp(epsilon) - 1))
        new_delta = proportion * delta

        return new_epsilon, new_delta


def check_sample_size(sample_size, data_size):
    """Checks that the sample size is smaller than the original.

    # Arguments:
        sample_size: One dimensional size of the sample.
        data_shape: Tuple, shape of the original data.
    """
    if sample_size > data_size[0]:
        raise ValueError("Sample size {} must be less than "
                         "data size: {}".format(sample_size, data_size))


def prod(iterable):
    """Multiplies all items together.
    """
    return reduce(mul, iterable, 1)


array_sampler = np.random.default_rng()
