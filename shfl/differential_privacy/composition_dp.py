from math import sqrt, log, exp

from shfl.private.data import DPDataAccessDefinition


class AdaptiveDifferentialPrivacy(DPDataAccessDefinition):
    """Defines adaptive differential privacy through privacy filters.

    It implements the class
    [DPDataAccessDefinition](../../private/data/#dpdataaccessdefinition-class).

    # Arguments:
        epsilon_delta: Tuple or array of length 2 containing
            the epsilon-delta privacy budget for this data.
        differentially_private_mechanism: Optional; The method to access data.
            If not set, it is mandatory to pass it in every query.

    # Properties:
        epsilon_delta: Returns the epsilon and delta values
            of the resulting mechanism.

    # Example:
        See the [notebook on adaptive differential privacy](https://github.com/
        sherpaai/Sherpa.ai-Federated-Learning-Framework/blob/master/
        notebooks/differential_privacy/differential_privacy_composition_concepts.ipynb).

    # References:
        [Privacy odometers and filters:
        Pay-as-you-go composition](https://arxiv.org/pdf/1605.08294.pdf)
    """

    def __init__(self, epsilon_delta, differentially_private_mechanism=None):
        self._check_epsilon_delta(epsilon_delta)

        self._epsilon_delta = epsilon_delta
        self._epsilon_delta_access_history = []
        self._private_data_epsilon_delta_access_history = []
        if differentially_private_mechanism is not None:
            _check_differentially_private_mechanism(
                differentially_private_mechanism)
        self._differentially_private_mechanism = \
            differentially_private_mechanism

    @property
    def epsilon_delta(self):
        return self._epsilon_delta

    def apply(self, data, differentially_private_mechanism=None):
        """Applies a differentially private mechanism
            if the privacy budget allows it.

        If the privacy budget is exceeded an exception is thrown.

        # Arguments:
            data: Input data which to be accessed with differential privacy.
            differentially_private_mechanism: The mechanism providing
                differential privacy.

        # Returns:
            result: Array-type object of same shape as the input
                containing the differentially-private randomized data.
        """
        differentially_private_mechanism_to_apply = \
            self._get_data_access_definition(differentially_private_mechanism)
        self._private_data_epsilon_delta_access_history.append(
            differentially_private_mechanism_to_apply.epsilon_delta)

        privacy_budget_exceeded = self.__basic_adaptive_comp_theorem()
        if 0 < self._epsilon_delta[1] < exp(-1):
            privacy_budget_exceeded &= self.__advanced_adaptive_comp_theorem()

        if privacy_budget_exceeded:
            self._private_data_epsilon_delta_access_history.pop()
            raise ExceededPrivacyBudgetError(epsilon_delta=self._epsilon_delta)

        return differentially_private_mechanism_to_apply.apply(data)

    def _get_data_access_definition(self, data_access_definition):
        """Checks and returns the provided differential privacy mechanism.

        It checks whether the given data access definition
        is differentially private. If None is provided, it
        ensures that the default data access definition is
        differentially private.

        # Arguments:
            data_access_definition: The mechanism to be checked.

        # Returns:
            The given data access definition
            or the default one given in the constructor.
        """
        if data_access_definition is not None:
            _check_differentially_private_mechanism(data_access_definition)
            return data_access_definition
        if self._differentially_private_mechanism is None:
            raise ValueError("Not data access definition provided "
                             "or default method established")
        return self._differentially_private_mechanism

    def __basic_adaptive_comp_theorem(self):
        """Implements the basic adaptive composition theorem.

        See theorem 3.6 in References.

        # Returns:
            True if the privacy budget if surpassed, False otherwise.
        """
        global_epsilon, global_delta = self._epsilon_delta
        epsilon_sum, delta_sum = \
            map(sum, zip(*self._private_data_epsilon_delta_access_history))
        return epsilon_sum > global_epsilon or delta_sum > global_delta

    def __advanced_adaptive_comp_theorem(self):
        """Implements the advance adaptive composition theorem.

        See theorem 5.1 in References.

        # Returns:
            True if the privacy budget if surpassed, False otherwise.
        """
        epsilon_history, delta_history = \
            zip(*self._private_data_epsilon_delta_access_history)
        global_epsilon, global_delta = self._epsilon_delta

        delta_sum = sum(delta_history)
        epsilon_squared_sum = sum(epsilon ** 2 for epsilon in epsilon_history)

        first_fraction = global_epsilon ** 2 / (28.04 * log(1 / global_delta))

        first_sum_epsilon = sum(eps * (exp(eps) - 1) * 0.5
                                for eps in epsilon_history)
        first_parentheses = epsilon_squared_sum + first_fraction
        second_parentheses = 2 + log(epsilon_squared_sum / first_fraction + 1)
        last_factor = log(2 / global_delta)

        privacy_loss_k = first_sum_epsilon + \
            sqrt(first_parentheses * second_parentheses * last_factor)

        return privacy_loss_k > global_epsilon or \
            delta_sum > (global_delta * 0.5)


def _check_differentially_private_mechanism(data_access_mechanism):
    """Checks whether the data access mechanism provides differential privacy.

    # Arguments:
        data_access_mechanism: The mechanism to be checked.
    """
    if not hasattr(data_access_mechanism, 'epsilon_delta'):
        raise ValueError("You can't access differentially private data "
                         "with a non differentially private mechanism.")


class ExceededPrivacyBudgetError(Exception):
    """Throws an exception when exceeding the maximum allowed privacy budget.

    When it is used, the data cannot be accessed anymore.

    # Arguments:
        epsilon_delta: The privacy budget (epsilon, delta)
            which has been surpassed.

    # Example:
        See the [notebook on adaptive differential privacy](https://github.com/
        sherpaai/Sherpa.ai-Federated-Learning-Framework/blob/master/
        notebooks/differential_privacy/differential_privacy_composition_concepts.ipynb).
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._epsilon_delta = None
        if kwargs:
            if "epsilon_delta" in kwargs:
                self._epsilon_delta = kwargs["epsilon_delta"]

    def __str__(self):
        return 'Error: Privacy Budget {} has been ' \
               'exceeded'.format(self._epsilon_delta)
