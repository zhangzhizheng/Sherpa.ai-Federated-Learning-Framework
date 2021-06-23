# Helper common functions: they can access protected members of self
# pylint: disable=protected-access


def get_model_params(self):
    """Gets the linear model's parameters."""
    return self._model.intercept_, self._model.coef_


def set_model_params(self, params):
    """Sets the linear model's parameters."""
    self._model.intercept_ = params[0]
    self._model.coef_ = params[1]


def _check_data(self, data):
    """Checks the linear model's input data."""
    if data.ndim == 1:
        if self._n_features != 1:
            raise AssertionError(
                "Data need to have the same number of features "
                "described by the model " + str(self._n_features) +
                ". Current data have only 1 feature.")
    elif data.shape[1] != self._n_features:
        raise AssertionError(
            "Data need to have the same number of features "
            "described by the model " + str(self._n_features) +
            ". Current data has " + str(data.shape[1]) + " features.")


def _check_labels(self, labels):
    """Checks neural network model's input labels."""
    if labels.shape[1:] != self._in_out_sizes[1]:
        raise AssertionError(
            "Labels need to have the same shape described by the model " +
            str(self._in_out_sizes[1]) + ". Current labels have shape " +
            str(labels.shape[1:]) + ".")
