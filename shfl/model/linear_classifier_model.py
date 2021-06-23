import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from shfl.model.model import TrainableModel
from .utils import get_model_params
from .utils import set_model_params
from .utils import _check_data


class LinearClassifierModel(TrainableModel):
    """Wraps scikit-learn linear classification models.

    Implements the class [TrainableModel](../#trainablemodel-class).

    # Arguments:
        n_features: Number of features.
        classes: Array-like object containing the classes to predict.
            At least 2 classes must be provided.
        model: Optional; Sklearn Linear Model instance to use.
            It has been tested with Logistic Regression and
            Linear C-Support Vector Classification models,
            but it should work for every linear model defined by
            intercept_ and coef_ attributes (by default, a Logistic Regression
            model is used).

    # References:
        [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/
        stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

        [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/
        generated/sklearn.svm.SVC.html)
    """

    get_model_params = get_model_params
    set_model_params = set_model_params
    _check_data = _check_data

    def __init__(self, n_features, classes, model=None):
        if model is None:
            model = LogisticRegression(solver='lbfgs', multi_class='auto')
        self._check_initialization(n_features, classes)
        self._model = model
        self._n_features = n_features
        classes = np.sort(np.asarray(classes))
        self._model.classes_ = classes
        n_classes = len(classes)
        if n_classes == 2:
            n_classes = 1
        self.set_model_params([np.zeros(n_classes),
                               np.zeros((n_classes, n_features))])

    def train(self, data, labels, **kwargs):
        """Trains the model.

        # Arguments:
            data: Array-like object of shape (n_samples, n_features)
                containing the data to train the model.
            labels: Array-like object of shape (n_samples,)
                or (n_samples, n_classes) containing the target labels.
            **kwargs: Optional named parameters.
        """

        self._check_data(data)
        self._check_labels_train(labels)
        self._model.fit(data, labels)

    def predict(self, data):
        """Makes a prediction on input data.

        # Arguments:
            data: Array-like object of shape (n_samples, n_features)
                containing the input data on which to make the prediction.

        # Returns:
            prediction: Model's prediction using the input data.
        """
        self._check_data(data)

        return self._model.predict(data)

    def evaluate(self, data, labels):
        """Evaluates the performance of the model.

        # Arguments:
            data: Array-like object of shape (n_samples, n_features)
                containing the input data on which to make the evaluation.
            labels: Array-like object of shape (n_samples,) or
                (n_samples, n_classes) containing the target labels.

        # Returns:
            balanced_accuracy: [Balanced accuracy score](https://scikit-learn.org/
            stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).

            cohen_kappa: [Cohen's kappa score](https://scikit-learn.org/
            stable/modules/generated/sklearn.metrics.cohen_kappa_score.html).
        """

        self._check_labels_predict(labels)

        prediction = self.predict(data)
        balanced_accuracy = metrics.balanced_accuracy_score(labels, prediction)
        cohen_kappa = metrics.cohen_kappa_score(labels, prediction)

        return balanced_accuracy, cohen_kappa

    def performance(self, data, labels):
        """Evaluates the performance of the model using
            the most representative metrics.

        # Arguments:
            data: Array-like object of shape (n_samples, n_features)
                containing the input data on which to make the evaluation.
            labels: Array-like object of shape (n_samples,) or
                (n_samples, n_classes) containing the target labels.

        # Returns:
            balanced_accuracy: [Balanced accuracy score](https://scikit-learn.org/
            stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).
        """

        self._check_data(data)
        self._check_labels_predict(labels)

        prediction = self.predict(data)
        balanced_accuracy = metrics.balanced_accuracy_score(labels, prediction)

        return balanced_accuracy

    def _check_labels_train(self, labels):
        """Checks whether the classes to train are correct.

        When training, the classes in client's data must be the same
        as the input ones.
        """
        classes = np.unique(np.asarray(labels))
        if not np.array_equal(self._model.classes_, classes):
            raise AssertionError(
                "When training, labels need to have the same classes "
                "described by the model " + str(self._model.classes_) +
                ". Labels of this node are " + str(classes) + ".")

    def _check_labels_predict(self, labels):
        """Checks whether the classes to predict are correct.

        When predicting, the classes in data must be a subset
        of the trained ones.
        """
        classes = np.unique(np.asarray(labels))
        if not set(classes) <= set(self._model.classes_):
            raise AssertionError(
                "When predicting, labels need to be a subset of the classes "
                "described by the model " + str(self._model.classes_) +
                ". Labels in the given data are " + str(classes) + ".")

    @staticmethod
    def _check_initialization(n_features, classes):
        """Checks whether the model's initialization is correct.

        The number of features must be an integer equal or greater to one,
        and there must be at least two classes.

        # Arguments:
            n_features: Number of features.
            classes: Array-like object containing target classes.
        """
        if not isinstance(n_features, int):
            raise AssertionError(
                "n_features must be a positive integer number. Provided " +
                str(n_features) + " features.")
        if n_features < 0:
            raise AssertionError(
                "It must verify that n_features > 0. Provided value " +
                str(n_features) + ".")
        if len(classes) < 2:
            raise AssertionError(
                "It must verify that the number of classes > 1. Provided " +
                str(len(classes)) + " classes.")
        if len(np.unique(classes)) != len(classes):
            classes = list(classes)
            duplicated_classes = [i_class for i_class in classes
                                  if classes.count(i_class) > 1]
            raise AssertionError(
                "No duplicated classes allowed. Class(es) duplicated: " +
                str(duplicated_classes) + ".")
