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

import numpy as np


def check_data_features(model_features, data):
    """Checks the linear model's input data."""
    if data.ndim == 1:
        if model_features != 1:
            raise AssertionError(
                "Data need to have the same number of features "
                "described by the model " + str(model_features) +
                ". Current data have only 1 feature.")
    elif data.shape[1] != model_features:
        raise AssertionError(
            "Data need to have the same number of features "
            "described by the model " + str(model_features) +
            ". Current data has " + str(data.shape[1]) + " features.")


def check_target_size(model_n_targets, labels):
    """Checks the linear regression model's targets size."""
    if labels.ndim == 1:
        if model_n_targets != 1:
            raise AssertionError(
                "Labels need to have the same number of targets "
                "described by the model " + str(model_n_targets) +
                ". Current labels have only 1 target.")
    elif labels.shape[1] != model_n_targets:
        raise AssertionError(
            "Labels need to have the same number of targets "
            "described by the model " + str(model_n_targets) +
            ". Current labels have " + str(labels.shape[1]) +
            " targets.")


def check_labels_size(in_out_sizes, labels):
    """Checks neural network model's input labels."""
    if labels.shape[1:] != in_out_sizes[1]:
        raise AssertionError(
            "Labels need to have the same shape described by the model " +
            str(in_out_sizes[1]) + ". Current labels have shape " +
            str(labels.shape[1:]) + ".")


def check_initialization_regression(n_dimensions):
    """Checks whether the model's initialization is correct.

    The number of features and targets must be an integer
    equal or greater to one.

    # Arguments:
        n_rounds: Number of features or targets.
    """
    if not isinstance(n_dimensions, int):
        raise AssertionError(
            "n_features and n_targets must be a positive integer number. "
            "Provided value " + str(n_dimensions) + ".")
    if n_dimensions < 1:
        raise AssertionError(
            "n_features and n_targets must be equal or greater that 1. "
            "Provided value " + str(n_dimensions) + ".")


def check_initialization_classification(n_features, classes):
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


def check_data_recommender(data):
    """Checks whether the data belongs to a single user.

    # Arguments:
        data: Array-like object containing the data.
    """
    number_of_clients = len(np.unique(data[:, 0]))

    if number_of_clients > 1:
        raise AssertionError(
            "Data need to correspond to a single user. "
            "Current data includes "
            "{} clients.".format(number_of_clients))


def check_data_labels_recommender(data, labels):
    """Checks whether the data and the labels
    have matching dimensions.

    # Arguments:
        data: Data to train the model.
        labels: Target labels.
    """
    if len(data) != len(labels):
        raise AssertionError(
            "Data and labels do not have matching dimensions. "
            "Current data has {} rows and there are "
            "{} labels".format(len(data), len(labels)))
