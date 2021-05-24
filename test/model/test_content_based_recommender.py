from unittest.mock import Mock
import pytest
import pandas as pd
import numpy as np

from shfl.model.content_based_recommender import ContentBasedRecommender


@pytest.fixture(name="df_items")
def fixture_data_frame_items():
    """Returns a data frame containing the catalogue of available items."""
    num_items = 10
    num_features = 3
    items_catalogue = pd.DataFrame(
        np.random.randint(low=1, high=100, size=(num_items, num_features)))
    items_catalogue.columns = ["Feature_" + str(i) for i in range(num_features)]
    items_catalogue.index.name = "item_id"

    return items_catalogue


@pytest.fixture(name="data_and_labels")
def fixture_data_and_labels():
    """Returns random data features and rating labels."""
    data = np.array([[2, 1],
                     [2, 1],
                     [2, 2],
                     [2, 2],
                     [2, 0]])
    labels = np.array([3, 2, 5, 6, 7])

    return data, labels


def test_initialization(df_items):
    """Checks that the content based recommender is properly initialized."""
    recommender = ContentBasedRecommender(df_items)

    assert hasattr(recommender, "_client_identifier")
    assert hasattr(recommender, "_mean_rating")
    assert hasattr(recommender, "_profile")


def test_initialization_wrong_input(df_items):
    """Checks that the content base recommender raises an error
    if initialized with wrong input.

    The input must be a dataframe."""
    array_items = df_items.to_numpy()

    with pytest.raises(TypeError):
        ContentBasedRecommender(array_items)


def test_train(df_items, data_and_labels):
    """Checks that the content base recommender trains correctly."""
    data, labels = data_and_labels
    recommender = ContentBasedRecommender(df_items)

    recommender.train(data, labels)

    df_data = pd.DataFrame(data, columns=["user_id", "item_id"])
    df_joined = df_data.join(df_items, on="item_id").drop(["user_id", "item_id"], axis=1)
    mean_rating, profile = recommender.get_model_params()
    assert mean_rating == np.mean(labels)
    np.testing.assert_equal(
        profile,
        df_joined.multiply(labels - np.mean(labels), axis=0).mean().values)


def test_train_wrong_number_of_columns(df_items, data_and_labels):
    """Checks that the content base recommender raises an error
    if trained with wrong input.

    The input data must contain only 2 columns containing the user id and item id."""
    data = np.array([[2, 3, 51],
                     [2, 34, 6],
                     [2, 33, 7],
                     [2, 13, 65],
                     [2, 3, 15]])
    _, labels = data_and_labels
    recommender = ContentBasedRecommender(df_items)

    with pytest.raises(AssertionError):
        recommender.train(data, labels)


def test_train_wrong_items(df_items, data_and_labels):
    """Checks that an error is raised when items different from the catalogue are provided.

    There are N items in the catalogue, so valid item ids are in the range (0, N-1)."""
    data = np.array([[2, 50],
                     [2, 49],
                     [2, 33],
                     [2, 13],
                     [2, 3]])
    _, labels = data_and_labels
    recommender = ContentBasedRecommender(df_items)

    with pytest.raises(AssertionError):
        recommender.train(data, labels)


def test_predict(df_items, data_and_labels):
    """Checks that the content base recommender predicts correctly."""
    data, _ = data_and_labels
    recommender = ContentBasedRecommender(df_items)
    mean_rating, profile = (1, np.random.rand(df_items.shape[1]))
    recommender.set_model_params((mean_rating, profile))

    predictions = recommender.predict(data)

    joined_data = pd.DataFrame(data, columns=['userid', "item_id"]). \
        join(df_items, on="item_id"). \
        drop(["userid", "item_id"], axis=1)
    predictions_test = mean_rating + joined_data.values.dot(profile)

    np.testing.assert_equal(predictions, predictions_test)


def test_evaluate(df_items, data_and_labels):
    """Checks that the content base recommender evaluates correctly."""
    data, labels = data_and_labels
    recommender = ContentBasedRecommender(df_items)
    mean_rating, profile = (1, np.random.rand(df_items.shape[1]))
    recommender.set_model_params((mean_rating, profile))
    evaluation = recommender.evaluate(data, labels)

    predictions = recommender.predict(data)
    true_evaluation = np.sqrt(np.mean((predictions - labels) ** 2))

    assert evaluation == true_evaluation


def test_evaluate_no_data(df_items):
    """Checks that the evaluation is zero when no data is provided."""
    data = np.empty((0, 2))
    labels = np.empty(0)

    recommender = ContentBasedRecommender(df_items)
    mean_rating, profile = (1, np.random.rand(df_items.shape[1]))
    recommender.set_model_params((mean_rating, profile))
    evaluation = recommender.evaluate(data, labels)

    assert evaluation == 0


def test_performance(df_items, data_and_labels):
    """Check that the content based recommender correctly calls the performance."""
    data, labels = data_and_labels
    recommender = ContentBasedRecommender(df_items)
    recommender.evaluate = Mock()

    recommender.performance(data, labels)

    recommender.evaluate.assert_called_once_with(data, labels)


def test_model_params(df_items):
    """Checks that the content base recommender correctly sets and gets the model's parameters."""
    recommender = ContentBasedRecommender(df_items)
    mean_rating, profile = (1, np.random.rand(df_items.shape[1]))

    recommender.set_model_params((mean_rating, profile))
    params = recommender.get_model_params()

    np.testing.assert_equal(mean_rating, params[0])
    np.testing.assert_equal(profile, params[1])
