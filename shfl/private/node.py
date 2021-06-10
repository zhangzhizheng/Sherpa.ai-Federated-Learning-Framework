import copy

from shfl.private.data import UnprotectedAccess


class DataNode:
    """Represents an independent data node.

    Typically, a data node possesses its own private data and provides methods.
    The access to the private data must be configured with
    an access policy before querying it or an exception will be raised.
    A method to transform the node's private data is also provided,
    allowing data preprocessing and similar tasks.

    A model can be deployed in the
    node to learn from its private data (see class [Model](../../model)).
    The access to the model must be also configured
    before making queries.

    # Properties:
        model: Access to the model.
        private_data: Access to private train data.
        private_data_test: Access to private test data.
    """

    def __init__(self):
        self._private_data = {}
        self._private_test_data = {}
        self._private_data_access_policies = {}
        self._model = None
        self.configure_model_params_access(UnprotectedAccess())
        self._model_access_policy = None

    @property
    def model(self):
        """Allows to see the model for this node, but not to retrieve it.
        """
        print("You can't get the model, you need to query the params to access.")
        print(type(self._model))
        print(self._model)

    @property
    def private_data(self):
        """Allows to see train data for this node, but not to retrieve it.
        """
        print("You can see the node's private train data for debug purposes, "
              "but the data remains in the node.")
        print(type(self._private_data))
        print(self._private_data)

    @property
    def private_test_data(self):
        """Allows to see test data for this node, but not to retrieve it.
        """
        print("You can see the node's private test data for debug purposes, "
              "but the data remains in the node.")
        print(type(self._private_test_data))
        print(self._private_test_data)

    def set_model(self, model):
        """Sets the model to use in the node.

        # Arguments:
            model: Instance of a class implementing ~TrainableModel.
        """
        self._model = copy.deepcopy(model)

    def set_private_data(self, name, data):
        """Copies the data in private memory.

        The name is used as key. If there is a previous value with this key the
        data will be overwritten.

        # Arguments:
            name: String identifying the private data.
            data: Data to be stored in the private memory of the data node.
        """
        self._private_data[name] = copy.deepcopy(data)

    def set_private_test_data(self, name, data):
        """Copies the test data in private memory.

        The name is used as key. If there is a previous value with this key the
        data will be overwritten.

        # Arguments:
            name: String identifying the private data.
            data: Data to be stored in the private memory of the data node.
        """
        self._private_test_data[name] = copy.deepcopy(data)

    def configure_data_access(self, name, data_access_definition):
        """Sets the access policy for the specific private data.

        By default, the access to the node's data is protected.
        The access definition can be changed using this method.

        # Arguments:
            name: String identifying the private data.
            data_access_definition: Policy that specifies how to access the data
                (see: [DataAccessDefinition](../data/#dataaccessdefinition-class)).

        # Example:
            In order to return raw private data, unprotected access must
            be set (see [UnprotectedAccess](../data/#unprotectedaccess-class)):

            ```{python}
            import numpy as np

             from shfl.private.node import DataNode
             from shfl.private.data import UnprotectedAccess
             from shfl.private.data import LabeledData

             data = np.array([[1,2,3], [4,5,6]])
             labels = np.array([1,0,1])

             node = DataNode()
             node.set_private_data(name="private_data1", data=LabeledData(data, labels))

             # Raises "ValueError: Data access must be configured before querying the data.":
             node.query("private_data1").data

             # After setting access type, returns private data and labels:
             node.configure_data_access("private_data1", UnprotectedAccess())
             node.query("private_data1").data
             node.query("private_data1").label
             ```
        """
        self._private_data_access_policies[name] = copy.deepcopy(data_access_definition)

    def configure_model_params_access(self, data_access_definition):
        """Sets the access policy for the model's parameters.

        By default, the access to the node's model's parameters is **not** protected.
        The access definition can be changed using this method.

        # Arguments:
            data_access_definition: Policy that specifies how to access the model's parameters \
            (see: [DataAccessDefinition](../data/#dataaccessdefinition-class)).
        """
        self._model_params_access_policy = copy.deepcopy(data_access_definition)

    def configure_model_access(self, data_access_definition):
        """Sets the access policy for the queries to the model.

        By default, the access to the node's model is protected.
        The access definition can be changed using this method.

        # Arguments:
            data_access_definition: Policy that specifies how to access the model \
            (see: [DataAccessDefinition](../data/#dataaccessdefinition-class)).

        # Example:
            Let's suppose we want to access the method `get_meta_params` of the node's  model.
            In this case we would define an access to the node's __model__
            (Note that this time the private property is the node's model, which will be
            passed as argument):

            ```{Python}
            import numpy as np

            from shfl.private.node import DataNode
            from shfl.private.data import DataAccessDefinition


            class QueryMetaParameters(DataAccessDefinition):
                def apply(self, model, **kwargs):
                    return model.get_meta_params(**kwargs)

            node = DataNode()
            node.configure_model_access(QueryMetaParameters())
            node.query_model()
            ```
        """
        self._model_access_policy = copy.deepcopy(data_access_definition)

    def apply_data_transformation(self, private_property, federated_transformation):
        """Applies a transformation over the private data.

        # Arguments:
            private_property: String identifying the private data.
            federated_transformation: Transformation to apply
                (see: [Federated Operation](../federated_operation)).
        """
        federated_transformation.apply(self._private_data[private_property])

    def query(self, private_property, **kwargs):
        """Queries the private data.

        If the access has not been previously configured,
        an exception will be raised.

        # Arguments:
            private_property: String identifying the private data.
            **kwargs: Optional named parameters.

        # Returns:
            result: Result from the query.
        """
        if private_property not in self._private_data_access_policies:
            raise ValueError("Data access must be configured before "
                             "querying the data.")

        data_access_policy = self._private_data_access_policies[private_property]
        return data_access_policy.apply(self._private_data[private_property], **kwargs)

    def query_model_params(self):
        """Queries model's parameters.

        # Returns:
            params: Parameters defining the model.
        """
        return self._model_params_access_policy.apply(self._model.get_model_params())

    def query_model(self, **kwargs):
        """Queries the model.

        If the access to the model has not been previously configured,
        an exception will be raised.

        # Arguments:
            **kwargs: Optional named parameters.

        # Returns:
            result: Result from the query.
        """
        if self._model_access_policy is None:
            raise ValueError("By default, the model cannot be accessed. "
                             "You need to define a model access policy first.")

        return self._model_access_policy.apply(self._model, **kwargs)

    def set_model_params(self, model_params):
        """Sets the parameters of the model used in the node.

        # Arguments:
            model_params: Parameters of the model.
        """
        self._model.set_model_params(copy.deepcopy(model_params))

    def train_model(self, training_data_key, **kwargs):
        """Trains node's model.

        # Arguments:
            training_data_key: String identifying the private data from which
                the model will learn. The private data must be of
                class LabeledData (see: [LabeledData](../data/#labeleddata)).
            **kwargs: Optional named parameters.
        """
        labeled_data = self._private_data.get(training_data_key)
        if not hasattr(labeled_data, 'data') or not hasattr(labeled_data, 'label'):
            raise ValueError("Private data needs to have 'data' and 'label' to train a model")
        self._model.train(labeled_data.data, labeled_data.label, **kwargs)

    def predict(self, data):
        """Makes a prediction on input data using the node's model.

        # Arguments:
            data: The input data on which to make the prediction.

        # Returns:
            prediction: The node's model prediction using the input data.
        """
        return self._model.predict(data)

    def evaluate(self, data, labels):
        """Evaluates the performance of the node's model.

        # Arguments:
            data: The data on which to make the evaluation.
            labels: The true labels.

        # Returns:
            metrics: Metrics for the evaluation.
        """
        return self._model.evaluate(data, labels)

    def performance(self, data, labels):
        """Evaluates the performance of the node's model using
            the most representative metrics.

        # Arguments:
            data: The data on which to make the evaluation.
            labels: The true labels.

        # Returns:
            metrics: Most representative metrics for the evaluation.
        """
        return self._model.performance(data, labels)

    def local_evaluate(self, data_key):
        """Evaluates the performance of the node's model on local test data.

        # Arguments:
            data_key: String identifying the private data.

        # Returns:
            metrics: Metrics for the evaluation. If local test data
                is not present on the node, returns None.
        """
        metrics = None
        if bool(self._private_test_data):
            labeled_data = self._private_test_data.get(data_key)
            metrics = self._model.evaluate(labeled_data.data, labeled_data.label)

        return metrics
