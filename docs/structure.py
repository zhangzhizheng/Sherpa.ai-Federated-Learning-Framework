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

from shfl import differential_privacy
from shfl import private
from shfl import model
from shfl import data_base
from shfl import federated_government
from shfl import data_distribution
from shfl import federated_aggregator

# For each class to document, it is possible to:
# 1) Document only the class: [classA, classB, ...]
# 2) Document all its methods: [classA, (classB, "*")]
# 3) Choose which methods to document (methods listed as strings):
# [classA, (classB, ["method1", "method2", ...]), ...]
# 4) Choose which methods to document (methods listed as qualified names):
# [classA, (classB, [module.classB.method1, module.classB.method2, ...]), ...]

PAGES = [
    {
        'page': 'private/data_node.md',
        'classes': [
            private.node.DataNode
        ],
        'methods': [
            private.node.DataNode.set_private_data,
            private.node.DataNode.set_private_test_data,
            private.node.DataNode.configure_data_access,
            private.node.DataNode.configure_model_params_access,
            private.node.DataNode.configure_model_access,
            private.node.DataNode.apply_data_transformation,
            private.node.DataNode.query,
            private.node.DataNode.query_model_params,
            private.node.DataNode.query_model,
            private.node.DataNode.set_model_params,
            private.node.DataNode.train_model,
            private.node.DataNode.predict,
            private.node.DataNode.evaluate,
            private.node.DataNode.performance,
            private.node.DataNode.local_evaluate,
        ],
    },
    {
        'page': 'private/data.md',
        'classes': [
            private.data.LabeledData,
            private.data.DPDataAccessDefinition
        ]
    },
    {
        'page': 'private/federated_operation.md',
        'classes': [
            (private.federated_operation.NodesFederation, ["append_data_node",
                                                           "num_nodes"]),
            (private.federated_operation.FederatedDataNode, ['configure_data_access',
                                                             'set_private_data',
                                                             'set_private_test_data',
                                                             'train_model',
                                                             'apply_data_transformation',
                                                             'split_train_test']),
            (private.federated_operation.ServerDataNode, ["deploy_collaborative_model",
                                                          "evaluate_collaborative_model",
                                                          "aggregate_weights"]),
            (private.federated_operation.VerticalServerDataNode, ["predict_collaborative_model",
                                                                  "evaluate_collaborative_model",
                                                                  "aggregate_weights"])
        ],
        'functions': [
            private.federated_operation.federate_array,
            private.federated_operation.federate_list,
        ]
    },
    {
        'page': 'private/federated_attack.md',
        'classes': [
            (private.federated_attack.FederatedPoisoningDataAttack, ['__call__'])
        ]
    },
    {
        'page': 'private/reproducibility.md',
        'classes': [
            private.reproducibility.Reproducibility
        ],
        'methods': [
            private.reproducibility.Reproducibility.get_instance,
            private.reproducibility.Reproducibility.set_seed,
            private.reproducibility.Reproducibility.delete_instance
        ]
    },
    {
        'page': 'databases.md',
        'classes': [
            (data_base.data_base.DataBase, ['load_data']),
            (data_base.data_base.LabeledDatabase, ['load_data']),
            (data_base.data_base.WrapLabeledDatabase, ['load_data']),
            data_base.california_housing.CaliforniaHousing,
            data_base.cifar.Cifar10,
            data_base.cifar.Cifar100,
            data_base.emnist.Emnist,
            data_base.fashion_mnist.FashionMnist,
            data_base.federated_emnist.FederatedEmnist,
            data_base.iris.Iris,
            data_base.lfw.Lfw,
            data_base.purchase100.Purchase100
        ],
        'functions': [
            data_base.data_base.shuffle_rows,
            data_base.data_base.split_train_test,
            data_base.data_base.vertical_split,
        ]
    },
    {
        'page': 'data_distribution.md',
        'classes': [
            (data_distribution.data_distribution.DataDistribution, ["get_nodes_federation",
                                                                    "make_data_federated"]),
            (data_distribution.data_distribution_iid.IidDataDistribution, ["make_data_federated"]),
            data_distribution.data_distribution_sampling.SamplingDataDistribution,
            (data_distribution.data_distribution_non_iid.NonIidDataDistribution, ["make_data_federated",
                                                                                  "_choose_labels"]),

        ]
    },
    {
        'page': 'model.md',
        'classes': [
            (model.model.TrainableModel, ["train", "predict", "evaluate", "get_model_params", "set_model_params",
                                          'performance']),
        ]
    },
    {
        'page': 'model/supervised.md',
        'classes': [
            model.deep_learning_model.DeepLearningModel,
            model.deep_learning_model_pt.DeepLearningModelPyTorch,
            model.linear_regression_model.LinearRegressionModel,
            model.linear_classifier_model.LinearClassifierModel,
            model.vertical_deep_learning_model.VerticalNeuralNetClientModel,
            model.vertical_deep_learning_model.VerticalNeuralNetServerModel
        ]
    },
    {
        'page': 'model/unsupervised.md',
        'classes': [
            model.kmeans_model.KMeansModel,
        ]
    },
    {
        'page': 'model/recommender.md',
        'classes': [
            model.recommender.Recommender,
            model.mean_recommender.MeanRecommender,
            model.content_based_recommender.ContentBasedRecommender,
        ]
    },
    {
        'page': 'federated_aggregator.md',
        'classes': [
            (federated_aggregator.federated_aggregator.FederatedAggregator, ["__call__"]),
            federated_aggregator.fedavg_aggregator.FedAvgAggregator,
            federated_aggregator.fedsum_aggregator.FedSumAggregator,
            federated_aggregator.weighted_fedavg_aggregator.WeightedFedAggregator,
            (federated_aggregator.iowa_federated_aggregator.IowaFederatedAggregator, ['set_ponderation', 'q_function',
                                                                                      'get_ponderation_weights']),
            federated_aggregator.norm_clip_aggregators.NormClipAggregator,
            federated_aggregator.norm_clip_aggregators.CDPAggregator,
            federated_aggregator.norm_clip_aggregators.WeakDPAggregator,
            federated_aggregator.cluster_fedavg_aggregator.cluster_fed_avg_aggregator
        ]
    },
    {
        'page': 'federated_government.md',
        'classes': [
            (federated_government.federated_government.FederatedGovernment, ['run_rounds',
                                                                             'evaluate_clients']),
            (federated_government.vertical_federated_government.VerticalFederatedGovernment, ['run_rounds',
                                                                                              'evaluate_collaborative_model']),
            (federated_government.federated_images_classifier.FederatedImagesClassifier, ['run_rounds',
                                                                                          'model_builder']),
            federated_government.federated_images_classifier.ImagesDataBases,
            (federated_government.federated_linear_regression.FederatedLinearRegression, ['run_rounds']),
            federated_government.federated_linear_regression.LinearRegressionDataBases,
            (federated_government.federated_clustering.FederatedClustering, ['run_rounds']),
            federated_government.federated_clustering.ClusteringDataBases,
            (federated_government.iowa_federated_government.IowaFederatedGovernment, ['performance_clients'])
        ]
    },
    {
        'page': 'differential_privacy/mechanisms.md',
        'classes': [
            differential_privacy.mechanism.RandomizedResponseCoins,
            differential_privacy.mechanism.RandomizedResponseBinary,
            differential_privacy.mechanism.LaplaceMechanism,
            differential_privacy.mechanism.GaussianMechanism,
            differential_privacy.mechanism.ExponentialMechanism
        ],
    },
    {
        'page': 'differential_privacy/sensitivity_sampler.md',
        'classes': [
            (differential_privacy.sensitivity_sampler.SensitivitySampler, ["sample_sensitivity"]),
        ],
    },
    {
        'page': 'differential_privacy/norm.md',
        'classes': [
            (differential_privacy.norm.SensitivityNorm, ["compute"]),
            differential_privacy.norm.L1SensitivityNorm,
            differential_privacy.norm.L2SensitivityNorm
        ],
    },
    {
        'page': 'differential_privacy/probability_distribution.md',
        'classes': [
            (differential_privacy.probability_distribution.ProbabilityDistribution, ["sample"]),
            differential_privacy.probability_distribution.NormalDistribution,
            differential_privacy.probability_distribution.GaussianMixture
        ],
    },
    {
        'page': 'differential_privacy/composition.md',
        'classes': [
            differential_privacy.composition.AdaptiveDifferentialPrivacy,
            differential_privacy.composition.ExceededPrivacyBudgetError,
        ],
    },
    {
        'page': 'differential_privacy/sampling.md',
        'classes': [
            (differential_privacy.privacy_amplification_subsampling.Sampler, ['epsilon_delta_reduction', 'sample']),
            differential_privacy.privacy_amplification_subsampling.SampleWithoutReplacement
        ],
        # 'functions': [
        #     differential_privacy.dp_sampling.prod,
        #     differential_privacy.dp_sampling.check_sample_size
        # ],
    }
]
ROOT = 'http://127.0.0.1/'
