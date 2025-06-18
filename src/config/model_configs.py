"""
Model configurations and hyperparameter grids for AutoML solution.
"""

from typing import Any, Dict, Tuple, Type

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class ModelConfigs:
    """
    Centralized configuration for all ML models and their hyperparameters.
    """

    @staticmethod
    def get_model_configs() -> Dict[str, Tuple[Type, Dict[str, Any]]]:
        """
        Get all model configurations with their default hyperparameter grids.

        Returns:
            Dict mapping model names to (model_class, param_grid) tuples
        """
        return {
            "SVM": ModelConfigs.get_svm_config(),
            "Decision Tree": ModelConfigs.get_decision_tree_config(),
            "Bagging": ModelConfigs.get_bagging_config(),
            "Random Forest": ModelConfigs.get_random_forest_config(),
            "AdaBoost": ModelConfigs.get_adaboost_config(),
            "XGBoost": ModelConfigs.get_xgboost_config(),
            "Neural Network": ModelConfigs.get_neural_network_config(),
            "Gradient Boosting": ModelConfigs.get_gradient_boosting_config(),
        }

    @staticmethod
    def get_svm_config() -> Tuple[Type, Dict[str, Any]]:
        """SVM configuration with hyperparameter grid."""
        return (
            SVC,
            {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                "degree": [2, 3, 4],  # For poly kernel
                "probability": [True],  # Enable probability predictions
            },
        )

    @staticmethod
    def get_decision_tree_config() -> Tuple[Type, Dict[str, Any]]:
        """Decision Tree configuration with hyperparameter grid."""
        return (
            DecisionTreeClassifier,
            {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 5, 10, 15, 20, 25, 30],
                "min_samples_split": [2, 5, 10, 15, 20],
                "min_samples_leaf": [1, 2, 4, 6, 8],
                "max_features": ["sqrt", "log2", None],
                "random_state": [42],
            },
        )

    @staticmethod
    def get_bagging_config() -> Tuple[Type, Dict[str, Any]]:
        """Bagging configuration with hyperparameter grid."""
        return (
            BaggingClassifier,
            {
                "n_estimators": [10, 50, 100, 200],
                "max_samples": [0.5, 0.7, 0.8, 1.0],
                "max_features": [0.5, 0.7, 0.8, 1.0],
                "bootstrap": [True, False],
                "bootstrap_features": [True, False],
                "random_state": [42],
            },
        )

    @staticmethod
    def get_random_forest_config() -> Tuple[Type, Dict[str, Any]]:
        """Random Forest configuration with hyperparameter grid."""
        return (
            RandomForestClassifier,
            {
                "n_estimators": [50, 100, 200, 300, 500],
                "max_depth": [None, 5, 10, 15, 20, 25, 30],
                "min_samples_split": [2, 5, 10, 15],
                "min_samples_leaf": [1, 2, 4, 6],
                "max_features": ["sqrt", "log2", None],
                "bootstrap": [True, False],
                "random_state": [42],
            },
        )

    @staticmethod
    def get_adaboost_config() -> Tuple[Type, Dict[str, Any]]:
        """AdaBoost configuration with hyperparameter grid."""
        return (
            AdaBoostClassifier,
            {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.5, 1.0, 1.5],
                "algorithm": ["SAMME", "SAMME.R"],
                "random_state": [42],
            },
        )

    @staticmethod
    def get_xgboost_config() -> Tuple[Type, Dict[str, Any]]:
        """XGBoost configuration with hyperparameter grid."""
        return (
            XGBClassifier,
            {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.2, 0.3],
                "max_depth": [3, 4, 5, 6, 7, 8],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_alpha": [0, 0.1, 0.5, 1],
                "reg_lambda": [0, 0.1, 0.5, 1],
                "random_state": [42],
                "use_label_encoder": [False],
                "eval_metric": ["logloss"],
            },
        )

    @staticmethod
    def get_neural_network_config() -> Tuple[Type, Dict[str, Any]]:
        """Neural Network configuration with hyperparameter grid."""
        return (
            MLPClassifier,
            {
                "hidden_layer_sizes": [
                    (50,),
                    (100,),
                    (150,),
                    (200,),
                    (50, 50),
                    (100, 50),
                    (100, 100),
                    (50, 50, 50),
                    (100, 50, 25),
                ],
                "activation": ["relu", "tanh", "logistic"],
                "solver": ["adam", "sgd", "lbfgs"],
                "alpha": [0.0001, 0.001, 0.01, 0.1],
                "learning_rate": ["constant", "invscaling", "adaptive"],
                "learning_rate_init": [0.001, 0.01, 0.1],
                "max_iter": [500, 1000, 1500],
                "early_stopping": [True],
                "validation_fraction": [0.1],
                "random_state": [42],
            },
        )

    @staticmethod
    def get_gradient_boosting_config() -> Tuple[Type, Dict[str, Any]]:
        """Gradient Boosting configuration with hyperparameter grid."""
        return (
            GradientBoostingClassifier,
            {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.2, 0.3],
                "max_depth": [3, 4, 5, 6, 7],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [0.6, 0.8, 1.0],
                "max_features": ["sqrt", "log2", None],
                "random_state": [42],
            },
        )

    @staticmethod
    def get_model_descriptions() -> Dict[str, str]:
        """
        Get descriptions for all models.

        Returns:
            Dict mapping model names to descriptions
        """
        return {
            "SVM": "Support Vector Machine - Powerful for high-dimensional spaces, works well with clear margin of separation.",
            "Decision Tree": "Decision Tree - Simple to understand and visualize, but can easily overfit.",
            "Bagging": "Bagging Classifier - Uses multiple instances of a base estimator to reduce variance.",
            "Random Forest": "Random Forest - An ensemble of decision trees that works well for many problems.",
            "AdaBoost": "AdaBoost - Focuses on difficult samples by boosting their importance.",
            "XGBoost": "XGBoost - Gradient boosting implementation known for its speed and performance.",
            "Neural Network": "Neural Network - Can capture complex patterns but might require more data.",
            "Gradient Boosting": "Gradient Boosting - Sequential ensemble method that builds models to correct previous errors.",
        }

    @staticmethod
    def get_simplified_configs() -> Dict[str, Tuple[Type, Dict[str, Any]]]:
        """
        Get simplified configurations for faster training during development.

        Returns:
            Dict with reduced hyperparameter grids
        """
        return {
            "SVM": (
                SVC,
                {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"],
                    "probability": [True],
                },
            ),
            "Decision Tree": (
                DecisionTreeClassifier,
                {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            ),
            "Random Forest": (
                RandomForestClassifier,
                {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["sqrt", "log2"],
                },
            ),
            "XGBoost": (
                XGBClassifier,
                {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.1, 0.2],
                    "max_depth": [3, 6, 10],
                    "subsample": [0.8, 1.0],
                    "use_label_encoder": [False],
                    "eval_metric": ["logloss"],
                },
            ),
            "Neural Network": (
                MLPClassifier,
                {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                    "activation": ["relu", "tanh"],
                    "solver": ["adam"],
                    "alpha": [0.0001, 0.001],
                    "max_iter": [500],
                },
            ),
        }


# Convenience function to get configurations
def get_all_model_configs(
    simplified: bool = False,
) -> Dict[str, Tuple[Type, Dict[str, Any]]]:
    """
    Get all model configurations.

    Args:
        simplified: Whether to use simplified configs for faster training

    Returns:
        Dict of model configurations
    """
    if simplified:
        return ModelConfigs.get_simplified_configs()
    else:
        return ModelConfigs.get_model_configs()


def get_model_descriptions() -> Dict[str, str]:
    """Get model descriptions."""
    return ModelConfigs.get_model_descriptions()
