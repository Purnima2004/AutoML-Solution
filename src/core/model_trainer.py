"""
Core model training module for AutoML solution.
Handles model training, evaluation, and hyperparameter optimization.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles model training, hyperparameter optimization, and evaluation.
    """

    def __init__(self, models_config: Dict[str, Any]):
        self.models_config = models_config
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.evaluation_results = {}

    def create_model_pipeline(self, model, preprocessor) -> Pipeline:
        """
        Create a pipeline with preprocessor and model.

        Args:
            model: ML model instance
            preprocessor: Data preprocessor

        Returns:
            Pipeline: Complete ML pipeline
        """
        return Pipeline([("preprocessor", preprocessor), ("classifier", model)])

    def train_single_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        preprocessor=None,
        param_grid: Optional[Dict] = None,
        cv_folds: int = 3,
        n_iter: int = 10,
        scoring: str = "f1_weighted",
    ) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter optimization.

        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training targets
            preprocessor: Data preprocessor (optional if data already preprocessed)
            param_grid: Hyperparameter grid
            cv_folds: Number of CV folds
            n_iter: Number of iterations for RandomizedSearchCV
            scoring: Scoring metric

        Returns:
            Dict containing training results
        """
        if model_name not in self.models_config:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        model_class, default_params = self.models_config[model_name]
        param_grid = param_grid or default_params

        logger.info(f"Training {model_name}...")
        start_time = time.time()

        try:
            # Create model instance
            model = model_class()

            # Create pipeline if preprocessor provided
            if preprocessor is not None:
                pipeline = self.create_model_pipeline(model, preprocessor)
                # Adjust parameter names for pipeline
                param_grid = {
                    f"classifier__{k}": v for k, v in param_grid.items()
                }
                search_model = pipeline
            else:
                search_model = model

            # Handle class imbalance for applicable models
            if hasattr(model, "class_weight") and len(np.unique(y_train)) > 1:
                if (
                    "class_weight" not in param_grid
                    and "classifier__class_weight" not in param_grid
                ):
                    param_key = (
                        "classifier__class_weight"
                        if preprocessor
                        else "class_weight"
                    )
                    param_grid[param_key] = ["balanced", None]

            # Calculate n_iter based on parameter grid size
            total_combinations = 1
            for param_values in param_grid.values():
                if isinstance(param_values, (list, tuple)):
                    total_combinations *= len(param_values)

            n_iter = min(n_iter, total_combinations)

            # Perform hyperparameter search
            search = RandomizedSearchCV(
                estimator=search_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scoring,
                random_state=42,
                n_jobs=-1,
                verbose=0,
            )

            search.fit(X_train, y_train)

            training_time = time.time() - start_time

            # Store results
            result = {
                "model": search.best_estimator_,
                "best_params": search.best_params_,
                "best_score": search.best_score_,
                "cv_results": search.cv_results_,
                "training_time": training_time,
                "model_name": model_name,
            }

            self.trained_models[model_name] = result

            logger.info(
                f"{model_name} training completed in {training_time:.2f}s"
            )
            logger.info(f"Best CV score: {search.best_score_:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise

    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        classes: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            classes: Class labels

        Returns:
            Dict containing evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")

        try:
            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "f1_score": f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
            }

            # Add ROC AUC for binary classification
            if len(np.unique(y_test)) == 2:
                try:
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_test)[:, 1]
                        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
                    elif hasattr(model, "decision_function"):
                        y_scores = model.decision_function(X_test)
                        metrics["roc_auc"] = roc_auc_score(y_test, y_scores)
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}")

            # Classification report
            report = classification_report(
                y_test,
                y_pred,
                target_names=classes if classes is not None else None,
                output_dict=True,
                zero_division=0,
            )

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            evaluation_result = {
                "model_name": model_name,
                "metrics": metrics,
                "classification_report": report,
                "confusion_matrix": cm,
                "predictions": y_pred,
                "test_accuracy": metrics["accuracy"],
            }

            # Add probability predictions if available
            if hasattr(model, "predict_proba"):
                evaluation_result["prediction_probabilities"] = (
                    model.predict_proba(X_test)
                )

            self.evaluation_results[model_name] = evaluation_result

            logger.info(f"{model_name} evaluation completed")
            logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")

            return evaluation_result

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            raise

    def train_multiple_models(
        self,
        model_names: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        preprocessor=None,
        param_grids: Optional[Dict] = None,
        classes: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Train and evaluate multiple models.

        Args:
            model_names: List of model names to train
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            preprocessor: Data preprocessor
            param_grids: Custom parameter grids for models
            classes: Class labels

        Returns:
            Dict containing results for all models
        """
        results = {
            "training_results": {},
            "evaluation_results": {},
            "best_model_name": None,
            "best_model": None,
            "best_score": -np.inf,
        }

        param_grids = param_grids or {}

        for model_name in model_names:
            try:
                # Train model
                param_grid = param_grids.get(model_name)
                training_result = self.train_single_model(
                    model_name, X_train, y_train, preprocessor, param_grid
                )
                results["training_results"][model_name] = training_result

                # Evaluate model
                evaluation_result = self.evaluate_model(
                    training_result["model"],
                    X_test,
                    y_test,
                    model_name,
                    classes,
                )
                results["evaluation_results"][model_name] = evaluation_result

                # Track best model
                current_score = evaluation_result["metrics"]["f1_score"]
                if current_score > results["best_score"]:
                    results["best_score"] = current_score
                    results["best_model_name"] = model_name
                    results["best_model"] = training_result["model"]
                    self.best_model = training_result["model"]
                    self.best_model_name = model_name

            except Exception as e:
                logger.error(f"Failed to train/evaluate {model_name}: {str(e)}")
                continue

        logger.info(
            f"Best model: {results['best_model_name']} (F1: {results['best_score']:.4f})"
        )

        return results

    def get_feature_importance(
        self, model_name: str
    ) -> Optional[Dict[str, float]]:
        """
        Get feature importance from trained model.

        Args:
            model_name: Name of the model

        Returns:
            Dict mapping feature names to importance scores
        """
        if model_name not in self.trained_models:
            logger.warning(f"Model {model_name} not found in trained models")
            return None

        model = self.trained_models[model_name]["model"]

        # Extract the actual classifier from pipeline if needed
        if hasattr(model, "named_steps"):
            classifier = model.named_steps.get("classifier", model)
        else:
            classifier = model

        if hasattr(classifier, "feature_importances_"):
            return classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            # For linear models, use absolute coefficients
            return (
                np.abs(classifier.coef_[0])
                if len(classifier.coef_.shape) > 1
                else np.abs(classifier.coef_)
            )
        else:
            logger.info(
                f"Model {model_name} does not support feature importance"
            )
            return None

    def save_model(self, model_name: str, save_path: str) -> bool:
        """
        Save a trained model to disk.

        Args:
            model_name: Name of the model to save
            save_path: Path to save the model

        Returns:
            bool: Success status
        """
        if model_name not in self.trained_models:
            logger.error(f"Model {model_name} not found in trained models")
            return False

        try:
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            model = self.trained_models[model_name]["model"]
            joblib.dump(model, save_path)

            logger.info(f"Model {model_name} saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
            return False

    def load_model(self, model_path: str) -> Any:
        """
        Load a saved model from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded model
        """
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using a trained model.

        Args:
            model_name: Name of the model
            X: Features to predict

        Returns:
            Predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")

        model = self.trained_models[model_name]["model"]
        return model.predict(X)

    def predict_proba(
        self, model_name: str, X: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Get prediction probabilities using a trained model.

        Args:
            model_name: Name of the model
            X: Features to predict

        Returns:
            Prediction probabilities or None if not supported
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")

        model = self.trained_models[model_name]["model"]

        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        else:
            logger.warning(
                f"Model {model_name} does not support probability predictions"
            )
            return None
