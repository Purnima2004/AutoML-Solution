"""
Core data processing module for AutoML solution.
Handles all data preprocessing, validation, and transformation.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles all data preprocessing operations including loading, cleaning,
    encoding, and splitting data for machine learning.
    """

    def __init__(self):
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        self.categorical_cols = []
        self.numerical_cols = []
        self.target_column = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file with error handling.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame: Loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(
                f"Data loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns"
            )
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path}' not found.")
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze dataset and return comprehensive statistics.

        Args:
            df: Input DataFrame

        Returns:
            Dict containing data analysis results
        """
        analysis = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "numeric_columns": df.select_dtypes(
                include=[np.number]
            ).columns.tolist(),
            "categorical_columns": df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist(),
        }

        # Add statistics for numeric columns
        if analysis["numeric_columns"]:
            analysis["numeric_stats"] = (
                df[analysis["numeric_columns"]].describe().to_dict()
            )

        # Add value counts for categorical columns (top 5 for each)
        categorical_info = {}
        for col in analysis["categorical_columns"]:
            categorical_info[col] = df[col].value_counts().head().to_dict()
        analysis["categorical_info"] = categorical_info

        return analysis

    def validate_target_column(
        self, df: pd.DataFrame, target_column: str
    ) -> None:
        """
        Validate that target column exists and is suitable for ML.

        Args:
            df: Input DataFrame
            target_column: Name of target column

        Raises:
            ValueError: If target column is invalid
        """
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in dataset."
            )

        if df[target_column].isnull().all():
            raise ValueError(
                f"Target column '{target_column}' contains only null values."
            )

        # Check if it's a classification problem
        unique_values = df[target_column].nunique()
        if unique_values < 2:
            raise ValueError(
                "Target column must have at least 2 unique values for classification."
            )

        logger.info(
            f"Target column '{target_column}' validated successfully. Unique values: {unique_values}"
        )

    def check_class_imbalance(self, y: pd.Series) -> Dict[str, Any]:
        """
        Check for class imbalance in target variable.

        Args:
            y: Target variable

        Returns:
            Dict containing imbalance information
        """
        value_counts = y.value_counts()
        total_samples = len(y)

        imbalance_info = {
            "class_distribution": value_counts.to_dict(),
            "class_percentages": (value_counts / total_samples * 100).to_dict(),
            "is_imbalanced": False,
            "imbalance_ratio": 1.0,
            "minority_class": None,
            "majority_class": None,
        }

        if len(value_counts) > 1:
            max_count = value_counts.max()
            min_count = value_counts.min()
            imbalance_ratio = max_count / min_count

            imbalance_info.update(
                {
                    "imbalance_ratio": imbalance_ratio,
                    "is_imbalanced": imbalance_ratio > 2.0,
                    "minority_class": value_counts.idxmin(),
                    "majority_class": value_counts.idxmax(),
                }
            )

            if imbalance_ratio > 10:
                logger.warning(
                    f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f})"
                )
            elif imbalance_ratio > 2:
                logger.info(
                    f"Class imbalance detected (ratio: {imbalance_ratio:.2f})"
                )

        return imbalance_info

    def create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create preprocessing pipeline for features.

        Args:
            X: Feature DataFrame

        Returns:
            ColumnTransformer: Configured preprocessor
        """
        # Identify column types
        self.numerical_cols = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        self.categorical_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        logger.info(f"Numerical columns: {len(self.numerical_cols)}")
        logger.info(f"Categorical columns: {len(self.categorical_cols)}")

        # Create transformers
        transformers = []

        if self.numerical_cols:
            numerical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(
                ("num", numerical_transformer, self.numerical_cols)
            )

        if self.categorical_cols:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False
                        ),
                    ),
                ]
            )
            transformers.append(
                ("cat", categorical_transformer, self.categorical_cols)
            )

        if not transformers:
            raise ValueError("No valid columns found for preprocessing.")

        self.preprocessor = ColumnTransformer(
            transformers=transformers, remainder="drop"
        )

        return self.preprocessor

    def prepare_target(self, y: pd.Series) -> np.ndarray:
        """
        Prepare target variable for training.

        Args:
            y: Target Series

        Returns:
            np.ndarray: Encoded target variable
        """
        if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            logger.info(
                f"Target encoded. Classes: {self.label_encoder.classes_}"
            )
            return y_encoded
        else:
            return y.values

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.

        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion for test set
            random_state: Random seed

        Returns:
            Tuple of X_train, X_test, y_train, y_test
        """
        # Check if stratification is possible
        try:
            # Try stratified split first
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            logger.info("Used stratified train-test split")
        except ValueError:
            # Fall back to regular split if stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logger.info(
                "Used regular train-test split (stratification not possible)"
            )

        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test

    def fit_transform_data(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor on training data and transform both sets.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Tuple of transformed X_train, X_test
        """
        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor not created. Call create_preprocessor first."
            )

        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)

        logger.info(
            f"Data transformed. Training shape: {X_train_transformed.shape}"
        )
        logger.info(f"Test shape: {X_test_transformed.shape}")

        return X_train_transformed, X_test_transformed

    def get_feature_names(self) -> list:
        """
        Get feature names after preprocessing.

        Returns:
            List of feature names
        """
        if self.preprocessor is None:
            return []

        feature_names = []

        # Get numerical feature names
        if self.numerical_cols:
            feature_names.extend(self.numerical_cols)

        # Get categorical feature names (after one-hot encoding)
        if self.categorical_cols:
            try:
                cat_transformer = self.preprocessor.named_transformers_["cat"]
                if hasattr(
                    cat_transformer.named_steps["onehot"],
                    "get_feature_names_out",
                ):
                    cat_features = cat_transformer.named_steps[
                        "onehot"
                    ].get_feature_names_out(self.categorical_cols)
                    feature_names.extend(cat_features)
                else:
                    # Fallback for older sklearn versions
                    feature_names.extend(
                        [f"cat_{i}" for i in range(len(self.categorical_cols))]
                    )
            except Exception as e:
                logger.warning(f"Could not get categorical feature names: {e}")

        return feature_names

    def process_data(
        self, file_path: str, target_column: str, test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Complete data processing pipeline.

        Args:
            file_path: Path to CSV file
            target_column: Name of target column
            test_size: Test set proportion

        Returns:
            Dict containing all processed data and metadata
        """
        try:
            # Load and validate data
            df = self.load_data(file_path)
            self.validate_target_column(df, target_column)
            self.target_column = target_column

            # Analyze data
            analysis = self.analyze_data(df)

            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Check class imbalance
            imbalance_info = self.check_class_imbalance(y)

            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)

            # Prepare target
            y_train_encoded = self.prepare_target(y_train)
            y_test_encoded = (
                self.prepare_target(y_test)
                if self.label_encoder
                else y_test.values
            )

            # Create and fit preprocessor
            self.create_preprocessor(X_train)
            X_train_transformed, X_test_transformed = self.fit_transform_data(
                X_train, X_test
            )

            # Get feature names
            feature_names = self.get_feature_names()

            result = {
                "X_train": X_train_transformed,
                "X_test": X_test_transformed,
                "y_train": y_train_encoded,
                "y_test": y_test_encoded,
                "preprocessor": self.preprocessor,
                "label_encoder": self.label_encoder,
                "feature_names": feature_names,
                "categorical_cols": self.categorical_cols,
                "numerical_cols": self.numerical_cols,
                "target_column": target_column,
                "data_analysis": analysis,
                "imbalance_info": imbalance_info,
                "original_data_shape": df.shape,
                "classes": self.label_encoder.classes_
                if self.label_encoder
                else np.unique(y),
            }

            logger.info("Data processing completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise
