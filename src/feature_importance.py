"""
Class object to measure feature important using majority voting across n-estimators.
"""
import os
import logging
import git
import pandas as pd
import numpy as np
from typing import Dict, Union, List
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

# Feature Importance
import shap

# Data
from sklearn.datasets import make_classification, make_regression

# Processing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from category_encoders.count import CountEncoder

# Use the git library to find the root directory of the project.  return a string value.
DIR_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir
DIR_DATA = os.path.join(DIR_ROOT, "data")
DIR_DATA_EXAMPLES = os.path.join(DIR_DATA, "examples")

# Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FeatureImportanceBaseClass:
    """
    Base class for building and evaluating machine learning models.

    This class provides basic functionalities for loading and preprocessing data,
    splitting data into training and testing sets, fitting estimators, and
    transforming data.

    Args:
        objective (str, optional): The objective of the model. Must be 'classification' or 'regression'. Defaults to 'classification'.
        data (pd.DataFrame, optional): The pandas DataFrame containing the data. Defaults to empty DataFrame.
        target_column (str, optional): The name of the target column in the data. Defaults to 'TARGET'.
        test_size (float, optional): The proportion of data to use for the test set. Defaults to 0.33.
        numeric_features (list, optional): List of names of numeric features in the data. Defaults to empty list.
        categorical_features (list, optional): List of names of categorical features in the data. Defaults to empty list.
        generate_synthetic_data (bool, optional): Whether to generate synthetic data if no data is provided. Defaults to False.
        num_samples_synthetic (int, optional): Number of samples to generate if synthetic data is used. Defaults to 10000.
        estimators (tuple, optional): A tuple containing a name and an sklearn estimator object. Defaults to an empty tuple.

    Attributes:
        objective (str): The objective of the model.
        data (pd.DataFrame): The pandas DataFrame containing the data.
        target_column (str): The name of the target column in the data.
        test_size (float): The proportion of data to use for the test set.
        numeric_features (list): List of names of numeric features in the data.
        categorical_features (list): List of names of categorical features in the data.
        generate_synthetic_data (bool): Whether synthetic data is generated.
        num_samples_synthetic (int): Number of samples generated if synthetic data is used.
        estimators (tuple): A tuple containing a name and a sklearn estimator object.
        categorical_transformer (sklearn.pipeline.Pipeline): Pipeline for categorical data transformation.
        numeric_transformer (sklearn.pipeline.Pipeline): Pipeline for numeric data transformation.
        column_transformer (sklearn.compose.ColumnTransformer): ColumnTransformer for combined transformations.
        X_train (pd.DataFrame): The training data features.
        X_test (pd.DataFrame): The test data features.
        y_train (pd.DataFrame): The training data target labels.
        y_test (pd.DataFrame): The test data target labels.
        estimator_pipelines (dict): A dictionary containing the estimator name and the corresponding pipeline.
        estimator_predictions (dict): A dictionary containing the estimator name and the corresponding predictions.
        estimator_feature_importance (dict): A dictionary containing the estimator name and the corresponding feature importance.
        feature_importance_df (pd.DataFrame): A DataFrame containing the joined feature importance data.
        plot_importance (bool): Whether to plot feature importance.

    """

    def __init__(
        self,
        objective: str = "classification",
        data: pd.DataFrame = pd.DataFrame(),
        target_column: str = "TARGET",
        plot_importance: bool = False,
        test_size: float = 0.33,
        numeric_features: list = [],
        categorical_features: list = [],
        generate_synthetic_data: bool = False,
        num_samples_synthetic: int = 10_000,
        estimators: List[
            tuple[str, Union[BaseEstimator, ClassifierMixin, RegressorMixin]]
        ] = (),
    ):
        self.data = data
        self.target_column = target_column
        self.test_size = test_size
        self.numeric_features = numeric_features
        self.feature_set: list = []
        self.categorical_features = categorical_features
        self.generate_synthetic_data = generate_synthetic_data
        self.num_samples_synthetic = num_samples_synthetic
        self.estimators = estimators
        self.categorical_transformer = Pipeline(steps=[])
        self.numeric_transformer = Pipeline(steps=[])
        self.column_transformer = ColumnTransformer(transformers=[])
        self.estimator_pipelines: Dict[str, Union[BaseEstimator, Pipeline]] = {}
        self.estimator_predictions: Dict[str, np.ndarray] = {}
        self.estimator_feature_importance: Dict[str, pd.DataFrame] = {}
        self.feature_importance_df = pd.DataFrame()
        self.plot_importance = plot_importance
        self.X_train = pd.DataFrame({})
        self.X_test = pd.DataFrame({})
        self.y_train = pd.DataFrame({})
        self.y_test = pd.DataFrame({})
        msg = "number of estimators must be between 2 and 10"
        assert all([len(self.estimators) >= 2, len(self.estimators) <= 10]), msg
        logger.info(f"Class object {self.__class__.__name__} instantiated successfully")

    def _build_categorical_column_transformer(self):
        logger.info("Building the categorical transformer.")
        self.categorical_transformer = Pipeline(
            steps=[
                (
                    "encoder",
                    CountEncoder(handle_unknown="value", handle_missing="value"),
                ),
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )
        return self

    def _build_numeric_column_transformer(self):
        logger.info("Building the numeric transformer.")
        self.numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        return self

    def _build_final_column_transformer(self):
        logger.info("Building the column transformer.")
        self.column_transformer = ColumnTransformer(
            transformers=[
                (
                    "num",
                    self.numeric_transformer,
                    [self.feature_set.index(f) for f in self.numeric_features],
                ),
                (
                    "cat",
                    self.categorical_transformer,
                    [self.feature_set.index(f) for f in self.categorical_features],
                ),
            ]
        )
        return self

    def _build_estimator_pipelines(self):
        assert self.estimators, "No estimators found."
        logger.info("Building estimator pipelines.")
        for name, estimator in self.estimators:
            conditions = [
                hasattr(estimator, "predict"),
                hasattr(estimator, "fit"),
            ]
            if not all(conditions):
                logger.warn(
                    f"\t Unable to Add Estimator {name} as it lacks the required methods."
                )
            else:
                logger.info(f"\t\t Adding Estimator {name}")
                self.estimator_pipelines[name] = Pipeline(
                    steps=[
                        ("preprocessor", self.column_transformer),
                        ("estimator", estimator),
                    ]
                )
        return self

    def _join_feature_importance(self):
        """
        Joins the feature importance dataframes for each estimator into a single dataframe.

        Returns:
            pandas.DataFrame: A DataFrame containing the joined feature importance data.
        """
        logger.info("Joining feature importance data.")
        # Create a list of dataframes from the feature importance dictionary
        dfs = [df for df in self.estimator_feature_importance.values()]
        # Join the dataframes
        self.feature_importance_df = pd.concat(dfs, axis=1)
        # Rename the columns
        self.feature_importance_df.columns = [
            f"{k}" for k in self.estimator_feature_importance.keys()
        ]
        return self

    def _calculate_importance(self):
        for model_name in self.feature_importance_df.columns:
            # Calculate Median Importance Value
            model_median_val = self.feature_importance_df[model_name].median()
            logger.info(f"Median value for {model_name} is {model_median_val}")

            # Create Significance Flag
            self.feature_importance_df[f"{model_name}_IS_SIGNIFICANT"] = list(
                map(
                    lambda x: 1 if x >= model_median_val else 0,
                    self.feature_importance_df[model_name],
                )
            )
        return self

    def _total_votes(self):
        """
        Calculate total significance votes.
        """
        logger.info("Calculating total votes.")
        significance_columns = [
            c for c in self.feature_importance_df.columns if "IS_SIGNIFICANT" in c
        ]
        self.feature_importance_df["TOTAL_VOTES"] = (
            self.feature_importance_df[significance_columns].sum(axis=1).values
        )
        return self

    def _majority_vote(self):
        """
        Function to create a majority vote column.
        If N-1 models agree a feature is important then 1 else 0.
        """
        logger.info("Calculating majority vote.")
        self.feature_importance_df["IS_MAJORITY"] = list(
            map(
                lambda x: 1
                if x >= (len(self.estimator_feature_importance.keys()) - 1)
                else 0,
                self.feature_importance_df["TOTAL_VOTES"],
            )
        )
        return self


class FeatureImportanceClassification(FeatureImportanceBaseClass):
    """
    Inherited methods from FeatureImportanceBaseClass:
    - _generate_data(self)
    - _generate_train_test_split(self)
    - _calculate_importance(self, importance_df)
    - _total_votes(self, importance_df)
    """

    def __init(self):
        self.objective = "classification"
        super().__init__()

    def _generate_data(self):
        if self.generate_synthetic_data:
            logger.info(
                "Generate synthetic data with num-samples {self.num_samples_synthetic}"
            )
            self.X, self.y = make_classification(n_samples=self.num_samples_synthetic)
            logger.info(
                f"\t Generated Data shape: {self.X.shape}, Target shape: {self.y.shape}, Labels {Counter(self.y)}"
            )
            # Define Feature Sets
            self.feature_set = [f"feature_{i}" for i in range(self.X.shape[1])]
            self.categorical_features = self.feature_set[: int(self.X.shape[1])]
            self.numeric_features = self.feature_set[int(self.X.shape[1]) :]
        else:
            self.y = self.data[self.target_column]
            self.X = self.data.drop(self.target_column, axis=1)
            self.feature_set = (
                self.X.columns.tolist() if not self.feature_set else self.feature_set
            )

        conditions = [
            [f for f in self.numeric_features if f in self.feature_set],
            [f for f in self.categorical_features if f in self.feature_set],
        ]
        assert all(conditions), "Features not found in feature set."

        return self

    def _generate_train_test_split(self):
        logger.info("Generating the train test split.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, stratify=self.y
        )
        return self

    def _generate_feature_importance(self, pipeline: Pipeline, name: str):
        """
        Calculates and returns feature importance using SHAP for the given pipeline.

        Args:
            pipeline: A scikit-learn pipeline containing the trained model.

        Returns:
            pandas.DataFrame: A DataFrame with two columns:
                - "feature": The feature name.
                - "shap_value": The SHAP value for feature importance.
        """
        logger.info(f"\t Generating feature importance for {name}")
        # Access the final estimator from the pipeline
        estimator = pipeline.named_steps["estimator"]
        # Create an Explainer for the estimator and obtain shap values
        explainer = shap.Explainer(estimator, self.X_train)
        shap_values = explainer(self.X_test).values
        # Calculate average SHAP values (absolute or mean, choose based on your preference)
        shap_abs = np.abs(shap_values)  # Calculate absolute values
        # For classification take average of the shap values
        mean_abs_importance = shap_abs.mean(axis=0)
        # Check if Classification and the shape of the mean_abs_importance is 2D
        if len(mean_abs_importance.shape) != 1:
            # Check if second dimension is > 2 meaning we have multiple classes
            if mean_abs_importance.shape[1] >= 2:
                # If two dimensions shap is returning values per class.  we take mean as both values are the same.
                mean_abs_importance = mean_abs_importance.mean(axis=1)
        # Plot
        if self.plot_importance:
            shap.summary_plot(shap_values, self.X_test)
        # Create the DataFrame
        importance_df = pd.DataFrame(
            {
                f"{name}_IMPORTANCE": mean_abs_importance,
            },
            index=self.feature_set,
        )
        return importance_df

    def _build(self):
        assert self.estimators, "No estimators found."
        logger.info("Building Estimators.")
        for name, pipeline in self.estimator_pipelines.items():
            # Fit Estimator
            logger.info(f"\t Fitting Estimator {name}.")
            pipeline.fit(self.X_train, self.y_train)
            # # Generate Feature Importance
            self.estimator_feature_importance[name] = self._generate_feature_importance(
                pipeline, name
            )
            # Generate Predictions
            logger.info(f"\t Generating Predictions for {name}.")
            self.estimator_predictions[name] = pipeline.predict(self.X_test)
            logger.info(f"\t\t Predictions for {name} generated successfully.")
        return self

    def fit(self):
        logger.info("Fitting the model.")
        self._generate_data()
        self._generate_train_test_split()
        self._build_categorical_column_transformer()
        self._build_numeric_column_transformer()
        self._build_final_column_transformer()
        self._build_estimator_pipelines()
        return self

    def transform(self):
        logger.info("Transforming the data.")
        self.X_train = self.column_transformer.fit_transform(self.X_train)
        self.X_test = self.column_transformer.transform(self.X_test)
        self._build()
        self._join_feature_importance()
        self._calculate_importance()
        self._total_votes()
        self._majority_vote()
        return self

    def fit_transform(self):
        logger.info("Fitting and transforming the model.")
        self.fit()
        self.transform()
        return self


class FeatureImportanceRegression(FeatureImportanceBaseClass):
    """
    We need to separate the feature importance for classification and regression models because the
    evaluation metrics and feature importance will be different.

    Inherited methods from FeatureImportanceBaseClass:
        - _generate_data(self)
        - _generate_train_test_split(self)
        - _calculate_importance(self, importance_df)
        - _total_votes(self, importance_df)

    """

    # TODO: Add regression metrics
    def __init(self):
        self.objective = "regression"
        super().__init__()

    def _generate_data(self):
        if self.generate_synthetic_data:
            logger.info(
                f"Generate synthetic data elected with num-samples {self.num_samples_synthetic}"
            )
            self.X, self.y = make_regression(n_samples=self.num_samples_synthetic)

            # Define Feature Sets
            self.feature_set = [f"feature_{i}" for i in range(self.X.shape[1])]
            self.categorical_features = self.feature_set[: int(self.X.shape[1])]
            self.numeric_features = self.feature_set[int(self.X.shape[1]) :]
        else:
            self.y = self.data[self.target_column]
            self.X = self.data.drop(self.target_column, axis=1)
            self.feature_set = (
                self.X.columns.tolist() if not self.feature_set else self.feature_set
            )

        conditions = [
            [f for f in self.numeric_features if f in self.feature_set],
            [f for f in self.categorical_features if f in self.feature_set],
        ]
        assert all(conditions), "Features not found in feature set."

        return self

    def _generate_train_test_split(self):
        logger.info("Generating the train test split.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size
        )
        return self

    def _generate_feature_importance(self, pipeline: Pipeline, name: str):
        """
        Calculates and returns feature importance using SHAP for the given pipeline.

        Args:
            pipeline: A scikit-learn pipeline containing the trained model.

        Returns:
            pandas.DataFrame: A DataFrame with two columns:
                - "feature": The feature name.
                - "shap_value": The SHAP value for feature importance.
        """
        logger.info(f"\t Generating feature importance for {name}")
        # Access the final estimator from the pipeline
        estimator = pipeline.named_steps["estimator"]
        # Create an Explainer for the estimator and obtain shap values
        explainer = shap.Explainer(estimator, self.X_train)
        shap_values = explainer(self.X_test).values
        # Calculate average SHAP values (absolute or mean, choose based on your preference)
        shap_abs = np.abs(shap_values)  # Calculate absolute values
        mean_abs_importance = shap_abs.mean(axis=0)
        # Plot
        if self.plot_importance:
            shap.summary_plot(shap_values, self.X_test)
        # Create the DataFrame
        importance_df = pd.DataFrame(
            {
                f"{name}_IMPORTANCE": mean_abs_importance,
            },
            index=self.feature_set,
        )
        return importance_df

    def _build(self):
        assert self.estimators, "No estimators found."
        logger.info("Building Estimators & Generating Feature Importance")
        for name, pipeline in self.estimator_pipelines.items():
            # Fit Estimator
            logger.info(f"\t Fitting Estimator {name}.")
            pipeline.fit(self.X_train, self.y_train)
            # # Generate Feature Importance
            self.estimator_feature_importance[name] = self._generate_feature_importance(
                pipeline, name
            )
            # Generate Predictions
            logger.info(f"\t Generating Predictions for {name}.")
            self.estimator_predictions[name] = pipeline.predict(self.X_test)
            logger.info(f"\t\t Predictions for {name} generated successfully.")
        return self

    def fit(self):
        logger.info("Fitting the model.")
        self._generate_data()
        self._generate_train_test_split()
        self._build_categorical_column_transformer()
        self._build_numeric_column_transformer()
        self._build_final_column_transformer()
        self._build_estimator_pipelines()
        return self

    def transform(self):
        logger.info("Transforming the data.")
        self.X_train = self.column_transformer.fit_transform(self.X_train)
        self.X_test = self.column_transformer.transform(self.X_test)
        self._build()
        self._join_feature_importance()
        self._calculate_importance()
        self._total_votes()
        self._majority_vote()
        return self

    def fit_transform(self):
        logger.info("Fitting and transforming the model.")
        self.fit()
        self.transform()
        return self
