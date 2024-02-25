"""
Class object to measure feature important using majority voting across n-estimators.
"""
import os
import logging
import git
from uuid import uuid4
import pandas as pd
import numpy as np
from typing import Dict, Union
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from category_encoders.count import CountEncoder
from sklearn.base import BaseEstimator

from abc import ABC, abstractmethod

# Data
from sklearn.datasets import make_classification, make_regression

# Processing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Modeling
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier

# Project
from src import utils

# Directories

# Use the git library to find the root directory of the project.  return a string value.
DIR_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir
DIR_DATA = os.path.join(DIR_ROOT, "data")
DIR_DATA_EXAMPLES = os.path.join(DIR_DATA, "examples")

# Settings
logger = utils.Logger(
    directory=DIR_DATA_EXAMPLES, filename="example-majority-vote"
).get_logger()


class BaseClass(ABC):
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

    Raises:
        AssertionError: If the objective is not 'classification' or 'regression'.
        NotImplementedError: If the class is instantiated directly.
    """

    @abstractmethod
    def __init__(
        self,
        objective: str = "classification",
        data: pd.DataFrame = pd.DataFrame(),
        target_column: str = "TARGET",
        test_size: float = 0.33,
        numeric_features: list = [],
        categorical_features: list = [],
        generate_synthetic_data: bool = False,
        num_samples_synthetic: int = 10_000,
        estimators: tuple[str, BaseEstimator] = (),
    ):
        self.objective = objective
        self.data = data
        self.target_column = target_column
        self.test_size = test_size
        self.numeric_features = numeric_features
        self.feature_set = numeric_features + categorical_features
        self.categorical_features = categorical_features
        self.generate_synthetic_data = generate_synthetic_data
        self.num_samples_synthetic = num_samples_synthetic
        self.estimators = estimators
        self.categorical_transformer = Pipeline(steps=[])
        self.numeric_transformer = Pipeline(steps=[])
        self.column_transformer = ColumnTransformer(transformers=[])
        self.estimator_pipelines: Dict[str, Union[BaseEstimator, Pipeline]] = {}
        self.estimator_predictions: Dict[str, np.ndarray] = {}
        self.X_train = pd.DataFrame({})
        self.X_test = pd.DataFrame({})
        self.y_train = pd.DataFrame({})
        self.y_test = pd.DataFrame({})
        msg = "objective must be classification or regression"
        assert self.objective in ("classification", "regression"), msg
        logger.info(f"Class object {self.__class__.__name__} instantiated successfully")

    def _generate_data(self):
        if self.generate_synthetic_data:
            logger.info("Generate synthetic data elected")
            if self.objective == "classification":
                logger.info(
                    f"\t Generating data for classification with num-samples {self.num_samples_synthetic}"
                )
                X, y = make_classification(n_samples=self.num_samples_synthetic)
                data = pd.DataFrame(
                    X, columns=[f"feature_{i}" for i in range(X.shape[1])]
                )
                data[self.target_column] = y
                self.feature_set = data.columns.tolist()
                self.numeric_features = self.feature_set[:10]
                self.categorical_features = self.feature_set[10:-1]
                logger.info(
                    f"\t Data shape: {data.shape}, X shape: {X.shape}, y shape: {y.shape}"
                )
                self.data = data
            else:
                logger.info(
                    f"\t Generating data for regression with num-samples {self.num_samples_synthetic}"
                )
                X, y = make_regression(n_samples=self.num_samples_synthetic)
                data = pd.DataFrame(
                    X, columns=[f"feature_{i}" for i in range(X.shape[1])]
                )
                data[self.target_column] = y
                self.feature_set = data.columns.tolist()
                self.numeric_features = self.feature_set[:10]
                self.categorical_features = self.feature_set[10:-1]
                logger.info(
                    f"\t Data shape: {data.shape}, X shape: {X.shape}, y shape: {y.shape}"
                )
                self.data = data
        return self

    def _generate_train_test_split(self):
        logger.info("Generating the train test split.")
        X = self.data[self.feature_set]
        y = self.data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y
        )
        return self

    def build_categorical_column_transformer(self):
        logger.info("Building the categorical transformer.")
        cat_features = [self.feature_set.index(f) for f in self.categorical_features]
        self.categorical_transformer = Pipeline(
            steps=[
                (
                    "encoder",
                    CountEncoder(handle_unknown="value", handle_missing="value"),
                )
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
                hasattr(estimator, "predict_proba"),
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

    def _generate_estimator_predictions(self):
        assert self.estimators, "No estimators found."
        logger.info("Generating predictions.")
        for name, pipeline in self.estimator_pipelines.items():
            logger.info(f"\t Generating predictions for {name}")
            pipeline.fit(self.X_train, self.y_train)
            self.estimator_predictions[name] = pipeline.predict(self.X_test)
            logger.info(f"\t\t Predictions for {name} generated successfully.")
        return self

    def fit(self):
        logger.info("Fitting the model.")
        self._generate_data()
        self._generate_train_test_split()
        self.build_categorical_column_transformer()
        self._build_numeric_column_transformer()
        self._build_final_column_transformer()
        self._build_estimator_pipelines()
        return self

    def transform_data(self):
        logger.info("Transforming the data.")
        self.X_train = self.column_transformer.fit_transform(self.X_train)
        self.X_test = self.column_transformer.transform(self.X_test)
        return self

    def fit_transform(self):
        logger.info("Fitting and transforming the model.")
        self.fit()
        self.transform_data()
        self._generate_estimator_predictions()
        return self


class FeatureImportanceClassification:
    def __init(self):
        pass


class FeatureImportanceRegression:
    """
    We need to separate the feature importance for classification and regression models because the
    evaluation metrics and feature importance will be different.
    """

    def __init(self):
        pass


if __name__ == "__main__":
    mvfi = BaseClass(
        generate_synthetic_data=True,
        objective="classification",
        estimators=(
            ("RandomForestClassifier", RandomForestClassifier()),
            ("LGBMClassifier", LGBMClassifier()),
        ),
    )

    mvfi.fit_transform()
