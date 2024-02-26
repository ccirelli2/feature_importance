"""
Class object to measure feature important using majority voting across n-estimators.

"""
import os
import logging
import git
import pandas as pd
import numpy as np
from typing import Dict, Union, List, Tuple
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


class DataGeneratorBuilder:
    """
    Base class for handling data.
    """

    def __init__(
        self,
        data: pd.DataFrame = pd.DataFrame({}),
        target_column: str = "TARGET",
        num_samples_synthetic: float = 1_000,
        objective: str = "classification",
        test_size: float = 0.33,
        generate_synthetic_data: bool = False,
    ):
        self.data = data
        self.objective = objective
        self.target_column = target_column
        self.num_samples_synthetic = num_samples_synthetic
        self.test_size: test_size
        self.generate_synthetic_data = generate_synthetic_data
        self.X = pd.DataFrame()
        self.y = pd.DataFrame()
        self.feature_set = []
        self.numeric_features = []
        self.categorical_features = []
        self.data_generator: dict = {
            "classification": "make_classification",
            "regression": "make_regression",
        }
        assert (
            self.objective in self.data_generator.keys()
        ), "Objective must be either classification or regression"
        logger.info(f"Class object {self.__class__.__name__} instantiated successfully")

    def _generate_synthetic_data(
        self, generate_synthetic_data: bool, num_samples_synthetic: int
    ):
        """
        Generate synthetic data if no data is provided.
        :param generate_synthetic_data:
        :param num_samples_synthetic:
        :return:
        """
        if self.generate_synthetic_data:
            logger.info(
                "Generate synthetic data with num-samples {}".format(
                    self.num_samples_synthetic
                )
            )
            self.X, self.y = self.data_generator[self.objective](
                n_samples=self.num_samples_synthetic
            )
            self.num_labels = len(Counter(self.y))
            logger.info(
                "\t Generated Data shape: {}, Target shape: {}, Labels {}".format(
                    self.X.shape, self.y.shape, Counter(self.y)
                )
            )
            # Define Feature Sets
            self.feature_set = [f"feature_{i}" for i in range(self.X.shape[1])]
            self.categorical_features = self.feature_set[: int(self.X.shape[1] / 2)]
            self.numeric_features = self.feature_set[int(self.X.shape[1] / 2) :]
        else:
            logger.info("Using provided data.")
        return self

    def _split_features(self):
        """
        Split the features into X & Y.  Define feature set.
        :return:
        """
        if all([self.X.empty, self.y.empty]):
            self.y = self.data[self.target_column]
            self.X = self.data.drop(self.target_column, axis=1)
            self.feature_set = (
                self.X.columns.tolist() if not self.feature_set else self.feature_set
            )
        conditions = [
            [f for f in self.numeric_features if f in self.feature_set],
            [f for f in self.categorical_features if f in self.feature_set],
        ]
        msg = "Features not found in feature set. \n Feature Set {}\n Numeric {}, \n Categorical {}".format(
            self.feature_set, self.numeric_features, self.categorical_features
        )
        assert all(conditions), msg

        return self

    def _generate_train_test_split(self):
        logger.info("Generating the train test split.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            stratify=self.y if self.objective == "classification" else None,
        )
        return self

    def build(self):
        self._generate_synthetic_data()
        self._split_features()
        self._generate_train_test_split()
        return self


class DataTransformerBuilder:
    """
    Base class for building transformers for numeric and categorical features.
    """

    def __init__(
        self, feature_set, numeric_features: List[str], categorical_features: List[str]
    ):
        self.feature_set = feature_set
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.numeric_pipeline = Pipeline()
        self.categorical_pipeline = Pipeline()
        self.column_transformer = ColumnTransformer(transformers=[])
        msg = "Feature set must be equal to the sum of numeric and categorical features"
        assert numeric_features + categorical_features == feature_set, msg

    def _build_numeric_column_pipeline(self):
        logger.info("Building the numeric transformer.")
        self.numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        return self

    def _build_categorical_piepline(self):
        logger.info("Building the categorical transformer.")
        self.categorical_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    CountEncoder(handle_unknown="value", handle_missing="value"),
                ),
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )
        return self

    def _build_column_transformer(self):
        logger.info("Building the column transformer.")
        self.column_transformer = ColumnTransformer(
            transformers=[
                (
                    "num",
                    self.numeric_pipeline,
                    [self.feature_set.index(f) for f in self.numeric_features],
                ),
                (
                    "cat",
                    self.categorical_pipeline,
                    [self.feature_set.index(f) for f in self.categorical_features],
                ),
            ]
        )
        return self

    def build(self):
        self._build_numeric_column_pipeline()
        self._build_categorical_piepline()
        self._build_column_transformer()
        return self


class EstimatorPipelineBuilder:
    def __init__(
        self,
        estimators: List[
            Tuple[str, Union[BaseEstimator, ClassifierMixin, RegressorMixin]]
        ],
        column_transformer: ColumnTransformer,
    ):
        self.estimators = estimators
        self.column_transformer = column_transformer
        self.estimator_pipelines = {}

    def _build_estimator_pipelines(self):
        assert self.estimators, "No estimators found."
        logger.info("Building estimator pipelines.")
        for name, estimator in self.estimators:
            conditions = [
                hasattr(estimator, "predict"),
                hasattr(estimator, "fit"),
            ]
            if not all(conditions):
                logger.warning(
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

    def build(self):
        self._build_estimator_pipelines()
        return self


class FeatureImportanceGenerator:
    """
    TODO: Do we need to fit the pipeline in order to generate feature importance?
        Ultimately the data will need to be transformed.
    """

    def __init__(
        self,
        estimator_pipelines,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train,
        feature_set: list,
        pipeline: Pipeline,
        plot_importance: bool = False,
    ):
        self.estimator_pipelines = estimator_pipelines
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.feature_set = feature_set
        self.pipeline = pipeline
        self.num_labels = len(Counter(self.y_train))
        self.plot_importance = plot_importance
        self.estimator_feature_importance = {}

    def _generate_feature_importance(self, name: str, pipeline: Pipeline):
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
        # Plot
        if self.plot_importance:
            explainer = shap.Explainer(estimator)
            shap_values = explainer(self.X_train)
            for i in range(self.num_labels):
                logger.info(f"Feature Importance Class => {i}")
                shap.plots.waterfall(shap_values[0, :, i])

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

        # Create the DataFrame
        feature_imp_df = pd.DataFrame(
            {
                f"{name}_IMPORTANCE": mean_abs_importance,
            },
            index=self.feature_set,
        )
        return feature_imp_df

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

    def _get_importance_features(self):
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

    def _get_total_feature_votes(self):
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

    def _get_majority_vote(self):
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

    def _build(self):
        assert self.estimator_pipelines, "No estimators found."
        logger.info("Building Estimators.")
        for name, pipeline in self.estimator_pipelines.items():
            logger.info(f"\t Fitting Estimator {name}.")
            pipeline.fit(self.X_train, self.y_train)
            self.estimator_feature_importance[name] = self._generate_feature_importance(
                pipeline, name
            )
        self._join_feature_importance()
        self._get_importance_features()
        self._get_total_feature_votes()
        self._get_majority_vote()
        return self


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
        objective: str,
        data: pd.DataFrame = pd.DataFrame({}),
        target_column: str = "",
        generate_synthetic_data: bool = False,
        plot_importance: bool = False,
        test_size: float = 0.33,
        feature_set: list = [],
        numeric_features: list = [],
        categorical_features: list = [],
        num_samples_synthetic: int = 10_000,
        estimators: List[
            tuple[str, Union[BaseEstimator, ClassifierMixin, RegressorMixin]]
        ] = (),
        estimator_pipelines: Dict[str, Union[BaseEstimator, Pipeline]] = {},
        pipeline: Pipeline = Pipeline(steps=[]),  # data transform + estimator
        column_transformer: ColumnTransformer = ColumnTransformer(transformers=[]),
        X_train: pd.DataFrame = pd.DataFrame({}),
        X_test: pd.DataFrame = pd.DataFrame({}),
        y_train: pd.DataFrame = pd.DataFrame({}),
        y_test: pd.DataFrame = pd.DataFrame({}),
    ) -> None:
        # Composite Classes
        self.data_generator_builder = DataGeneratorBuilder(
            data=data,
            target_column=target_column,
            objective=objective,
            test_size=test_size,
            generate_synthetic_data=generate_synthetic_data,
            num_samples_synthetic=num_samples_synthetic,
        )
        self.data_transformer_builder = DataTransformerBuilder(
            feature_set=feature_set,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        self.estimator_pipeline_builder = EstimatorPipelineBuilder(
            estimators=estimators, column_transformer=column_transformer
        )
        self.feature_importance_generator = FeatureImportanceGenerator(
            estimator_pipelines=estimator_pipelines,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            feature_set=feature_set,
            pipeline=pipeline,
            plot_importance=plot_importance,
        )

        # Attributes
        self.objective = objective
        self.data = data
        self.X_train = pd.DataFrame({})
        self.X_test = pd.DataFrame({})
        self.y_train = pd.DataFrame({})
        self.y_test = pd.DataFrame({})
        self.target_column = target_column if target_column else "TARGET"
        self.generate_synthetic_data = generate_synthetic_data
        self.plot_importance = plot_importance
        self.test_size = test_size
        self.feature_set = feature_set
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.num_samples_synthetic = num_samples_synthetic
        self.estimators = estimators
        self.estimator_pipelines = estimator_pipelines
        self.pipeline = pipeline
        self.column_transformer = column_transformer
        self.numeric_pipeline = Pipeline()
        self.categorical_pipeline = Pipeline()
        self.estimator_feature_importance: Dict[str, pd.DataFrame] = {}
        self.feature_importance_df = pd.DataFrame()
        self.feature_importance_majority_vote: Dict[str, pd.DataFrame] = {}
        msg = "number of estimators must be between 2 and 10"
        assert all([len(self.estimators) >= 2, len(self.estimators) <= 10]), msg
        if self.data.empty:
            assert (
                self.generate_synthetic_data
            ), "If data is not passed generate-synthetic-data must be true"
        logger.info(f"Class object {self.__class__.__name__} instantiated successfully")


class FeatureImportanceClassification(FeatureImportanceBaseClass):
    """
    Inherited methods from FeatureImportanceBaseClass:
    - _generate_data(self)
    - _generate_train_test_split(self)
    - _calculate_importance(self, importance_df)
    - _total_votes(self, importance_df)
    # TODO: Add evaluation metrics for classification models.
    """

    def __init(self):
        self.objective = "classification"
        super().__init__()

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
        return self.feature_importance_df

    def fit_transform(self):
        logger.info("Fitting and transforming the model.")
        self.fit()
        self.transform()
        return self.feature_importance_df


class FeatureImportanceRegression(FeatureImportanceBaseClass):
    """
    We need to separate the feature importance for classification and regression models because the
    evaluation metrics and feature importance will be different.

    Inherited methods from FeatureImportanceBaseClass:
        - _generate_data(self)
        - _generate_train_test_split(self)
        - _calculate_importance(self, importance_df)
        - _total_votes(self, importance_df)

    # TODO: Add evaluation metrics for regression models.
    """

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
            self.categorical_features = self.feature_set[: int(self.X.shape[1] / 2)]
            self.numeric_features = self.feature_set[int(self.X.shape[1] / 2) :]
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
            logger.info(f"\t Fitting Estimator {name}.")
            pipeline.fit(self.X_train, self.y_train)
            self.estimator_feature_importance[name] = self._generate_feature_importance(
                pipeline, name
            )
            self.estimator_predictions[name] = pipeline.predict(self.X_test)
            self._join_feature_importance()
            self._calculate_importance()
            self._total_votes()
            self._majority_vote()
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
        return self.feature_importance_df

    def fit_transform(self):
        logger.info("Fitting and transforming the model.")
        self.fit()
        self.transform()
        return self.feature_importance_df
