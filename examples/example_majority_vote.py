"""
Class object to measure feature important using majority voting across n-estimators.
"""
import os
import logging
import git
from uuid import uuid4
import pandas as pd
import numpy as np

# Data
from sklearn.datasets import make_classification, make_regression

# Processing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Modeling
from lightgbm import LGBMClassifier, LGBMRegressor

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


class MajorityVoteFeatureImportance:
    """
    Class object to measure feature important using majority voting across n-estimators.

    ::objective: str value.  The objective of the model.  Either 'classification' or 'regression'.
    ::vote_threshold: float value > 0 and < 1.  The threshold value is used to select the most important features by model.
        If a feature is important it is assigned a one.
    ::estimators: list of estimators to use for majority voting.  All estimators must implement fit and
        predict methods.  All estimators will be called using the default parameter values.
    """

    def __init__(
        self,
        objective: str = "classification",
        data: pd.DataFrame = pd.DataFrame(),
        target_column: str = "TARGET",
        vote_threshold: float = 0.5,
        estimators: list = None,
    ):
        self.objective = objective
        self.data = data
        self.samples_synthetic = 10_000
        self.target_column = target_column
        self.vote_threshold = vote_threshold
        self.estimators = estimators
        assert self.objective in (
            "classification",
            "regression",
        ), "objective must be 'classification' or 'regression'"
        logger.info(
            f"Class object {self.__class__.__name__} instantiated successfully with the following parameters:"
        )
        logger.info(
            f"\t objective: {self.objective}, threshold: {self.vote_threshold}, estimators: {self.estimators}"
        )

    def _generate_data(self):
        if self.data.empty:
            logger.info("No data was provided.  Generating synthetic data.")
            if self.objective == "classification":
                X, y = make_classification(n_samples=self.samples_synthetic)
                data = pd.DataFrame(
                    X, columns=[f"feature_{i}" for i in range(X.shape[1])]
                )
                data[self.target_column] = y
                logger.info(
                    f"Data shape: {data.shape}, X shape: {X.shape}, y shape: {y.shape}"
                )
                self.data = data
            else:
                X, y = make_regression(n_samples=self.samples_synthetic)
                data = pd.DataFrame(
                    X, columns=[f"feature_{i}" for i in range(X.shape[1])]
                )
                data[self.target_column] = y
                self.data = data
        return self

    def fit(self):
        logger.info("Fitting the model.")
        self._generate_data()
        return self


if __name__ == "__main__":
    mvfi = MajorityVoteFeatureImportance()
    mvfi.fit()
