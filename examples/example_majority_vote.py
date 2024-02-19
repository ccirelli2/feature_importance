"""
Class object to measure feature important using majority voting across n-estimators.
"""
import os
import logging
import git
from uuid import uuid4
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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
    ::threshold: float value > 0 and < 1.  The threshold value is used to select the most important features by model.
        If a feature is important it is assigned a one.
    ::estimators: list of estimators to use for majority voting.  All estimators must implement fit and
        predict methods.  All estimators will be called using the default parameter values.
    """

    def __init__(
        self,
        objective: str = "classification",
        threshold: float = 0.5,
        estimators: list = None,
    ):
        self.objective = objective
        self.threshold = threshold
        self.estimators = estimators
        assert self.objective in (
            "classification",
            "regression",
        ), "objective must be 'classification' or 'regression'"
        logger.info(
            f"Class object {self.__class__.__name__} instantiated successfully with the following parameters:"
        )
        logger.info(
            f"\t objective: {self.objective}, threshold: {self.threshold}, estimators: {self.estimators}"
        )


if __name__ == "__main__":
    mvfi = MajorityVoteFeatureImportance()
