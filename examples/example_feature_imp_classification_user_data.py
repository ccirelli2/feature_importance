import os
import git
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from src.feature_importance import FeatureImportanceClassification

# Settings
logging.basicConfig(level=logging.DEBUG)

# Globals
DIR_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir
DIR_DATA = os.path.join(DIR_ROOT, "data")
DIR_DATA_EXAMPLES = os.path.join(DIR_DATA, "examples")

# Load Data
df = pd.read_csv(os.path.join(DIR_DATA_EXAMPLES, "synthetic_classification.csv"))
feature_set = [col for col in df.columns if col != "y"]
categorical_features = feature_set[int(0.5 * len(feature_set)) :]
numeric_features = [col for col in feature_set if col not in categorical_features]

# Instantiate our class object
fp = FeatureImportanceClassification(
    generate_synthetic_data=False,
    data=df,
    target_column="y",
    feature_set=feature_set,
    categorical_features=categorical_features,
    numeric_features=numeric_features,
    plot_importance=False,
    test_size=0.33,
    estimators=(
        ("RandomForestClassifier", RandomForestClassifier()),
        ("LGBMClassifier", LGBMClassifier()),
    ),
)

# Fit Model
fp.fit()

df_importance = fp.fit_transform()
print(df_importance)
