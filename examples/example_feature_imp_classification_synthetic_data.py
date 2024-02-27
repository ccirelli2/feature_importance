import os
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, LGBMRegressor

from src.feature_importance import FeatureImportanceClassification

# Settings
logging.basicConfig(level=logging.DEBUG)

# Instantiate our class object
fp = FeatureImportanceClassification(
    generate_synthetic_data=True,
    plot_importance=False,
    test_size=0.33,
    num_samples_synthetic=1000,
    estimators=(
        ("RandomForestClassifier", RandomForestClassifier()),
        ("LGBMClassifier", LGBMClassifier()),
    ),
)

# Fit Model
fp.fit()

df_importance = fp.fit_transform()
print(df_importance)
