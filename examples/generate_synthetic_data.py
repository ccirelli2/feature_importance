import os
import pandas as pd
import git
from sklearn.datasets import make_classification, make_regression

# Globals
DIR_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir
DIR_DATA = os.path.join(DIR_ROOT, "data")
DIR_DATA_EXAMPLES = os.path.join(DIR_DATA, "examples")

# Generate Synthetic Classification Data
X, y = make_classification()
data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
data["y"] = y
data.to_csv(
    os.path.join(DIR_DATA_EXAMPLES, "synthetic_classification.csv"), index=False
)

# Generate Synthetic Regression Data
X, y = make_regression()
data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
data["y"] = y
data.to_csv(os.path.join(DIR_DATA_EXAMPLES, "synthetic_regression.csv"), index=False)
