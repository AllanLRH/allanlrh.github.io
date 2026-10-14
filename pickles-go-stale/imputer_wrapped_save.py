# /// script
# dependencies = ["scikit-learn==1.7.2", "pandas"]
# ///
from pathlib import Path

import pandas as pd
from imputer_simple_wrapped import SerializeableImputer
from sklearn.impute import SimpleImputer

X = pd.DataFrame({"a": [1.0, None, 3.0]})
imputer = SimpleImputer().fit(X)

SerializeableImputer(imputer, features=["a"], version="1").save(
    Path("imputer_wrapped.json")
)
