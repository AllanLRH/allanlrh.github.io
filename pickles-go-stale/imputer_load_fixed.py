# /// script
# dependencies = ["scikit-learn==1.8.0", "pandas"]
# ///
import pickle

import pandas as pd

with open("imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

# 1.8.0's fit() sets the new _fill_dtype to the same value as the existing _fit_dtype.
imputer._fill_dtype = imputer._fit_dtype

print(imputer.transform(pd.DataFrame({"a": [None]})))
