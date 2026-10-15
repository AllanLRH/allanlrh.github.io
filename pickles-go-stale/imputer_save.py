# /// script
# dependencies = ["scikit-learn==1.7.2", "pandas"]  # (1)
# ///
import pickle

import pandas as pd
from sklearn.impute import SimpleImputer

imputer = SimpleImputer().fit(pd.DataFrame({"a": [1.0, None, 3.0]}))
print(imputer.transform(pd.DataFrame({"a": [None]})))
# Output:
# [[2.]]

with open("imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)
