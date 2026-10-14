# /// script
# dependencies = ["scikit-learn==1.8.0", "pandas"]
# ///
import pickle

import pandas as pd

with open("imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

print(imputer.transform(pd.DataFrame({"a": [None]})))
