# /// script
# dependencies = ["scikit-learn==1.8.0", "pandas"]
# ///
from pathlib import Path

import pandas as pd
from imputer_simple_wrapped import SerializeableImputer

imputer = SerializeableImputer.load(Path("imputer_wrapped.json"))
print(imputer.transform(pd.DataFrame({"a": [None]})))
