# /// script
# dependencies = ["onnxruntime", "pandas"]
# ///
import pandas as pd
from imputer_onnx_wrapped import SerializeableImputer

imputer = SerializeableImputer.load("imputer_onnx_wrapped.json")
print(imputer.transform(pd.DataFrame({"a": [None]})))
# Output:
#      a
# 0  2.0
