# /// script
# dependencies = ["scikit-learn==1.7.2", "pandas", "skl2onnx", "onnxruntime"]
# ///
import pandas as pd
from imputer_onnx_wrapped import SerializeableImputer
from sklearn.impute import SimpleImputer

X = pd.DataFrame({"a": [1.0, None, 3.0]}, dtype="float32")
imputer = SimpleImputer().fit(X)
print(imputer.transform(pd.DataFrame({"a": [None]})))
# Output:
# [[2.]]

wrapped = SerializeableImputer.from_scikit_learn_model(imputer, X)
wrapped.save("imputer_onnx_wrapped.json")
