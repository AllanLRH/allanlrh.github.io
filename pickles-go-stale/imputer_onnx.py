# /// script
# dependencies = ["scikit-learn==1.7.2", "pandas", "skl2onnx", "onnxruntime"]
# ///
import onnxruntime
import pandas as pd
import skl2onnx
from sklearn.impute import SimpleImputer

X_train = pd.DataFrame({"a": [1.0, None, 3.0]}, dtype="float32")

# Train the base SimpleImputer...
imputer = SimpleImputer().fit(X_train)
# ...convert to onnx...
imputer_onnx = skl2onnx.to_onnx(model=imputer, X=X_train[:1])
# ...and load it into a session to get an executeable object
imputer_session = onnxruntime.InferenceSession(imputer_onnx.SerializeToString())

X_test = pd.DataFrame({"a": [None]}, dtype="float32")
inputs = {col: X_test[[col]].to_numpy() for col in X_test.columns}

print(imputer_session.run(output_names=None, input_feed=inputs)[0])
# Output:
# [[2.]]
print(imputer.transform(X_test))
# Output:
# [[2.]]
