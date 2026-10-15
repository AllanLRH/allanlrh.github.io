import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import onnxruntime as ort
import pandas as pd

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class SerializeableImputer:
    # Default for new conversions; ONNX Runtime's Imputer operator only supports float32
    DTYPE = "float32"
    # Bump when the saved format changes, and handle older versions in load()
    VERSION = 1

    def __init__(self, onnx_model: bytes, features: list[str], dtype: str = DTYPE) -> None:
        self.onnx_model = onnx_model
        self.features = features
        self.dtype = dtype
        self.session = ort.InferenceSession(onnx_model)

    @classmethod
    def from_scikit_learn_model(
        cls, imputer: BaseEstimator, sample_data: pd.DataFrame
    ) -> Self:
        """
        Alternative constructor, allows creating a SerializeableImputer from a
        scikit-learn model and sample data.
        """
        # Only needed when converting, so loading works without scikit-learn and skl2onnx
        from skl2onnx import to_onnx

        # We only save the ONNX-converted model, so that this class always use
        # that rather than the Scikit Learn model. That ensure the model behaves
        # the same both before and after a serilisation-deserialization
        # round-trip
        onnx_model = to_onnx(imputer, sample_data.head(1).astype(cls.DTYPE))
        return cls(onnx_model.SerializeToString(), list(sample_data.columns))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # We handle the data conversion for the user of this class, making the
        # underlying model implementation transparrent to them.
        inputs = {
            feature: X[[feature]].to_numpy(dtype=self.dtype) for feature in self.features
        }
        (output,) = self.session.run(None, inputs)
        # ONNX Runtime may also return sparse tensors, lists and dicts
        if not isinstance(output, np.ndarray):
            raise TypeError(f"Unexpected output type: {type(output)}")
        # Assumes one output column per feature, i.e. no all-missing features were dropped in fit
        return pd.DataFrame(output, columns=self.features, index=X.index)

    def save(self, path: str | Path) -> None:
        serialized = {
            "features": self.features,
            "version": self.VERSION,
            "dtype": self.dtype,
            "onnx_model": base64.b64encode(self.onnx_model).decode(),
        }
        Path(path).write_text(json.dumps(serialized, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        raw = json.loads(Path(path).read_text())
        version = raw["version"]
        if version > cls.VERSION:
            raise ValueError(
                f"Model saved with format version {version}, "
                f"but this code only supports up to version {cls.VERSION}"
            )
        return cls(
            onnx_model=base64.b64decode(raw["onnx_model"]),
            features=raw["features"],
            dtype=raw["dtype"],
        )
