import base64
import json
import pickle
from pathlib import Path
from typing import Protocol, Self

import pandas as pd


class SupportsTransform(Protocol):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...


class SerializeableImputer:
    """
    Wrapper for a model object supporting the transform method.
    Implements saving, loading and calling of the transform method.
    """

    # Bump when the saved format changes, and handle older versions in load()
    VERSION = 1

    def __init__(self, imputer: SupportsTransform, features: list[str]) -> None:
        self.imputer = imputer
        self.features = features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        XX = X[self.features]
        return self.imputer.transform(XX)

    def save(self, path: Path) -> None:
        serialized = dict(
            features=self.features,
            model_pickle=base64.b64encode(pickle.dumps(self.imputer)).decode(),
            version=self.VERSION,
        )
        path.write_text(json.dumps(serialized))

    @classmethod
    def load(cls, path: Path) -> Self:
        raw = json.loads(path.read_bytes())
        version = raw["version"]
        if version > cls.VERSION:
            raise ValueError(
                f"Model saved with format version {version}, "
                f"but this code only supports up to version {cls.VERSION}"
            )
        imputer = pickle.loads(base64.b64decode(raw["model_pickle"]))
        # This special case handling not be in the first version of this class,
        # but as the code evolves, we can handle backwards compatability issues
        # like this.
        if version == 1:
            imputer._fill_dtype = imputer._fit_dtype
        return cls(
            imputer=imputer,
            features=raw["features"],
        )
