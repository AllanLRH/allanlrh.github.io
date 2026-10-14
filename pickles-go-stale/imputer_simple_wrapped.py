import base64
import json
import pickle
from pathlib import Path
from typing import Protocol, Self

import pandas as pd


class SupportsTransform(Protocol):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...


class SerializeableImputer:
    def __init__(
        self, imputer: SupportsTransform, features: list[str], version: str
    ) -> None:
        self.imputer = imputer
        self.features = features
        self.version = version

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        XX = X[self.features]
        return self.imputer.transform(XX)

    def save(self, path: Path) -> None:
        serialized = dict(
            features=self.features,
            model_pickle=base64.b64encode(pickle.dumps(self.imputer)).decode(),
            version=self.version,
        )
        path.write_text(json.dumps(serialized))

    @classmethod
    def load(cls, path: Path) -> Self:
        raw = json.loads(path.read_bytes())
        version = raw["version"]
        imputer = pickle.loads(base64.b64decode(raw["model_pickle"]))
        if version == "1":
            imputer._fill_dtype = imputer._fit_dtype
        return cls(
            imputer=imputer,
            features=raw["features"],
            version=version,
        )
