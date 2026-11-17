"""Download (into ./data) and split the benchmark datasets.

Every dataset is split once, with a fixed seed, into train / validation / test.
Validation is used for early stopping only; all reported numbers are on test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_covtype, fetch_openml
from sklearn.model_selection import train_test_split

DATA_HOME = os.environ.get("GBDT_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))


@dataclass
class Dataset:
    name: str
    task: str  # "regression" | "binary" | "multiclass"
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.y_train).shape[0]) if self.task != "regression" else 0

    def describe(self) -> str:
        n_miss = int(np.isnan(self.X_train).sum())
        return (
            f"{self.name}: task={self.task} train={self.X_train.shape} val={self.X_val.shape[0]} "
            f"test={self.X_test.shape[0]} missing_train={n_miss}"
        )


def _split(X, y, seed, val=0.2, test=0.2, stratify=True):
    strat = y if stratify else None
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test, random_state=seed, stratify=strat)
    strat = y_tmp if stratify else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val / (1 - test), random_state=seed, stratify=strat
    )
    return X_tr, y_tr, X_val, y_val, X_test, y_test


def california(seed: int = 0) -> Dataset:
    d = fetch_california_housing(data_home=DATA_HOME)
    parts = _split(d.data, d.target, seed, stratify=False)
    return Dataset("california", "regression", *parts, list(d.feature_names))


def adult(seed: int = 0) -> Dataset:
    """Census income (>50K). Categoricals are ordinal-coded; '?' entries stay NaN."""
    d = fetch_openml(data_id=1590, as_frame=True, data_home=DATA_HOME, parser="auto")
    df = d.data.copy()
    for c in df.columns:
        if str(df[c].dtype) == "category":
            codes = df[c].cat.codes.astype(np.float64)
            df[c] = codes.where(codes >= 0, np.nan)
    X = df.to_numpy(dtype=np.float64)
    y = (d.target.astype(str).str.strip() == ">50K").to_numpy().astype(np.int64)
    parts = _split(X, y, seed)
    return Dataset("adult", "binary", *parts, list(df.columns))


def higgs(seed: int = 0) -> Dataset:
    """OpenML HIGGS (98k-row version of the UCI Higgs boson dataset)."""
    X, y = fetch_openml(data_id=23512, as_frame=False, data_home=DATA_HOME, parser="liac-arff", return_X_y=True)
    X = X.astype(np.float64)
    y = (y.astype(str) == "1").astype(np.int64)
    parts = _split(X, y, seed)
    return Dataset("higgs", "binary", *parts, [f"f{i}" for i in range(X.shape[1])])


def covertype(seed: int = 0, n_rows: int = 200_000) -> Dataset:
    """Forest cover type (7 classes); a stratified random subset of ``n_rows`` rows."""
    d = fetch_covtype(data_home=DATA_HOME)
    y = d.target.astype(np.int64) - 1
    X, _, y, _ = train_test_split(d.data, y, train_size=n_rows, random_state=seed, stratify=y)
    parts = _split(X, y, seed)
    return Dataset("covertype", "multiclass", *parts, list(d.feature_names))


LOADERS = {"california": california, "adult": adult, "higgs": higgs, "covertype": covertype}


def load(name: str, seed: int = 0) -> Dataset:
    return LOADERS[name](seed)
