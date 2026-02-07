# market_simulation/models/utils_ensemble_model.py

from __future__ import annotations

import bisect
import glob
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset
from zarr.storage import DirectoryStore


def _key_from_features(path: str) -> str:
    base = os.path.basename(path)
    if not (base.startswith("features_") and base.endswith("_cut.parquet")):
        raise ValueError(f"Unexpected features filename: {base}")
    return base[len("features_") : -len("_cut.parquet")]


def _key_from_next1(path: str) -> str:
    base = os.path.basename(path)
    if not (base.startswith("next1_tokens_") and base.endswith(".zarr")):
        raise ValueError(f"Unexpected next1 filename: {base}")
    return base[len("next1_tokens_") : -len(".zarr")]


class MultiFileEnsembleDataset(Dataset):
    """
    - parquets/features_<KEY>_cut.parquet   columns: i, Time, f0..f14 (ints)
    - next1s/next1_tokens_<KEY>.zarr        shape: (N, 64) ints

    Returns:
      features: torch.int64 (15,)     # f0..f14
      next1_tokens: torch.int64 (64,)
    """

    def __init__(
        self,
        parquets_dir: str,
        next1s_dir: str,
        *,
        parquet_pattern: str = "features_*_cut.parquet",
        next1_pattern: str = "next1_tokens_*.zarr",
        array_path: str = "arr_0",
    ):
        self.parquets_dir = str(parquets_dir)
        self.next1s_dir = str(next1s_dir)
        self.array_path = str(array_path)

        self.feature_cols = [f"f{k}" for k in range(15)]

        feat_paths = sorted(glob.glob(os.path.join(self.parquets_dir, parquet_pattern)))
        next_paths = sorted(glob.glob(os.path.join(self.next1s_dir, next1_pattern)))

        feat_by_key = {_key_from_features(p): p for p in feat_paths}
        next_by_key = {_key_from_next1(p): p for p in next_paths}

        keys = sorted(set(feat_by_key).intersection(next_by_key))
        if not keys:
            raise FileNotFoundError(
                "No matching (parquet, zarr) pairs found.\n"
                f"parquets_dir={self.parquets_dir} pattern={parquet_pattern}\n"
                f"next1s_dir={self.next1s_dir} pattern={next1_pattern}"
            )

        self.pairs = []
        self.lens = []

        for k in keys:
            parquet_path = feat_by_key[k]
            zarr_dir = next_by_key[k]

            if zarr_dir.endswith(".zarr.zip"):
                raise ValueError(
                    f"Found .zarr.zip but expected extracted .zarr dir: {zarr_dir}"
                )

            df = pd.read_parquet(parquet_path, engine="pyarrow")

            missing = [c for c in self.feature_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Missing feature columns in {parquet_path}: {missing}\n"
                    f"Columns present: {list(df.columns)}"
                )

            n_feat = int(df.shape[0])

            A = zarr.open(DirectoryStore(zarr_dir), path=self.array_path, mode="r")
            n_tok = int(A.shape[0])

            if n_feat != n_tok:
                raise ValueError(
                    f"Length mismatch for key={k}: parquet={n_feat}, zarr={n_tok}\n"
                    f"  {parquet_path}\n"
                    f"  {zarr_dir} (array_path={self.array_path})"
                )

            self.pairs.append((parquet_path, zarr_dir))
            self.lens.append(n_feat)

        self.cum = []
        s = 0
        for L in self.lens:
            s += L
            self.cum.append(s)

    def __len__(self) -> int:
        return self.cum[-1] if self.cum else 0

    @staticmethod
    @lru_cache(maxsize=8)
    def _open_parquet(parquet_path: str) -> pd.DataFrame:
        return pd.read_parquet(parquet_path, engine="pyarrow")

    @staticmethod
    @lru_cache(maxsize=32)
    def _open_zarr(zarr_dir: str, array_path: str):
        return zarr.open(DirectoryStore(zarr_dir), path=array_path, mode="r")

    def __getitem__(self, idx: int):
        fi = bisect.bisect_right(self.cum, idx)
        prev = 0 if fi == 0 else self.cum[fi - 1]
        j = idx - prev

        parquet_path, zarr_dir = self.pairs[fi]

        df = self._open_parquet(parquet_path)

        # Select f0..f14 only, keep ints
        feats = df.loc[df.index[j], self.feature_cols].to_numpy(copy=False)
        feats = np.asarray(feats, dtype=np.int64)  # ensure consistent dtype

        A = self._open_zarr(zarr_dir, self.array_path)
        toks = np.asarray(A[j].reshape(-1), dtype=np.int64)
        if toks.shape[0] != 64:
            raise ValueError(f"Expected 64 next1 tokens, got {toks.shape} in {zarr_dir}")

        return torch.from_numpy(feats).long(), torch.from_numpy(toks).long()
