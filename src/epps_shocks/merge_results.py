# src/merge_results.py
from __future__ import annotations
import glob
import os
from typing import List, Optional
import pandas as pd


def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def merge_model_results(input_dir, output_file, pattern = "batch_*.csv",
    id_col = "spec_id",
    delete_after = False,
):

    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        if os.path.exists(output_file):
            return _read_any(output_file)
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if id_col in df.columns:
                frames.append(df)
        except Exception:
            pass

    if not frames:
        if os.path.exists(output_file):
            return _read_any(output_file)
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    if id_col in merged.columns:
        merged.drop_duplicates(subset=[id_col], inplace=True)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    if output_file.lower().endswith(".parquet"):
        merged.to_parquet(output_file, index=False)
    else:
        merged.to_csv(output_file, index=False)

    if delete_after:
        for fp in files:
            try:
                os.remove(fp)
            except Exception:
                pass

    return merged


def rank_models(df, score_col = "aicc", ascending = True, top_n = 50):
    if df.empty or score_col not in df.columns:
        return df
    ranked = df.sort_values(by=[score_col], ascending=ascending).copy()
    if top_n:
        ranked = ranked.head(top_n)
    return ranked
