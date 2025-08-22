# src/modeling_grid.py
from __future__ import annotations
import math, os, glob
import itertools as it
import hashlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import pandas as pd


def _hash_spec(row, cols):
    s = "|".join(str(row[c]) for c in cols)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _as_list(x) -> List[str]:
    if x is None or x == "":
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def generate_model_grid(
    predictors,
    fixed_effects,
    scopes,
    dv = "outbreak",
    max_predictors_per_model = 6,
    min_predictors_per_model = 1,
    include_intercept = True,
    year_terms_by_scope = None,
    random_terms_by_scope = None,
    engine = "glmmTMB",
):
    scopes = list(scopes)

    if year_terms_by_scope is None:
        year_terms_by_scope = {s: ["scale(Year)"] for s in scopes}
    if random_terms_by_scope is None:
        random_terms_by_scope = {s: [""] for s in scopes} 

    pred_subsets: List[Tuple[str, ...]] = []
    for k in range(min_predictors_per_model, max_predictors_per_model + 1):
        pred_subsets.extend(it.combinations(predictors, k))

    rows = []
    for scope in scopes:
        fe_opts = _as_list(fixed_effects.get(scope, [""]))
        yr_opts = _as_list(year_terms_by_scope.get(scope, [""]))
        re_opts = _as_list(random_terms_by_scope.get(scope, [""]))
        for preds in pred_subsets:
            base_terms = list(preds)
            for fe in fe_opts:
                for yr in yr_opts:
                    for re in re_opts:
                        rhs_parts = []
                        rhs_parts.extend(base_terms)
                        if fe: rhs_parts.append(fe)
                        if yr: rhs_parts.append(yr)
                        if re: rhs_parts.append(re)
                        rhs = " + ".join(rhs_parts) if rhs_parts else "1"
                        rows.append({
                            "scope": scope,
                            "dv": dv,
                            "predictors": " + ".join(base_terms),  
                            "year_term": yr or "",
                            "random": re or "",
                            "extra_fe": fe or "",
                            "engine": engine,
                            "rhs": rhs,         
                        })

    grid = pd.DataFrame(rows)
    hash_cols = ["scope", "dv", "predictors", "year_term", "random", "extra_fe", "engine"]
    grid["spec_id"] = grid.apply(lambda r: _hash_spec(r, hash_cols), axis=1)

    grid = grid[["spec_id", "scope", "dv", "predictors", "year_term", "random", "extra_fe", "engine", "rhs"]]
    grid.drop_duplicates(subset=["spec_id"], inplace=True)

    return grid


def write_grid_batches(
    grid,
    out_dir = "specs",
    basename = "model_grid",
    batch_size = 1000,
) :
    os.makedirs(out_dir, exist_ok=True)
    n = len(grid)
    paths = []
    for i in range(math.ceil(n / batch_size)):
        batch = grid.iloc[i*batch_size:(i+1)*batch_size].copy()
        p = f"{out_dir}/{basename}_{i+1:04}.csv"
        batch.to_csv(p, index=False)
        paths.append(p)
    return paths


def filter_pending_models(
    model_grid,
    output_dir,
    merged_file = None,
    batch_glob = "batch_*.csv",
    id_col = "spec_id",
):
    done_ids = set()

    if merged_file and os.path.exists(merged_file):
        if merged_file.lower().endswith(".parquet"):
            merged = pd.read_parquet(merged_file, columns=[id_col])
        else:
            merged = pd.read_csv(merged_file, usecols=[id_col])
        done_ids.update(merged[id_col].dropna().astype(str).unique().tolist())

    if os.path.isdir(output_dir):
        for p in glob.glob(os.path.join(output_dir, batch_glob)):
            try:
                df = pd.read_csv(p, usecols=[id_col])
                done_ids.update(df[id_col].dropna().astype(str).unique().tolist())
            except Exception:
                pass

    if not done_ids:
        return model_grid

    pending = model_grid[~model_grid[id_col].astype(str).isin(done_ids)].copy()
    return pending
