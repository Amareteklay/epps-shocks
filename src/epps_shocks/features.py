from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence
from .config import MAX_LAG


def _add_lags_leads_avgs(df, cols, group_col, time_col, max_lag):
    if max_lag < 1 or not cols:
        return df

    out = df.sort_values([group_col, time_col]).copy()
    g = out.groupby(group_col, sort=False, group_keys=False)

    for col in cols:
        # average of positive lags
        lag_frames = [g[col].shift(i) for i in range(1, max_lag + 1)]
        out[f"{col}_lag_avg"] = pd.concat(lag_frames, axis=1).mean(axis=1, skipna=True)

        # average of leads
        lead_frames = [g[col].shift(-i) for i in range(1, max_lag + 1)]
        out[f"{col}_lead_avg"] = pd.concat(lead_frames, axis=1).mean(axis=1, skipna=True)

    return out


def build_event_panel(df, *, don_df, max_lag = MAX_LAG):

    dv = (
        df.loc[df["Shock_type"] == "Infectious disease"]
          .groupby(["Country", "Continent", "Year"], as_index=False)["count"]
          .sum()
          .rename(columns={"count": "Infectious_disease"})
    )

    events = dv.query("Infectious_disease > 0").rename(columns={"Year": "DON_year"})
    if events.empty:
        return pd.DataFrame(columns=["Country", "Continent", "DON_year", "Year_rel", "Year",
                                     "Infectious_disease", "CasesTotal", "Deaths"])

    events = events.drop(columns="Infectious_disease").assign(key=1)
    rel = pd.DataFrame({"Year_rel": np.arange(-max_lag, max_lag + 1, dtype=int), "key": 1})
    grid = events.merge(rel, on="key").drop(columns="key")
    grid["Year"] = grid["DON_year"] + grid["Year_rel"]

    preds_long = df.loc[df["Shock_type"] != "Infectious disease",
                        ["Country", "Continent", "Year", "Shock_category", "count"]]

    preds = (
        preds_long.pivot_table(index=["Country", "Continent", "Year"],
                               columns="Shock_category",
                               values="count",
                               aggfunc="sum",
                               fill_value=0)
        .rename_axis(None, axis=1)
        .reset_index()
    )

    for col in ("ECOLOGICAL", "GEOPHYSICAL"):
        if col in preds.columns:
            preds[col] = (preds[col] > 0).astype(int)

    panel = grid.merge(preds, on=["Country", "Continent", "Year"], how="left").fillna(0)

    panel = panel.merge(dv.rename(columns={"Year": "DON_year"}),
                        on=["Country", "Continent", "DON_year"], how="left")

    don_slim = (
        don_df.groupby(["Country", "Year"], as_index=False)[["CasesTotal", "Deaths"]]
              .sum()
              .rename(columns={"Year": "DON_year"})
    )
    panel = panel.merge(don_slim, on=["Country", "DON_year"], how="left")

    for c in ("Infectious_disease", "CasesTotal", "Deaths"):
        if c in panel.columns:
            panel[c] = panel[c].fillna(0).astype(int)

    meta = ["Country", "Continent", "DON_year", "Year"]
    outcomes = ["Infectious_disease", "CasesTotal", "Deaths", "Year_rel"]
    predictors = sorted(c for c in panel.columns if c not in meta + outcomes)
    panel = panel.loc[:, meta + outcomes + predictors]

    exclude = set(meta + outcomes)
    lag_cols = [c for c in panel.columns if c not in exclude]
    panel = _add_lags_leads_avgs(panel, lag_cols, "Country", "Year", max_lag)

    numeric = panel.select_dtypes(include="number").columns
    no_center = set(["Infectious_disease", "CasesTotal", "Deaths", "Year_rel"])
    for c in numeric.difference(no_center):
        if panel[c].nunique(dropna=True) > 2:
            panel[c] = panel[c] - panel[c].mean()

    return panel


def build_full_panel(shocks_df, don_df, *, max_lag = MAX_LAG):
    full_index = shocks_df[["Country", "Continent", "Year"]].drop_duplicates()

    preds_long = shocks_df.loc[shocks_df["Shock_type"] != "Infectious disease",
                               ["Country", "Continent", "Year", "Shock_category", "count"]]
    preds = (
        preds_long.pivot_table(index=["Country", "Continent", "Year"],
                               columns="Shock_category",
                               values="count",
                               aggfunc="sum",
                               fill_value=0)
        .rename_axis(None, axis=1)
        .reset_index()
    )

    for col in ("ECOLOGICAL", "GEOPHYSICAL"):
        if col in preds.columns:
            preds[col] = (preds[col] > 0).astype(int)

    panel = full_index.merge(preds, on=["Country", "Continent", "Year"], how="left").fillna(0)

    dv = (
        shocks_df.loc[shocks_df["Shock_type"] == "Infectious disease"]
                  .groupby(["Country", "Continent", "Year"], as_index=False)["count"]
                  .sum()
                  .rename(columns={"count": "Infectious_disease"})
    )
    panel = panel.merge(dv, on=["Country", "Continent", "Year"], how="left")
    panel["Infectious_disease"] = panel["Infectious_disease"].fillna(0).astype(int)

    don_slim = (
        don_df.groupby(["Country", "Year"], as_index=False)[["CasesTotal", "Deaths"]]
              .sum()
    )
    panel = panel.merge(don_slim, on=["Country", "Year"], how="left")
    panel["CasesTotal"] = panel["CasesTotal"].fillna(0).astype(int)
    panel["Deaths"] = panel["Deaths"].fillna(0).astype(int)

    meta = ["Country", "Continent", "Year"]
    outcomes = ["Infectious_disease", "CasesTotal", "Deaths"]
    predictors = sorted(c for c in panel.columns if c not in meta + outcomes)
    panel = panel.loc[:, meta + outcomes + predictors]

    lag_cols = predictors[:]  
    panel = _add_lags_leads_avgs(panel, lag_cols, "Country", "Year", max_lag)

    numeric = panel.select_dtypes(include="number").columns
    no_center = set(outcomes) | {"Year"}
    for c in numeric.difference(no_center):
        if panel[c].nunique(dropna=True) > 2:
            panel[c] = panel[c] - panel[c].mean()

    return panel
