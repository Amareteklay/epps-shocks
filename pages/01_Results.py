# 01_inspect_results.py
import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Results", layout="wide")  

st.title("Results")
st.write("Loaded!")

PARQUET = Path("./results/summaries/model_results_lags.parquet")  

df = pd.read_parquet(PARQUET)
st.write("Rows:", len(df))
st.write("Cols:", df.columns.tolist())

# Common cleanups
if "converged" in df.columns:
    df = df[df["converged"].fillna(False)]

# Rank by AICc (fallback to AIC)
score = "aicc" if "aicc" in df.columns else "aic"
ranked = df.sort_values(score, ascending=True).reset_index(drop=True)

# Top 30 overall
st.write(ranked[[c for c in ["spec_id","scope","dv","formula","engine",score, "aic", "n"] if c in ranked.columns]].head(30))

# Best per scope
if "scope" in ranked.columns:
    best_per_scope = ranked.sort_values([ "scope", score ]).groupby("scope", as_index=False).first()
    st.write("\nBest per scope:")
    st.write(best_per_scope[["scope","spec_id","engine",score, "aic", "n","formula"]])
    best_per_scope.to_csv("results/summaries/best_per_scope_lags.csv", index=False)
