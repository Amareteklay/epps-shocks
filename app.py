from __future__ import annotations
import io
import streamlit as st
import pandas as pd

from epps_shocks.config import DATA_RAW, DATA_INTERIM, MAX_LAG
from epps_shocks.prep import build_and_save
from epps_shocks.features import build_full_panel

st.title("Data Overview")

st.subheader("Raw data")
don_path    = DATA_RAW / "DONdatabase.csv"                 
shocks_path = DATA_RAW / "Shocks_Database_counts.csv"  

don_out, shocks_out = build_and_save(don_path, shocks_path)
st.success(f"Processed files saved")
        
don_df    = pd.read_csv(don_out)
shocks_df = pd.read_csv(shocks_out)

st.subheader("DON data")
st.dataframe(don_df.head())

st.subheader("Shocks data")
st.dataframe(shocks_df.head())

st.subheader("Preprocessed data")

shocks_file = DATA_INTERIM / "shocks_processed.csv"
don_file = DATA_INTERIM / "don_processed.csv"

shocks_df = pd.read_csv(shocks_file)
don_df = pd.read_csv(don_file)

panel = build_full_panel(shocks_df=shocks_df, don_df=don_df, max_lag=int(MAX_LAG))
st.subheader("Preview")
st.caption(f"Rows: {len(panel):,} • Columns: {panel.shape[1]}")
st.dataframe(panel.head(100), use_container_width=True)

with st.expander("Quick checks"):
    if "Infectious_disease" in panel.columns:
        counts = panel["Infectious_disease"].value_counts(dropna=False).to_frame("n")
        counts["share"] = counts["n"] / counts["n"].sum()
        st.write("Infectious_disease counts:", counts)
    pred_cols = [c for c in panel.columns if c not in {"Country","Continent","Year","Infectious_disease","CasesTotal","Deaths"}]
    st.write("Sample predictors:", pred_cols[:10])