import streamlit as st
import pandas as pd

from epps_shocks.config import DATA_RAW, DATA_INTERIM
from epps_shocks.prep import build_and_save

st.title("Test: Preprocessing Pipeline")

# Paths to raw files
don_path    = DATA_RAW / "DONdatabase.csv"                 # adjust filename
shocks_path = DATA_RAW / "Shocks_Database_counts.csv"  # adjust if needed

# Run pipeline when button is clicked
if st.button("Run preprocessing pipeline"):
    try:
        don_out, shocks_out = build_and_save(don_path, shocks_path)
        st.success(f"Processed files saved:\n- {don_out}\n- {shocks_out}")
        
        # Load processed files back to show a preview
        don_df    = pd.read_csv(don_out)
        shocks_df = pd.read_csv(shocks_out)

        st.subheader("DON data (head)")
        st.dataframe(don_df.head())

        st.subheader("Shocks data (head)")
        st.dataframe(shocks_df.head())
    except Exception as e:
        st.error(f"Error running pipeline: {e}")
