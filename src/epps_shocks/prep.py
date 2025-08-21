from __future__ import annotations
from pathlib import Path
import pandas as pd
import country_converter as coco
import streamlit as st

from .config import YEAR_MIN, YEAR_MAX, DATA_RAW, DATA_INTERIM, RARE_THRESHOLD

def prepare_don_data(don_df):
    don_df['ReportDate'] = pd.to_datetime(don_df['ReportDate'], errors='coerce')
    don_df['Year'] = don_df['ReportDate'].dt.year

    don_df['Deaths'] = (
        don_df['Deaths']
        .astype(str)
        .str.replace('>', '', regex=False)
        .str.strip()
    )
    don_df['Deaths'] = pd.to_numeric(don_df['Deaths'], errors='coerce')
    don_df['CasesTotal'] = pd.to_numeric(don_df['CasesTotal'], errors='coerce')

    don_df = don_df[['Country', 'DiseaseLevel1', 'Year', 'CasesTotal', 'Deaths']]
    st.write(f"Before aggregating data:", don_df.shape)
    st.dataframe(don_df)
    don_df = (
        don_df
        .groupby(['Country', 'Year', 'DiseaseLevel1'], as_index=False)
        .agg({'CasesTotal': 'sum', 'Deaths': 'sum'})
    )
    st.write(f"After aggregating data:", don_df.shape)
    st.dataframe(don_df)
    return don_df

def prepare_shocks_data(shocks_df):
    df = shocks_df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[ \-]+", "_", regex=True)
    )
    df = df.rename(columns={"Country_name": "Country"})

    df = df.drop_duplicates()

    df = df.query("@YEAR_MIN <= Year <= @YEAR_MAX").dropna(subset=["Shock_category"])

    df["Country"] = df["Country"].str.replace("TÃ¼rkiye", "Türkiye", regex=False)

    countries = df["Country"].unique()
    continents = coco.convert(
        names=countries,
        src="name_short",
        to="Continent",
        not_found=None
    )
    df["Continent"] = df["Country"].map(dict(zip(countries, continents)))

    type_counts = df['Shock_type'].value_counts()
    frequent_types = type_counts[type_counts >= RARE_THRESHOLD].index
    df = df[df['Shock_type'].isin(frequent_types)].copy()

    shocks_agg = (
        df.groupby(['Country','Continent','Year','Shock_category','Shock_type'], as_index=False)
          .agg({'count':'sum'})
    )
    return shocks_agg


def build_and_save(don_raw, shocks_raw,
                   don_out = "don_processed.csv",
                   shocks_out = "shocks_processed.csv"):
    don_df    = prepare_don_data(pd.read_csv(don_raw))
    shocks_df = prepare_shocks_data(pd.read_csv(shocks_raw))

    out1 = DATA_INTERIM / don_out
    out2 = DATA_INTERIM / shocks_out
    out1.parent.mkdir(parents=True, exist_ok=True)
    don_df.to_csv(out1, index=False)
    shocks_df.to_csv(out2, index=False)

    return out1, out2

