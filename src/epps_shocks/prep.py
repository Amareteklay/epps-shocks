from __future__ import annotations
from pathlib import Path
import pandas as pd
import country_converter as coco

from .config import YEAR_MIN, YEAR_MAX, DATA_RAW, DATA_INTERIM, RARE_THRESHOLD

# ---------- DON ----------
def prepare_don_data(don_df: pd.DataFrame) -> pd.DataFrame:
    """Clean DON dataset and aggregate repeated disease outbreaks by Country-Year."""
    # Parse dates and extract year
    don_df['ReportDate'] = pd.to_datetime(don_df['ReportDate'], errors='coerce')
    don_df['Year'] = don_df['ReportDate'].dt.year

    # Clean numeric columns
    don_df['Deaths'] = (
        don_df['Deaths']
        .astype(str)
        .str.replace('>', '', regex=False)
        .str.strip()
    )
    don_df['Deaths'] = pd.to_numeric(don_df['Deaths'], errors='coerce')
    don_df['CasesTotal'] = pd.to_numeric(don_df['CasesTotal'], errors='coerce')

    # Keep only relevant columns
    don_df = don_df[['Country', 'DiseaseLevel1', 'Year', 'CasesTotal', 'Deaths']]

    # Aggregate by Country, Year, Disease
    don_df = (
        don_df
        .groupby(['Country', 'Year', 'DiseaseLevel1'], as_index=False)
        .agg({'CasesTotal': 'sum', 'Deaths': 'sum'})
    )
    return don_df

# ---------- Shocks ----------
def prepare_shocks_data(shocks_df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline for shock data."""
    df = shocks_df.copy()

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[ \-]+", "_", regex=True)
    )
    df = df.rename(columns={"Country_name": "Country"})

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Keep year range and drop rows missing a shock category
    df = df.query("@YEAR_MIN <= Year <= @YEAR_MAX").dropna(subset=["Shock_category"])

    # Fix country typos (done inline here)
    df["Country"] = df["Country"].str.replace("TÃ¼rkiye", "Türkiye", regex=False)

    # Add continent
    countries = df["Country"].unique()
    continents = coco.convert(
        names=countries,
        src="name_short",
        to="Continent",
        not_found=None
    )
    df["Continent"] = df["Country"].map(dict(zip(countries, continents)))

    # Drop rare shock types
    type_counts = df['Shock_type'].value_counts()
    frequent_types = type_counts[type_counts >= RARE_THRESHOLD].index
    df = df[df['Shock_type'].isin(frequent_types)].copy()

    # Aggregate
    shocks_agg = (
        df.groupby(['Country','Continent','Year','Shock_category','Shock_type'], as_index=False)
          .agg({'count':'sum'})
    )
    return shocks_agg


def build_and_save(don_raw: str | Path, shocks_raw: str | Path,
                   don_out: str = "don_processed.csv",
                   shocks_out: str = "shocks_processed.csv") -> tuple[Path, Path]:
    """Run full pipeline and save to data/02_interim; returns output paths."""
    don_df    = prepare_don_data(pd.read_csv(don_raw))
    shocks_df = prepare_shocks_data(pd.read_csv(shocks_raw))

    out1 = DATA_INTERIM / don_out
    out2 = DATA_INTERIM / shocks_out
    out1.parent.mkdir(parents=True, exist_ok=True)
    don_df.to_csv(out1, index=False)
    shocks_df.to_csv(out2, index=False)

    return out1, out2

