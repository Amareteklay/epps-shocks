from pathlib import Path

# Project root (…/epps-shocks)
ROOT = Path(__file__).resolve().parent.parent.parent

# Data folders (reuse your names)
DATA_RAW      = ROOT / "data" / "01_raw"
DATA_INTERIM  = ROOT / "data" / "02_interim"

# Raw counts file (adjust name if needed)
RAW_COUNTS_PATH = DATA_RAW / "Shocks_Database_counts.csv"

# Globals for processing/modeling
YEAR_MIN, YEAR_MAX = 1990, 2019
RARE_THRESHOLD     = 10
MAX_LAG            = 5
SEED               = 42

# Where to store fitted models (if/when you need it)
SAVED_MODELS_DIR = ROOT / "saved_models"

# Panel convention: country-level shock counts over several years
INDEX_VARS = ["Country", "Year"]
