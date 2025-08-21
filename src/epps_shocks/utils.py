from pathlib import Path
import pandas as pd

def save_csv(df: pd.DataFrame, path: Path, index: bool=False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path
