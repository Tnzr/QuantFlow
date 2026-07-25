from __future__ import annotations

from pathlib import Path

import pandas as pd

from .persistence import get_engine


def _prepare_output_path(out_path: str) -> Path:
    path = Path(out_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def export_dataframe_to_parquet(df: pd.DataFrame, out_path: str) -> str:
    path = _prepare_output_path(out_path)
    df.to_parquet(path, index=False)
    return str(path)


def export_table_to_parquet(table_name: str, db_path: str, out_path: str) -> str:
    engine = get_engine(db_path)
    df = pd.read_sql_table(table_name, con=engine)
    return export_dataframe_to_parquet(df, out_path)