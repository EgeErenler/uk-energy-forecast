"""NESO's own day-ahead demand forecast for every half hour since April 2021.

This is the benchmark: the official forecast, published the morning before each day.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from energyforecast import http
from energyforecast.config import NESO_FORECAST_URL

log = logging.getLogger(__name__)


def fetch(raw_dir: Path) -> Path:
    return http.download(NESO_FORECAST_URL, raw_dir / "neso" / "day_ahead_forecast.csv")


def parse(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    out = pd.DataFrame(
        {
            "settlement_date": pd.to_datetime(df["Date"]).dt.normalize(),
            "settlement_period": df["Settlement_Period"].astype(int),
            "neso_forecast": pd.to_numeric(df["Demand_Forecast"], errors="coerce"),
            "neso_outturn": pd.to_numeric(df["Demand_Outturn"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["neso_forecast"])
    return out.drop_duplicates(["settlement_date", "settlement_period"], keep="last")


def clean(csv_path: Path, clean_dir: Path) -> Path:
    df = parse(pd.read_csv(csv_path, encoding="utf-8-sig"))
    clean_dir.mkdir(parents=True, exist_ok=True)
    target = clean_dir / "neso_forecast.parquet"
    df.to_parquet(target, index=False)
    log.info(
        "neso forecast: %s half hours up to %s", f"{len(df):,}", df["settlement_date"].max().date()
    )
    return target
