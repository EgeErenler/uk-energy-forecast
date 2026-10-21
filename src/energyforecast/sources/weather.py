"""Hourly weather for Great Britain from Open-Meteo, as a population-weighted average of cities.

History comes from the ERA5 archive (a few days behind). Recent days and the next two days
come from the forecast API, so the same columns exist for training and for tomorrow.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from energyforecast import http
from energyforecast.config import (
    CITIES,
    FIRST_YEAR,
    METEO_ARCHIVE_URL,
    METEO_FORECAST_URL,
    WEATHER_VARIABLES,
)

log = logging.getLogger(__name__)
ARCHIVE_LAG_DAYS = 6


def _params(**extra) -> dict:
    lats = ",".join(str(c[0]) for c in CITIES.values())
    lons = ",".join(str(c[1]) for c in CITIES.values())
    return {
        "latitude": lats,
        "longitude": lons,
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "UTC",
        **extra,
    }


def fetch(raw_dir: Path, today: date | None = None) -> list[Path]:
    today = today or date.today()
    folder = raw_dir / "weather"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    archive_end = today - timedelta(days=ARCHIVE_LAG_DAYS)
    for year in range(FIRST_YEAR, archive_end.year + 1):
        target = folder / f"archive_{year}.json"
        final = year < archive_end.year
        if target.exists() and final:
            paths.append(target)
            continue
        end = date(year, 12, 31) if final else archive_end
        data = http.get_json(
            METEO_ARCHIVE_URL,
            _params(start_date=f"{year}-01-01", end_date=end.isoformat()),
            timeout=300,
        )
        target.write_text(json.dumps(data))
        paths.append(target)
    recent = folder / "forecast.json"
    data = http.get_json(METEO_FORECAST_URL, _params(past_days=14, forecast_days=3))
    recent.write_text(json.dumps(data))
    paths.append(recent)
    return paths


def national(data: list[dict] | dict) -> pd.DataFrame:
    """Weighted average across cities, one row per hour."""
    locations = data if isinstance(data, list) else [data]
    weights = np.array([c[2] for c in CITIES.values()], dtype=float)
    frames = []
    for loc in locations:
        h = loc["hourly"]
        frames.append(pd.DataFrame({v: h[v] for v in WEATHER_VARIABLES}, index=h["time"]))
    stacked = np.stack([f[list(WEATHER_VARIABLES)].to_numpy(dtype=float) for f in frames])
    w = weights[: len(frames), None, None]
    mask = ~np.isnan(stacked)
    avg = np.nansum(stacked * w, axis=0) / np.where(mask, w, 0).sum(axis=0)
    out = pd.DataFrame(avg, columns=list(WEATHER_VARIABLES))
    out.insert(0, "time_utc", pd.to_datetime(frames[0].index).tz_localize("UTC"))
    return out.dropna(subset=["temperature_2m"])


def clean(paths: list[Path], clean_dir: Path) -> Path:
    parts = []
    for p in paths:
        df = national(json.loads(p.read_text()))
        df["source"] = "forecast" if p.name == "forecast.json" else "archive"
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    # Prefer the archive (measured) over the forecast where both exist.
    df["rank"] = (df["source"] == "forecast").astype(int)
    df = df.sort_values(["time_utc", "rank"]).drop_duplicates("time_utc").drop(columns="rank")
    clean_dir.mkdir(parents=True, exist_ok=True)
    target = clean_dir / "weather.parquet"
    df.reset_index(drop=True).to_parquet(target, index=False)
    log.info("weather: %s hours up to %s", f"{len(df):,}", df["time_utc"].max())
    return target
