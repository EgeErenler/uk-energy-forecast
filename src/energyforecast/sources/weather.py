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
    SOLAR_SITES,
    SOLAR_VARIABLES,
    WEATHER_VARIABLES,
    WIND_SITES,
    WIND_VARIABLES,
)

log = logging.getLogger(__name__)
ARCHIVE_LAG_DAYS = 6


def _params(sites: dict = CITIES, variables=WEATHER_VARIABLES, **extra) -> dict:
    return {
        "latitude": ",".join(str(c[0]) for c in sites.values()),
        "longitude": ",".join(str(c[1]) for c in sites.values()),
        "hourly": ",".join(variables),
        "timezone": "UTC",
        **extra,
    }


SETS = {
    # name: (sites, variables)
    "archive": (CITIES, WEATHER_VARIABLES),
    "solar": (SOLAR_SITES, SOLAR_VARIABLES),
    "wind": (WIND_SITES, WIND_VARIABLES),
}


def months(start: date, end: date) -> list[tuple[date, date]]:
    """Calendar months from start to end, the last one cut at end."""
    out, cur = [], date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        out.append((cur, min(nxt - timedelta(days=1), end)))
        cur = nxt
    return out


def fetch(raw_dir: Path, today: date | None = None) -> list[Path]:
    """One file per month. Finished months never change and are kept; only the current
    month is downloaded again. This keeps the daily run far below Open-Meteo's limits."""
    today = today or date.today()
    folder = raw_dir / "weather"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    archive_end = today - timedelta(days=ARCHIVE_LAG_DAYS)
    for prefix, (sites, variables) in SETS.items():
        for first, last in months(date(FIRST_YEAR, 1, 1), archive_end):
            target = folder / f"{prefix}_{first:%Y-%m}.json"
            finished = last.month != archive_end.month or last.year != archive_end.year
            if not (target.exists() and finished):
                params = _params(
                    sites, variables, start_date=first.isoformat(), end_date=last.isoformat()
                )
                target.write_text(json.dumps(http.get_json(METEO_ARCHIVE_URL, params, timeout=300)))
            paths.append(target)
    for prefix, (sites, variables) in SETS.items():
        recent = folder / f"{prefix}_forecast.json"
        params = _params(sites, variables, past_days=14, forecast_days=3)
        recent.write_text(json.dumps(http.get_json(METEO_FORECAST_URL, params)))
        paths.append(recent)
    return paths


def national(data: list[dict] | dict, variables=WEATHER_VARIABLES, weights=None) -> pd.DataFrame:
    """Weighted average across places, one row per hour. By default the cities,
    weighted by population."""
    locations = data if isinstance(data, list) else [data]
    if weights is None:
        weights = [c[2] for c in CITIES.values()]
    weights = np.array(weights, dtype=float)
    variables = list(variables)
    frames = []
    for loc in locations:
        h = loc["hourly"]
        frames.append(pd.DataFrame({v: h[v] for v in variables}, index=h["time"]))
    stacked = np.stack([f[variables].to_numpy(dtype=float) for f in frames])
    w = weights[: len(frames), None, None]
    mask = ~np.isnan(stacked)
    avg = np.nansum(stacked * w, axis=0) / np.where(mask, w, 0).sum(axis=0)
    out = pd.DataFrame(avg, columns=variables)
    out.insert(0, "time_utc", pd.to_datetime(frames[0].index).tz_localize("UTC"))
    return out.dropna(subset=[variables[0]])


def _combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    # Prefer the archive (measured) over the forecast where both exist.
    df["rank"] = (df["source"] == "forecast").astype(int)
    df = df.sort_values(["time_utc", "rank"]).drop_duplicates("time_utc")
    return df.drop(columns="rank").reset_index(drop=True)


def clean(paths: list[Path], clean_dir: Path) -> Path:
    people, solar, wind = [], [], []
    for p in paths:
        data = json.loads(p.read_text())
        source = "forecast" if p.name.endswith("_forecast.json") else "archive"
        if p.name.startswith("wind_"):
            df = national(data, WIND_VARIABLES, weights=[1.0] * len(WIND_SITES))
            wind.append(df.assign(source=source))
        elif p.name.startswith("solar_"):
            df = national(data, SOLAR_VARIABLES, weights=[1.0] * len(SOLAR_SITES))
            solar.append(df.assign(source=source))
        else:
            people.append(national(data).assign(source=source))
    df = _combine(people)
    sun = _combine(solar).rename(
        columns={"shortwave_radiation": "solar_radiation", "cloud_cover": "solar_cloud"}
    )
    df = df.merge(sun.drop(columns="source"), on="time_utc", how="left")
    if wind:
        breeze = _combine(wind).rename(columns={"wind_speed_100m": "wind_100m"})
        df = df.merge(breeze.drop(columns="source"), on="time_utc", how="left")
    clean_dir.mkdir(parents=True, exist_ok=True)
    target = clean_dir / "weather.parquet"
    df.to_parquet(target, index=False)
    log.info("weather: %s hours up to %s", f"{len(df):,}", df["time_utc"].max())
    return target
