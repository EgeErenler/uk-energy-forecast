"""NESO historic demand: national electricity demand for every half hour since 2018."""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

import pandas as pd

from energyforecast import http
from energyforecast.config import DEMAND_DATASET, FIRST_YEAR, NESO_API
from energyforecast.timeutil import period_start_utc

log = logging.getLogger(__name__)

KEEP = {
    "settlement_date": "settlement_date",
    "settlement_period": "settlement_period",
    "nd": "nd",
    "tsd": "tsd",
    "embedded_wind_generation": "wind_gen",
    "embedded_wind_capacity": "wind_cap",
    "embedded_solar_generation": "solar_gen",
    "embedded_solar_capacity": "solar_cap",
}


def yearly_files(resources: list[dict]) -> dict[int, str]:
    """Map each year to the URL of its CSV, from the dataset's resource list."""
    out = {}
    for r in resources:
        m = re.search(r"(\d{4})", r.get("name") or "")
        if m and (r.get("url") or "").lower().endswith(".csv"):
            out[int(m.group(1))] = r["url"]
    return out


def fetch(raw_dir: Path, today: date | None = None) -> list[Path]:
    today = today or date.today()
    meta = http.get_json(NESO_API.format(dataset=DEMAND_DATASET))
    files = yearly_files(meta["result"]["resources"])
    paths = []
    for year in range(FIRST_YEAR, today.year + 1):
        if year not in files:
            log.warning("no demand file listed for %d", year)
            continue
        target = raw_dir / "demand" / f"demand_{year}.csv"
        # Older years are final. The current file grows every day, and the previous one
        # can still be revised early in the year.
        if not target.exists() or year >= today.year - 1:
            http.download(files[year], target)
        paths.append(target)
    return paths


def parse_dates(values: pd.Series) -> pd.Series:
    """Files write dates as 01-JAN-2019, 01-Jan-23 or 2025-12-09.
    Each format is parsed explicitly: guessing would read 2025-12-09 as 12 September."""
    text = values.astype(str).str.strip().str.strip('"')
    iso = text.str.match(r"^\d{4}-\d{2}-\d{2}")
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    out[iso] = pd.to_datetime(text[iso].str[:10], format="%Y-%m-%d")
    rest = text[~iso].str.title()
    for fmt in ("%d-%b-%Y", "%d-%b-%y", "%d/%m/%Y"):
        todo = out.index[~iso & out.isna()]
        if len(todo) == 0:
            break
        out[todo] = pd.to_datetime(rest[todo], format=fmt, errors="coerce")
    if out.isna().any():
        raise ValueError(f"Unknown date format, for example {text[out.isna()].iloc[0]!r}")
    return out


def read_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    if "forecast_actual_indicator" in df:
        df = df[df["forecast_actual_indicator"].astype(str).str.upper().eq("A")]
    df["settlement_date"] = parse_dates(df["settlement_date"])
    cols = [c for c in KEEP if c in df]
    return df[cols].rename(columns=KEEP)


def clean(paths: list[Path], clean_dir: Path) -> Path:
    df = pd.concat([read_file(p) for p in paths], ignore_index=True)
    df = df[df["nd"] > 0]
    df = df.drop_duplicates(["settlement_date", "settlement_period"], keep="last")
    df = df.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)
    df["start_utc"] = period_start_utc(df["settlement_date"], df["settlement_period"])
    clean_dir.mkdir(parents=True, exist_ok=True)
    target = clean_dir / "demand.parquet"
    df.to_parquet(target, index=False)
    log.info(
        "demand: %s half hours, %s to %s",
        f"{len(df):,}",
        df["settlement_date"].min().date(),
        df["settlement_date"].max().date(),
    )
    return target
