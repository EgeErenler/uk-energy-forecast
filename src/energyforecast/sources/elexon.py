"""Elexon BMRS: initial national demand outturn, published within the hour.

NESO's historic demand files arrive about three weeks late, but the forecast needs demand
from two days ago. Elexon's initial outturn (INDO) fills that gap. It is measured a little
differently from NESO's National Demand, so it is calibrated on the days both cover.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from energyforecast import http

log = logging.getLogger(__name__)
URL = "https://data.elexon.co.uk/bmrs/api/v1/demand/outturn"
DAYS = 90
CHUNK = 7


def fetch(raw_dir: Path, today: date | None = None) -> Path:
    today = today or date.today()
    start = today - timedelta(days=DAYS)
    rows = []
    while start <= today:
        end = min(start + timedelta(days=CHUNK - 1), today)
        data = http.get_json(
            URL,
            {
                "settlementDateFrom": start.isoformat(),
                "settlementDateTo": end.isoformat(),
                "format": "json",
            },
        )
        rows.extend(data.get("data", data) if isinstance(data, dict) else data)
        start = end + timedelta(days=1)
    target = raw_dir / "elexon" / "indo.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target, index=False)
    return target


def parse(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "settlement_date": pd.to_datetime(df["settlementDate"]).dt.normalize(),
            "settlement_period": df["settlementPeriod"].astype(int),
            "indo": pd.to_numeric(df["initialDemandOutturn"], errors="coerce"),
        }
    )
    return out.dropna().drop_duplicates(["settlement_date", "settlement_period"], keep="last")


def clean(path: Path, clean_dir: Path) -> Path:
    df = parse(pd.read_parquet(path))
    clean_dir.mkdir(parents=True, exist_ok=True)
    target = clean_dir / "elexon.parquet"
    df.to_parquet(target, index=False)
    log.info("elexon: %s half hours up to %s", f"{len(df):,}", df["settlement_date"].max().date())
    return target


def fill_recent(demand: pd.DataFrame, indo: pd.DataFrame | None) -> pd.DataFrame:
    """Add the days NESO has not published yet, using calibrated Elexon outturn.

    A straight line ND = a * INDO + b is fitted on the half hours both sources cover.
    The added rows are marked with source = "elexon", so they are never used as a
    training target, only as recent history for the lag features.
    """
    demand = demand.copy()
    demand["source"] = "neso"
    if indo is None or indo.empty:
        return demand
    keys = ["settlement_date", "settlement_period"]
    both = demand.merge(indo, on=keys)
    if len(both) >= 48 * 7:
        a, b = np.polyfit(both["indo"], both["nd"], 1)
        err = np.mean(np.abs(a * both["indo"] + b - both["nd"]) / both["nd"]) * 100
        log.info(
            "elexon calibration on %d half hours: ND = %.4f * INDO + %.0f (%.2f%% error)",
            len(both),
            a,
            b,
            err,
        )
    else:
        a, b = 1.0, 0.0
        log.warning("too little overlap with NESO data, using Elexon outturn as it is")
    last = demand["settlement_date"].max()
    new = indo[indo["settlement_date"] > last].copy()
    if new.empty:
        return demand
    new["nd"] = (a * new["indo"] + b).round()
    new["source"] = "elexon"
    from energyforecast.timeutil import period_start_utc

    new["start_utc"] = period_start_utc(new["settlement_date"], new["settlement_period"])
    return pd.concat([demand, new.drop(columns="indo")], ignore_index=True)
