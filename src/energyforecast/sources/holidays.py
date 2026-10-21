"""Bank holidays in England and Wales, where most of the demand is."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from energyforecast import http
from energyforecast.config import BANK_HOLIDAYS_URL

log = logging.getLogger(__name__)

# GOV.UK only lists recent years, so 2018 is added by hand.
EXTRA = [
    "2018-01-01", "2018-03-30", "2018-04-02", "2018-05-07",
    "2018-05-28", "2018-08-27", "2018-12-25", "2018-12-26",
]  # fmt: skip


def fetch(raw_dir: Path) -> Path:
    return http.download(BANK_HOLIDAYS_URL, raw_dir / "holidays" / "bank-holidays.json")


def parse(data: dict) -> pd.DataFrame:
    events = data["england-and-wales"]["events"]
    dates = {e["date"] for e in events} | set(EXTRA)
    return pd.DataFrame({"date": pd.to_datetime(sorted(dates))})


def clean(json_path: Path, clean_dir: Path) -> Path:
    import json

    df = parse(json.loads(json_path.read_text()))
    clean_dir.mkdir(parents=True, exist_ok=True)
    target = clean_dir / "holidays.parquet"
    df.to_parquet(target, index=False)
    log.info("holidays: %d days", len(df))
    return target
