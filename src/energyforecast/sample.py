"""Synthetic data in the same clean schemas, for tests, CI and working offline.

Demand follows a daily shape, a weekly pattern, the seasons and the temperature, so the
model has something real to learn. The numbers are invented.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from energyforecast.timeutil import periods_for_day


def write_sample(clean_dir: Path, start: str = "2023-01-01", days_ahead: int = 2, seed: int = 3):
    rng = np.random.default_rng(seed)
    today = date.today()
    end = today - timedelta(days=1)
    days = pd.date_range(start, end, freq="D")
    grid = pd.concat([periods_for_day(d) for d in days], ignore_index=True)

    hours = pd.date_range(
        grid["start_utc"].min(),
        pd.Timestamp(today + timedelta(days=days_ahead + 1), tz="UTC"),
        freq="h",
    )
    doy = hours.dayofyear.to_numpy()
    hod = hours.hour.to_numpy()
    temp = 11 - 7 * np.cos(2 * np.pi * (doy - 15) / 365) + 3 * np.sin(2 * np.pi * (hod - 9) / 24)
    temp = temp + rng.normal(0, 1.5, len(hours))
    sun = np.clip(np.sin(np.pi * (hod - 6) / 12), 0, None) * (
        300 + 250 * np.sin(2 * np.pi * (doy - 80) / 365)
    )
    weather = pd.DataFrame(
        {
            "time_utc": hours,
            "temperature_2m": temp,
            "shortwave_radiation": np.clip(sun * rng.uniform(0.3, 1, len(hours)), 0, None),
            "wind_speed_10m": rng.gamma(3, 5, len(hours)),
            "cloud_cover": rng.uniform(0, 100, len(hours)),
            "source": "archive",
        }
    )

    holidays = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-12-25",
                    "2023-12-26",
                    "2024-12-25",
                    "2024-12-26",
                    "2025-12-25",
                    "2025-12-26",
                    "2026-12-25",
                    "2026-12-26",
                ]
            )
        }
    )

    hourly_temp = weather.set_index("time_utc")["temperature_2m"]
    t = hourly_temp.reindex(grid["start_utc"].dt.floor("h")).to_numpy()
    local_hour = grid["start_utc"].dt.tz_convert("Europe/London").dt.hour.to_numpy()
    dow = grid["settlement_date"].dt.dayofweek.to_numpy()
    shape = (
        24000
        + 7000 * np.exp(-((local_hour - 18) ** 2) / 8)
        + 4000 * ((local_hour >= 7) & (local_hour <= 21))
    )
    nd = shape * np.where(dow >= 5, 0.88, 1.0) + 450 * np.clip(15.5 - t, 0, None)
    nd = nd * np.where(grid["settlement_date"].isin(holidays["date"]), 0.85, 1.0)
    nd = nd + rng.normal(0, 400, len(grid))
    demand = grid.copy()
    demand["nd"] = nd.round()
    demand["tsd"] = (nd * 1.05).round()
    demand["wind_gen"] = 2000.0
    demand["wind_cap"] = 6500.0
    demand["solar_gen"] = 0.0
    demand["solar_cap"] = 15000.0
    demand = demand[
        [
            "settlement_date",
            "settlement_period",
            "nd",
            "tsd",
            "wind_gen",
            "wind_cap",
            "solar_gen",
            "solar_cap",
            "start_utc",
        ]
    ]

    neso = demand[["settlement_date", "settlement_period"]].copy()
    neso["neso_outturn"] = demand["nd"].to_numpy()
    neso["neso_forecast"] = (demand["nd"] * rng.normal(1, 0.03, len(demand))).round()
    neso = neso[neso["settlement_date"] >= "2024-06-01"]

    clean_dir.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(clean_dir / "demand.parquet", index=False)
    weather.to_parquet(clean_dir / "weather.parquet", index=False)
    holidays.to_parquet(clean_dir / "holidays.parquet", index=False)
    neso.to_parquet(clean_dir / "neso_forecast.parquet", index=False)
    return clean_dir
