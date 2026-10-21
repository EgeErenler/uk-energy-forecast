"""Settlement periods and time zones.

GB electricity is settled in half hours numbered from 1 each local day. Most days have
48 periods, but the day the clocks go forward has 46 and the day they go back has 50.
"""

from __future__ import annotations

import pandas as pd

from energyforecast.config import TIMEZONE


def period_start_utc(dates: pd.Series, periods: pd.Series) -> pd.Series:
    """UTC start time of each (settlement date, settlement period)."""
    local_midnight = pd.to_datetime(dates).dt.tz_localize(TIMEZONE)
    utc_midnight = local_midnight.dt.tz_convert("UTC")
    return utc_midnight + pd.to_timedelta((periods.astype(int) - 1) * 30, unit="min")


def periods_for_day(day) -> pd.DataFrame:
    """Every settlement period of one local day, with its UTC start time."""
    day = pd.Timestamp(day).normalize()
    start = day.tz_localize(TIMEZONE).tz_convert("UTC")
    end = (day + pd.Timedelta(days=1)).tz_localize(TIMEZONE).tz_convert("UTC")
    starts = pd.date_range(start, end, freq="30min", inclusive="left")
    return pd.DataFrame(
        {
            "settlement_date": day,
            "settlement_period": range(1, len(starts) + 1),
            "start_utc": starts,
        }
    )
