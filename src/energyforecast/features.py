"""Turn demand, weather and holidays into one row of features per half hour.

Every feature must be known on the morning before the target day. Demand is only used
from two days back or more (see MIN_LAG_DAYS), and weather for the target day is the
forecast when the model runs live.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energyforecast.config import MIN_LAG_DAYS, TARGET
from energyforecast.timeutil import periods_for_day

FEATURES = [
    "settlement_period",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "is_christmas",
    "doy_sin",
    "doy_cos",
    "years_since_2018",
    "temperature",
    "temperature_24h",
    "heating_degrees",
    "radiation",
    "solar_potential",
    "wind_speed",
    "cloud_cover",
    "level_7d",
    "lag_2d_ratio",
    "lag_7d_ratio",
    "lag_14d_ratio",
    "daily_mean_2d_ratio",
    "daily_peak_2d_ratio",
]

# The model predicts demand as a multiple of the recent level (the average of the
# 7 newest known days). Demand in GB has been falling for years, and tree models cannot
# extrapolate a trend; predicting the shape around a known level avoids that problem.
RATIO_TARGET = "target_ratio"


def half_hourly_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Hourly weather interpolated to the start of each half hour."""
    w = weather.assign(
        time_utc=pd.to_datetime(weather["time_utc"], utc=True).astype("datetime64[ns, UTC]")
    )
    w = w.set_index("time_utc").sort_index()
    cols = ["temperature_2m", "shortwave_radiation", "wind_speed_10m", "cloud_cover"]
    idx = pd.date_range(w.index.min(), w.index.max(), freq="30min")
    w = w[cols].reindex(w.index.union(idx)).interpolate(method="time").reindex(idx)
    w = w.rename(
        columns={
            "temperature_2m": "temperature",
            "shortwave_radiation": "radiation",
            "wind_speed_10m": "wind_speed",
            "cloud_cover": "cloud_cover",
        }
    )
    # Buildings react to the last day of weather, not just the current half hour.
    w["temperature_24h"] = w["temperature"].rolling(48, min_periods=12).mean()
    return w.rename_axis("start_utc").reset_index()


def with_future_days(demand: pd.DataFrame, days: list) -> pd.DataFrame:
    """Add empty rows for days we want to forecast."""
    extra = [periods_for_day(d) for d in days]
    if not extra:
        return demand
    future = pd.concat(extra, ignore_index=True)
    future = future[
        ~future.set_index(["settlement_date", "settlement_period"]).index.isin(
            demand.set_index(["settlement_date", "settlement_period"]).index
        )
    ]
    return pd.concat([demand, future], ignore_index=True)


def build(
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    holidays: pd.DataFrame,
    forecast_days: list | None = None,
) -> pd.DataFrame:
    df = with_future_days(demand.copy(), forecast_days or [])
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.normalize()
    # Parquet files and pandas can label UTC with different time zone objects; unify them.
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True).astype("datetime64[ns, UTC]")
    df = df.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)

    # Calendar
    d = df["settlement_date"]
    df["day_of_week"] = d.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    holiday_set = set(pd.to_datetime(holidays["date"]).dt.normalize())
    df["is_holiday"] = d.isin(holiday_set).astype(int)
    md = d.dt.month * 100 + d.dt.day
    df["is_christmas"] = ((md >= 1222) | (md <= 102)).astype(int)
    doy = d.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["years_since_2018"] = (d - pd.Timestamp("2018-01-01")).dt.days / 365.25

    # Weather
    df = df.merge(half_hourly_weather(weather), on="start_utc", how="left")
    df["heating_degrees"] = (15.5 - df["temperature"]).clip(lower=0)

    # Embedded solar is not metered, so sunshine lowers the demand the grid sees.
    # Capacity grows every year; the latest known value is carried forward.
    cap = df["solar_cap"].ffill() if "solar_cap" in df else pd.Series(np.nan, index=df.index)
    df["solar_potential"] = cap * df["radiation"] / 1000

    # Demand lags, at least MIN_LAG_DAYS old
    keys = ["settlement_date", "settlement_period"]
    for days in (MIN_LAG_DAYS, 7, 14):
        name = "lag_2d" if days == MIN_LAG_DAYS else f"lag_{days}d"
        lag = df[[*keys, TARGET]].dropna().rename(columns={TARGET: name})
        lag["settlement_date"] = lag["settlement_date"] + pd.Timedelta(days=days)
        df = df.merge(lag, on=keys, how="left")

    daily = df.groupby("settlement_date")[TARGET].agg(["mean", "max"])
    daily.index = daily.index + pd.Timedelta(days=MIN_LAG_DAYS)
    df["daily_mean_2d"] = df["settlement_date"].map(daily["mean"])
    df["daily_peak_2d"] = df["settlement_date"].map(daily["max"])

    # Recent level: mean of the 7 newest known days (D-8 to D-2).
    day_mean = df.groupby("settlement_date")[TARGET].mean()
    full = day_mean.reindex(pd.date_range(day_mean.index.min(), df["settlement_date"].max()))
    level = full.rolling(7, min_periods=5).mean().shift(MIN_LAG_DAYS)
    df["level_7d"] = df["settlement_date"].map(level)
    for col in ("lag_2d", "lag_7d", "lag_14d", "daily_mean_2d", "daily_peak_2d"):
        df[f"{col}_ratio"] = df[col] / df["level_7d"]
    df[RATIO_TARGET] = df[TARGET] / df["level_7d"]
    return df
