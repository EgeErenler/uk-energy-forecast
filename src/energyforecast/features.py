"""Turn demand, weather and holidays into one row of features per half hour.

Every feature must be known on the morning before the target day. Full days of demand are
only used from two days back (see MIN_LAG_DAYS), and the weather for the target day is the
forecast when the model runs live.

The work happens in two steps, because the embedded solar and wind estimates (see
`embedded.py`) sit between them:

1. `base`: calendar and weather for every half hour.
2. `demand_features`: recent levels and lags of underlying demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energyforecast.config import MIN_LAG_DAYS, NIGHT_PERIODS, TARGET
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
    "temperature_change_2d",
    "temperature_change_7d",
    "radiation",
    "cloud_cover",
    "wind_speed",
    "solar_est",
    "wind_est",
    "level_7d",
    "lag_2d_ratio",
    "lag_7d_ratio",
    "lag_14d_ratio",
    "daily_mean_2d_ratio",
    "daily_peak_2d_ratio",
    "night_1d_ratio",
    "night_1d_vs_2d",
]

# Grid demand (National Demand) is what people use minus what small solar panels and wind
# farms produce out of sight of the grid. The model forecasts the smoother underlying demand
# as a multiple of its recent level, then takes the embedded generation estimate away.
# Predicting around a known level also avoids a weakness of tree models: they cannot
# extrapolate the long fall in GB demand.
UNDERLYING = "underlying"
RATIO_TARGET = "target_ratio"
KEYS = ["settlement_date", "settlement_period"]


def half_hourly_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Hourly weather interpolated to the start of each half hour."""
    w = weather.assign(
        time_utc=pd.to_datetime(weather["time_utc"], utc=True).astype("datetime64[ns, UTC]")
    )
    w = w.set_index("time_utc").sort_index()
    names = {
        "temperature_2m": "temperature",
        "shortwave_radiation": "radiation",
        "wind_speed_10m": "wind_speed",
        "cloud_cover": "cloud_cover",
        "solar_radiation": "solar_radiation",
        "solar_cloud": "solar_cloud",
        "wind_100m": "wind_100m",
    }
    cols = [c for c in names if c in w]
    idx = pd.date_range(w.index.min(), w.index.max(), freq="30min")
    w = w[cols].reindex(w.index.union(idx)).interpolate(method="time").reindex(idx)
    w = w.rename(columns=names)
    for col, fallback in (
        ("solar_radiation", "radiation"),
        ("solar_cloud", "cloud_cover"),
        ("wind_100m", "wind_speed"),
    ):
        if col not in w:
            w[col] = w[fallback]
    # Buildings react to the last day of weather, not just the current half hour.
    w["temperature_24h"] = w["temperature"].rolling(48, min_periods=12).mean()
    return w.rename_axis("start_utc").reset_index()


def with_future_days(demand: pd.DataFrame, days: list) -> pd.DataFrame:
    """Add empty rows for days we want to forecast."""
    extra = [periods_for_day(d) for d in days]
    if not extra:
        return demand
    future = pd.concat(extra, ignore_index=True)
    known = demand.set_index(KEYS).index
    future = future[~future.set_index(KEYS).index.isin(known)]
    return pd.concat([demand, future], ignore_index=True)


def base(
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    holidays: pd.DataFrame,
    forecast_days: list | None = None,
) -> pd.DataFrame:
    """Calendar and weather for every half hour, plus empty rows for forecast days."""
    df = with_future_days(demand.copy(), forecast_days or [])
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.normalize()
    # Parquet files and pandas can label UTC with different time zone objects; unify them.
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True).astype("datetime64[ns, UTC]")
    df = df.sort_values(KEYS).reset_index(drop=True)

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

    df = df.merge(half_hourly_weather(weather), on="start_utc", how="left")
    df["heating_degrees"] = (15.5 - df["temperature"]).clip(lower=0)
    # Installed capacity is published with the demand data; carry the latest value forward.
    for cap in ("solar_cap", "wind_cap"):
        if cap not in df:
            df[cap] = np.nan
        df[cap] = df[cap].ffill().bfill()

    for days in (MIN_LAG_DAYS, 7):
        past = df[[*KEYS, "temperature"]].rename(columns={"temperature": "t_past"})
        past["settlement_date"] = past["settlement_date"] + pd.Timedelta(days=days)
        df = df.merge(past, on=KEYS, how="left")
        df[f"temperature_change_{days}d"] = df["temperature"] - df["t_past"]
        df = df.drop(columns="t_past")
    return df


def _shift_days(df: pd.DataFrame, col: str, days: int, name: str) -> pd.DataFrame:
    lag = df[[*KEYS, col]].dropna().rename(columns={col: name})
    lag["settlement_date"] = lag["settlement_date"] + pd.Timedelta(days=days)
    return df.merge(lag, on=KEYS, how="left")


def demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recent level, lags and the known night, all for underlying demand."""
    u = UNDERLYING
    for days in (MIN_LAG_DAYS, 7, 14):
        df = _shift_days(df, u, days, f"lag_{days}d")
    # Last week's grid demand, for the seasonal naive baseline.
    df = _shift_days(df, TARGET, 7, "nd_lag_7d")

    daily = df.groupby("settlement_date")[u].agg(["mean", "max"])
    daily.index = daily.index + pd.Timedelta(days=MIN_LAG_DAYS)
    df["daily_mean_2d"] = df["settlement_date"].map(daily["mean"])
    df["daily_peak_2d"] = df["settlement_date"].map(daily["max"])

    # Recent level: mean of the 7 newest full days (D-8 to D-2).
    day_mean = df.groupby("settlement_date")[u].mean()
    full = day_mean.reindex(pd.date_range(day_mean.index.min(), df["settlement_date"].max()))
    level = full.rolling(7, min_periods=5).mean().shift(MIN_LAG_DAYS)
    df["level_7d"] = df["settlement_date"].map(level)
    for col in ("lag_2d", "lag_7d", "lag_14d", "daily_mean_2d", "daily_peak_2d"):
        df[f"{col}_ratio"] = df[col] / df["level_7d"]

    # The night of D-1 (00:00 to 06:00) is published before the forecast runs.
    night = df[df["settlement_period"] <= NIGHT_PERIODS].groupby("settlement_date")[u].mean()
    night_1d = df["settlement_date"].map(night.set_axis(night.index + pd.Timedelta(days=1)))
    night_2d = df["settlement_date"].map(night.set_axis(night.index + pd.Timedelta(days=2)))
    df["night_1d_ratio"] = night_1d / df["level_7d"]
    df["night_1d_vs_2d"] = night_1d / night_2d
    df[RATIO_TARGET] = df[u] / df["level_7d"]
    return df


def build(demand, weather, holidays, embedded_models, forecast_days=None) -> pd.DataFrame:
    """All features, given trained embedded generation models."""
    from energyforecast import embedded

    df = base(demand, weather, holidays, forecast_days)
    df = embedded.apply(df, embedded_models)
    return demand_features(df)
