"""Embedded solar and wind: generation the grid does not see directly.

Thousands of rooftop panels and small wind farms feed local networks, so they lower the
demand NESO measures. NESO publishes its estimate of their output. Two small models learn
that output as a share of installed capacity, from the weather, so it can be estimated
for days where it is not known yet, such as tomorrow.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from energyforecast.config import TARGET

SOLAR_FEATURES = [
    "solar_radiation",
    "solar_cloud",
    "radiation",
    "cloud_cover",
    "settlement_period",
    "doy_sin",
    "doy_cos",
    "temperature",
]
WIND_FEATURES = ["wind_100m", "wind_speed", "settlement_period", "doy_sin", "doy_cos"]
PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "verbose": -1,
}
ROUNDS = 600
SPEC = {
    "solar": ("solar_gen", "solar_cap", SOLAR_FEATURES),
    "wind": ("wind_gen", "wind_cap", WIND_FEATURES),
}


def fit(df: pd.DataFrame, until: str | None = None) -> dict[str, lgb.Booster]:
    models = {}
    for name, (gen, cap, feats) in SPEC.items():
        rows = df[df[gen].notna() & df[cap].notna() & (df[cap] > 0)]
        if "source" in rows:
            rows = rows[rows["source"].fillna("neso") == "neso"]
        if until:
            rows = rows[rows["settlement_date"] < until]
        share = (rows[gen] / rows[cap]).clip(0, 1)
        models[name] = lgb.train(PARAMS, lgb.Dataset(rows[feats], share), ROUNDS)
    return models


def apply(df: pd.DataFrame, models: dict[str, lgb.Booster]) -> pd.DataFrame:
    """Add solar_est and wind_est (MW), and underlying demand where demand is known.
    Underlying demand uses NESO's own estimate of embedded generation when it exists."""
    df = df.copy()
    for name, (gen, cap, feats) in SPEC.items():
        share = np.clip(models[name].predict(df[feats]), 0, 1)
        df[f"{name}_est"] = share * df[cap]
        if gen not in df:
            df[gen] = np.nan
    df["underlying"] = (
        df[TARGET] + df["solar_gen"].fillna(df["solar_est"]) + df["wind_gen"].fillna(df["wind_est"])
    )
    return df


def save(models: dict[str, lgb.Booster], out_dir: Path) -> None:
    for name, m in models.items():
        m.save_model(str(out_dir / f"embedded_{name}.txt"))


def load(out_dir: Path) -> dict[str, lgb.Booster]:
    return {name: lgb.Booster(model_file=str(out_dir / f"embedded_{name}.txt")) for name in SPEC}
