"""Train, evaluate, save and load the forecasting models.

Three LightGBM models are trained with a quantile loss: the median is the forecast, and the
10th and 90th percentiles give an 80% range. The backtest trains only on data before
TEST_START and scores every half hour after it, against two baselines:

- seasonal naive: the same half hour one week earlier
- NESO: the official day-ahead forecast, the bar to beat
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from energyforecast import config, embedded, features
from energyforecast.features import FEATURES, RATIO_TARGET

log = logging.getLogger(__name__)

PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
}
ROUNDS = 2000
EARLY_STOP = 100
# The newest part of the training data picks the number of trees and calibrates the range.
VALIDATION_SHARE = 0.1


def training_rows(df: pd.DataFrame, until: str | None = None) -> pd.DataFrame:
    rows = df[df[RATIO_TARGET].notna() & df["lag_2d"].notna()]
    if "source" in rows:
        rows = rows[rows["source"].fillna("neso") == "neso"]  # Elexon rows are history only
    for start, end in config.EXCLUDE_FROM_TRAINING:
        rows = rows[~rows["settlement_date"].between(start, end)]
    if until:
        rows = rows[rows["settlement_date"] < until]
    return rows


def fit(train: pd.DataFrame, quantiles=config.QUANTILES) -> tuple[dict[float, lgb.Booster], float]:
    """One model per quantile. The last 10% of the training period picks the number of trees
    and calibrates the range (see `conformal_width`). Returns the models and that width."""
    train = train.sort_values("start_utc")
    cut = int(len(train) * (1 - VALIDATION_SHARE))
    fit_part, val_part = train.iloc[:cut], train.iloc[cut:]
    models, first_pass = {}, {}
    for q in quantiles:
        params = {**PARAMS, "objective": "quantile", "alpha": q}
        dtrain = lgb.Dataset(fit_part[FEATURES], fit_part[RATIO_TARGET])
        dval = lgb.Dataset(val_part[FEATURES], val_part[RATIO_TARGET], reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=ROUNDS,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )
        first_pass[q] = booster
        # Refit on all training data with the chosen number of trees.
        full = lgb.Dataset(train[FEATURES], train[RATIO_TARGET])
        models[q] = lgb.train(params, full, num_boost_round=booster.best_iteration or ROUNDS)
        log.info("quantile %.1f: %d trees", q, models[q].num_trees())
    width = conformal_width(first_pass, val_part)
    log.info("80%% range widened by %.4f of the level", width)
    return models, width


def conformal_width(
    models: dict[float, lgb.Booster], val: pd.DataFrame, level: float = 0.8
) -> float:
    """Conformalised quantile regression: how much to widen the 10-90 range so that it
    really holds about 80% of outcomes on data the trees did not see."""
    lo = models[min(models)].predict(
        val[FEATURES], num_iteration=models[min(models)].best_iteration
    )
    hi = models[max(models)].predict(
        val[FEATURES], num_iteration=models[max(models)].best_iteration
    )
    # Score against grid demand plus the *estimated* embedded generation, so the range
    # also covers errors in the solar and wind estimates, not just in underlying demand.
    y = ((val[config.TARGET] + val["solar_est"] + val["wind_est"]) / val["level_7d"]).to_numpy()
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    k = min(1.0, np.ceil((n + 1) * level) / n)
    return float(max(0.0, np.quantile(scores, k)))


def predict(
    models: dict[float, lgb.Booster], rows: pd.DataFrame, width: float = 0.0
) -> pd.DataFrame:
    out = rows[["settlement_date", "settlement_period", "start_utc"]].copy()
    level = rows["level_7d"].to_numpy()
    hidden = (rows["solar_est"] + rows["wind_est"]).to_numpy()
    cols = [f"p{int(q * 100)}" for q in sorted(models)]
    ratio = np.column_stack([models[q].predict(rows[FEATURES]) for q in sorted(models)]).astype(
        float
    )
    # Quantile models are trained separately, so keep them in order.
    ratio = np.sort(ratio, axis=1)
    ratio[:, 0] -= width
    ratio[:, -1] += width
    # Underlying demand, minus the embedded solar and wind the grid never sees.
    out[cols] = ratio * level[:, None] - hidden[:, None]
    return out


def mape(actual, forecast) -> float:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    ok = ~(np.isnan(actual) | np.isnan(forecast)) & (actual != 0)
    return float(np.mean(np.abs(forecast[ok] - actual[ok]) / actual[ok]) * 100)


def mae(actual, forecast) -> float:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    ok = ~(np.isnan(actual) | np.isnan(forecast))
    return float(np.mean(np.abs(forecast[ok] - actual[ok])))


@dataclass
class Backtest:
    predictions: pd.DataFrame
    metrics: dict


def backtest(base: pd.DataFrame, neso: pd.DataFrame | None) -> Backtest:
    """Train before TEST_START, forecast every half hour after it, compare with baselines.
    The embedded generation models are also trained on the earlier period only."""
    gen_models = embedded.fit(base, until=config.TEST_START)
    df = features.demand_features(embedded.apply(base, gen_models))
    train = training_rows(df, until=config.TEST_START)
    test = training_rows(df)
    test = test[test["settlement_date"] >= config.TEST_START]
    models, width = fit(train)
    pred = predict(models, test, width)
    pred["actual"] = test[config.TARGET].to_numpy()
    pred["seasonal_naive"] = test["nd_lag_7d"].to_numpy()
    if neso is not None and len(neso):
        pred = pred.merge(neso, on=["settlement_date", "settlement_period"], how="left")
    else:
        pred["neso_forecast"] = np.nan
        pred["neso_outturn"] = np.nan

    both = pred.dropna(subset=["neso_forecast", "neso_outturn"])
    inside = (pred["actual"] >= pred["p10"]) & (pred["actual"] <= pred["p90"])
    metrics = {
        "test_start": config.TEST_START,
        "test_end": str(pred["settlement_date"].max().date()),
        "test_half_hours": int(len(pred)),
        "model_mape": mape(pred["actual"], pred["p50"]),
        "model_mae_mw": mae(pred["actual"], pred["p50"]),
        "naive_mape": mape(pred["actual"], pred["seasonal_naive"]),
        "interval_coverage": float(inside.mean() * 100),
        # On the half hours where NESO's forecast exists, both are scored against
        # NESO's own published outturn, so the comparison is like for like.
        "compared_half_hours": int(len(both)),
        "model_mape_vs_outturn": mape(both["neso_outturn"], both["p50"]),
        "neso_mape": mape(both["neso_outturn"], both["neso_forecast"]),
    }
    return Backtest(pred, metrics)


def save(models: dict[float, lgb.Booster], meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for q, m in models.items():
        m.save_model(str(out_dir / f"model_p{int(q * 100)}.txt"))
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2, default=str))


def load(out_dir: Path) -> tuple[dict[float, lgb.Booster], dict]:
    meta = json.loads((out_dir / "model.json").read_text())
    models = {
        q: lgb.Booster(model_file=str(out_dir / f"model_p{int(q * 100)}.txt"))
        for q in config.QUANTILES
    }
    return models, meta


def importance(models: dict[float, lgb.Booster]) -> pd.DataFrame:
    m = models[0.5]
    gain = m.feature_importance(importance_type="gain")
    df = pd.DataFrame({"feature": m.feature_name(), "gain": gain})
    df["share"] = df["gain"] / df["gain"].sum()
    return df.sort_values("share", ascending=False).reset_index(drop=True)
