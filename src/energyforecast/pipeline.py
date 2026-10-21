"""The steps the CLI and the daily workflow run: train, forecast and keep a record."""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from energyforecast import config, embedded, features, model
from energyforecast.sources import elexon

log = logging.getLogger(__name__)


def load_clean(clean_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("demand", "weather", "holidays", "neso_forecast", "elexon", "weather_dayahead"):
        path = clean_dir / f"{name}.parquet"
        out[name] = pd.read_parquet(path) if path.exists() else None
    out["demand"] = elexon.fill_recent(out["demand"], out["elexon"])
    return out


def tomorrow() -> date:
    return (pd.Timestamp.now(tz=config.TIMEZONE) + pd.Timedelta(days=1)).date()


def train(clean_dir: Path, artifact_dir: Path) -> dict:
    """Backtest, then train the final models on everything and save them."""
    data = load_clean(clean_dir)
    base = features.base(data["demand"], data["weather"], data["holidays"])
    test_weather, weather_note = backtest_weather(data["weather"], data["weather_dayahead"])
    base_bt = features.base(data["demand"], test_weather, data["holidays"])
    bt = model.backtest(base_bt, data["neso_forecast"])
    bt.metrics["test_weather"] = weather_note
    log.info("backtest: %s", json.dumps({k: v for k, v in bt.metrics.items()}, default=str))

    gen_models = embedded.fit(base)
    frame = features.demand_features(embedded.apply(base, gen_models))
    final, width = model.fit(model.training_rows(frame))
    meta = {
        "trained_at": datetime.now(UTC).isoformat(timespec="minutes"),
        "trained_until": str(
            frame.loc[frame[config.TARGET].notna(), "settlement_date"].max().date()
        ),
        "features": features.FEATURES,
        "params": model.PARAMS,
        "trees": {str(q): m.num_trees() for q, m in final.items()},
        "interval_width": width,
        "backtest": bt.metrics,
    }
    model.save(final, meta, artifact_dir)
    embedded.save(gen_models, artifact_dir)
    bt.predictions.to_parquet(artifact_dir / "backtest.parquet", index=False)
    model.importance(final).to_csv(artifact_dir / "importance.csv", index=False)
    log_to_mlflow(meta, artifact_dir)
    return meta


def backtest_weather(observed: pd.DataFrame, day_ahead: pd.DataFrame | None):
    """Observed weather before the test period; inside it, the forecast issued the day
    before wherever it exists. This is the information the model would really have had."""
    if day_ahead is None or day_ahead.empty:
        return observed, "observed"
    start = pd.Timestamp(config.TEST_START, tz="UTC")
    obs = observed.assign(time_utc=pd.to_datetime(observed["time_utc"], utc=True))
    ahead = day_ahead.assign(time_utc=pd.to_datetime(day_ahead["time_utc"], utc=True))
    ahead = ahead[ahead["time_utc"] >= start].dropna(subset=["temperature_2m"])
    merged = pd.concat([obs[~obs["time_utc"].isin(ahead["time_utc"])], ahead])
    merged = merged.sort_values("time_utc").reset_index(drop=True)
    test_hours = (obs["time_utc"] >= start).sum()
    share = len(ahead) / max(1, test_hours)
    log.info("backtest weather: day-ahead forecasts for %.0f%% of test hours", share * 100)
    return merged, "day-ahead" if share > 0.95 else f"day-ahead for {share:.0%} of hours"


def log_to_mlflow(meta: dict, artifact_dir: Path) -> None:
    """Record the run so experiments can be compared later. Never fails the pipeline."""
    try:
        import os

        # A plain folder of runs is enough here, and it needs no database server.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        mlflow.set_tracking_uri(config.MLFLOW_URI)
        mlflow.set_experiment("day-ahead-demand")
        with mlflow.start_run(run_name=f"train-{meta['trained_until']}"):
            mlflow.log_params({**meta["params"], "features": len(meta["features"])})
            mlflow.log_params({f"trees_{k}": v for k, v in meta["trees"].items()})
            mlflow.log_metrics(
                {k: v for k, v in meta["backtest"].items() if isinstance(v, int | float)}
            )
            for f in ("model.json", "importance.csv"):
                mlflow.log_artifact(str(artifact_dir / f))
    except Exception as exc:  # noqa: BLE001
        log.warning("MLflow logging skipped: %s", exc)


def forecast_day(day: date, clean_dir: Path, artifact_dir: Path) -> pd.DataFrame:
    """Forecast every half hour of one day with the saved models."""
    data = load_clean(clean_dir)
    models, meta = model.load(artifact_dir)
    frame = features.build(
        data["demand"], data["weather"], data["holidays"], embedded.load(artifact_dir), [day]
    )
    rows = frame[frame["settlement_date"] == pd.Timestamp(day)]
    if rows.empty or rows["temperature"].isna().all():
        raise ValueError(f"No weather forecast available for {day}")
    if rows["lag_2d"].isna().all():
        raise ValueError(f"Demand data is too old to forecast {day}")
    pred = model.predict(models, rows, meta.get("interval_width", 0.0))
    pred["temperature"] = rows["temperature"].to_numpy()
    pred["issued_at"] = datetime.now(UTC).isoformat(timespec="minutes")
    pred["model_trained_until"] = meta["trained_until"]
    return pred


def record(pred: pd.DataFrame, history_dir: Path) -> Path:
    """Keep every forecast we publish, so it can be scored once the day has happened.
    The first forecast issued for a half hour is kept; later runs never overwrite it."""
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / "forecasts.csv"
    cols = ["settlement_date", "settlement_period", "p10", "p50", "p90", "issued_at"]
    new = pred[cols].copy()
    new[["p10", "p50", "p90"]] = new[["p10", "p50", "p90"]].round()
    new["settlement_date"] = pd.to_datetime(new["settlement_date"]).dt.strftime("%Y-%m-%d")
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(["settlement_date", "settlement_period"], keep="first")
    new.sort_values(["settlement_date", "settlement_period"]).to_csv(path, index=False)
    return path


def run_forecast(clean_dir: Path, artifact_dir: Path, history_dir: Path, day=None) -> pd.DataFrame:
    day = day or tomorrow()
    pred = forecast_day(day, clean_dir, artifact_dir)
    record(pred, history_dir)
    latest = {
        "date": str(day),
        "issued_at": pred["issued_at"].iloc[0],
        "model_trained_until": pred["model_trained_until"].iloc[0],
        "half_hours": [
            {
                "period": int(r.settlement_period),
                "start_utc": r.start_utc.isoformat(),
                "p10": round(r.p10),
                "p50": round(r.p50),
                "p90": round(r.p90),
            }
            for r in pred.itertuples()
        ],
    }
    (artifact_dir / "latest_forecast.json").write_text(json.dumps(latest, indent=1))
    log.info(
        "forecast for %s: peak %.0f MW, total %.1f GWh",
        day,
        pred["p50"].max(),
        pred["p50"].sum() / 2 / 1000,
    )
    return pred


def score_history(history_dir: Path, clean_dir: Path) -> pd.DataFrame:
    """Join published forecasts with what actually happened and with NESO's forecast."""
    path = history_dir / "forecasts.csv"
    if not path.exists():
        return pd.DataFrame()
    hist = pd.read_csv(path, parse_dates=["settlement_date"])
    data = load_clean(clean_dir)
    actual = data["demand"][["settlement_date", "settlement_period", config.TARGET]]
    out = hist.merge(actual, on=["settlement_date", "settlement_period"], how="left")
    if data["neso_forecast"] is not None:
        out = out.merge(
            data["neso_forecast"], on=["settlement_date", "settlement_period"], how="left"
        )
    return out


def recent_days(n: int) -> list[date]:
    today = pd.Timestamp.now(tz=config.TIMEZONE).date()
    return [today - timedelta(days=i) for i in range(n, 0, -1)]
