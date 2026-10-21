"""A small HTTP API for the forecasts.

Run it with `energyforecast serve` or the Docker image, then open /docs for the
interactive documentation.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from energyforecast import config, pipeline

app = FastAPI(
    title="UK Energy Demand Forecast",
    description="Day-ahead forecasts of Great Britain's national electricity demand, "
    "every half hour, with an 80% range.",
    version="0.1.0",
)


def _model_meta() -> dict:
    path = config.ARTIFACT_DIR / "model.json"
    if not path.exists():
        raise HTTPException(503, "No trained model yet. Run `energyforecast train` first.")
    return json.loads(path.read_text())


@app.get("/health")
def health() -> dict:
    ready = (config.ARTIFACT_DIR / "model.json").exists()
    return {"status": "ok", "model_ready": ready}


@app.get("/model")
def model_info() -> dict:
    meta = _model_meta()
    return {k: meta[k] for k in ("trained_at", "trained_until", "features", "trees", "backtest")}


@app.get("/forecast/latest")
def latest() -> dict:
    path = config.ARTIFACT_DIR / "latest_forecast.json"
    if not path.exists():
        raise HTTPException(404, "No forecast has been published yet.")
    return json.loads(path.read_text())


@app.get("/forecast")
def forecast(
    day: Annotated[date, Query(description="Local date in Great Britain, YYYY-MM-DD")],
) -> dict:
    _model_meta()
    try:
        pred = pipeline.forecast_day(day, config.CLEAN_DIR, config.ARTIFACT_DIR)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "date": str(day),
        "unit": "MW",
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
