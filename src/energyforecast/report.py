"""Build the static results page from the backtest, the latest forecast and the track record."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import pandas as pd
from jinja2 import Template

from energyforecast import config, model, pipeline

log = logging.getLogger(__name__)


def build_site(clean_dir: Path, artifact_dir: Path, history_dir: Path, site_dir: Path) -> Path:
    data = collect(clean_dir, artifact_dir, history_dir)
    template = Template(files("energyforecast").joinpath("templates/index.html").read_text())
    site_dir.mkdir(parents=True, exist_ok=True)
    safe = json.dumps(data, separators=(",", ":"), default=str).replace("</", "<\\/")
    (site_dir / "index.html").write_text(template.render(data_json=safe), encoding="utf-8")
    (site_dir / "data.json").write_text(json.dumps(data, indent=1, default=str))
    log.info("site written to %s", site_dir)
    return site_dir


def collect(clean_dir: Path, artifact_dir: Path, history_dir: Path) -> dict:
    meta = json.loads((artifact_dir / "model.json").read_text())
    bt = pd.read_parquet(artifact_dir / "backtest.parquet")
    imp = pd.read_csv(artifact_dir / "importance.csv")
    latest_path = artifact_dir / "latest_forecast.json"
    latest = json.loads(latest_path.read_text()) if latest_path.exists() else None

    # Daily error in the backtest, for the model and for NESO, on the same half hours.
    both = bt.dropna(subset=["neso_forecast", "neso_outturn"]).copy()
    daily = (
        both.groupby("settlement_date")
        .apply(
            lambda g: pd.Series(
                {
                    "model": model.mape(g["neso_outturn"], g["p50"]),
                    "neso": model.mape(g["neso_outturn"], g["neso_forecast"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    monthly = (
        both.assign(month=both["settlement_date"].dt.to_period("M").astype(str))
        .groupby("month")
        .apply(
            lambda g: pd.Series(
                {
                    "model": model.mape(g["neso_outturn"], g["p50"]),
                    "neso": model.mape(g["neso_outturn"], g["neso_forecast"]),
                    "naive": model.mape(g["neso_outturn"], g["seasonal_naive"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    wins = int((daily["model"] < daily["neso"]).sum()) if len(daily) else 0

    # A sample week from the backtest, half hour by half hour.
    week_end = bt["settlement_date"].max()
    week = bt[bt["settlement_date"] > week_end - pd.Timedelta(days=7)]

    # Track record of forecasts actually published by the daily job.
    hist = pipeline.score_history(history_dir, clean_dir)
    track = []
    if len(hist):
        scored = hist.dropna(subset=[config.TARGET])
        for day, g in scored.groupby("settlement_date"):
            row = {"date": str(day.date()), "model": model.mape(g[config.TARGET], g["p50"])}
            if "neso_forecast" in g and g["neso_forecast"].notna().any():
                row["neso"] = model.mape(g["neso_outturn"], g["neso_forecast"])
            track.append(row)

    return {
        "built_at": datetime.now(UTC).isoformat(timespec="minutes"),
        "meta": meta,
        "latest": latest,
        "daily": daily.assign(settlement_date=daily["settlement_date"].astype(str))
        .round(3)
        .to_dict("records"),
        "monthly": monthly.round(3).to_dict("records"),
        "days_compared": int(len(daily)),
        "days_model_better": wins,
        "week": [
            {
                "t": r.start_utc.isoformat(),
                "actual": r.actual,
                "p10": round(r.p10),
                "p50": round(r.p50),
                "p90": round(r.p90),
                "neso": None if pd.isna(r.neso_forecast) else r.neso_forecast,
            }
            for r in week.itertuples()
        ],
        "importance": imp.head(12).round(4).to_dict("records"),
        "track": track,
    }
