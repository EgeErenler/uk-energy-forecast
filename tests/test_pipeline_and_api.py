import json

import pytest
from fastapi.testclient import TestClient

from energyforecast import config, model, pipeline
from energyforecast.sample import write_sample


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    clean, art, hist = root / "clean", root / "artifacts", root / "history"
    write_sample(clean, start="2024-06-01")
    old = (config.TEST_START, model.ROUNDS)
    config.TEST_START, model.ROUNDS = "2026-01-01", 200
    try:
        meta = pipeline.train(clean, art)
        pipeline.run_forecast(clean, art, hist)
    finally:
        config.TEST_START, model.ROUNDS = old
    return clean, art, hist, meta


def test_model_beats_the_naive_baseline_on_sample_data(trained):
    _, _, _, meta = trained
    bt = meta["backtest"]
    assert bt["model_mape"] < bt["naive_mape"]
    assert 50 < bt["interval_coverage"] <= 100


def test_forecast_is_recorded_once(trained):
    clean, art, hist, _ = trained
    first = (hist / "forecasts.csv").read_text()
    pipeline.run_forecast(clean, art, hist)  # a second run must not overwrite the first
    assert (hist / "forecasts.csv").read_text().splitlines()[1:] == first.splitlines()[1:]


def test_api(trained, monkeypatch):
    clean, art, _, _ = trained
    monkeypatch.setattr(config, "ARTIFACT_DIR", art)
    monkeypatch.setattr(config, "CLEAN_DIR", clean)
    from energyforecast.api import app

    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok", "model_ready": True}
    latest = client.get("/forecast/latest").json()
    assert len(latest["half_hours"]) in (46, 48, 50)
    day = latest["date"]
    r = client.get("/forecast", params={"day": day})
    assert r.status_code == 200
    p = r.json()["half_hours"][20]
    assert p["p10"] <= p["p50"] <= p["p90"]
    assert client.get("/forecast", params={"day": "2030-01-01"}).status_code == 422
    assert "backtest" in client.get("/model").json()
    json.dumps(latest)
