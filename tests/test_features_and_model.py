import numpy as np
import pandas as pd

from energyforecast import embedded, features, model
from energyforecast.sample import write_sample


def frame(tmp_path, forecast_days=None):
    write_sample(tmp_path, start="2025-09-01")
    d = {n: pd.read_parquet(tmp_path / f"{n}.parquet") for n in ("demand", "weather", "holidays")}
    base = features.base(d["demand"], d["weather"], d["holidays"])
    gen = embedded.fit(base)
    return d, features.build(d["demand"], d["weather"], d["holidays"], gen, forecast_days)


def test_lags_are_two_or_more_days_old(tmp_path):
    d, df = frame(tmp_path)
    row = df.dropna(subset=["lag_2d", "lag_7d"]).iloc[100]
    # Lags use underlying demand: grid demand plus the embedded generation it hides.
    under = d["demand"].assign(u=lambda x: x["nd"] + x["solar_gen"] + x["wind_gen"])
    demand = under.set_index(["settlement_date", "settlement_period"])["u"]
    key = row["settlement_period"]
    assert row["lag_2d"] == demand[(row["settlement_date"] - pd.Timedelta(days=2), key)]
    assert row["lag_7d"] == demand[(row["settlement_date"] - pd.Timedelta(days=7), key)]


def test_forecast_day_uses_no_demand_from_that_day(tmp_path):
    tomorrow = (
        (pd.Timestamp.now(tz="Europe/London") + pd.Timedelta(days=1)).normalize().tz_localize(None)
    )
    _, df = frame(tmp_path, forecast_days=[tomorrow])
    rows = df[df["settlement_date"] == tomorrow]
    assert len(rows) in (46, 48, 50)
    assert rows["nd"].isna().all()
    assert rows["lag_2d"].notna().all()  # yesterday's demand is known
    assert rows["temperature"].notna().all()  # from the weather forecast


def test_every_feature_exists(tmp_path):
    _, df = frame(tmp_path)
    assert set(features.FEATURES) <= set(df.columns)


def test_metrics():
    assert model.mape([100, 200], [110, 180]) == 10.0
    assert model.mae([100, 200], [110, 180]) == 15.0
    assert model.mape([100, np.nan], [110, 1]) == 10.0


def test_quantiles_are_kept_in_order():
    class Fixed:
        def __init__(self, v):
            self.v = v

        def predict(self, x):
            return np.full(len(x), self.v)

    rows = pd.DataFrame(
        {
            "settlement_date": [1],
            "settlement_period": [1],
            "start_utc": [1],
            **{f: [1] for f in features.FEATURES},
        }
    ).assign(solar_est=0, wind_est=0)
    out = model.predict({0.1: Fixed(5), 0.5: Fixed(3), 0.9: Fixed(4)}, rows)
    assert out[["p10", "p50", "p90"]].iloc[0].tolist() == [3, 4, 5]
