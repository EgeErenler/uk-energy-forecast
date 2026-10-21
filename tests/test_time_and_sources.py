import io

import pandas as pd
import pytest

from energyforecast.sources import demand, holidays, neso_forecast, weather
from energyforecast.timeutil import period_start_utc, periods_for_day


@pytest.mark.parametrize(
    ("day", "periods"),
    [("2026-03-29", 46), ("2026-06-01", 48), ("2026-10-25", 50)],
)
def test_clock_change_days_have_the_right_number_of_periods(day, periods):
    df = periods_for_day(day)
    assert len(df) == periods
    assert df["start_utc"].is_monotonic_increasing


def test_period_start_follows_london_time():
    starts = period_start_utc(pd.Series(["2026-01-15", "2026-07-15"]), pd.Series([1, 1]))
    assert starts.iloc[0] == pd.Timestamp("2026-01-15 00:00", tz="UTC")
    assert starts.iloc[1] == pd.Timestamp("2026-07-14 23:00", tz="UTC")  # BST


def test_demand_files_with_both_date_formats(tmp_path):
    old = (
        "SETTLEMENT_DATE,SETTLEMENT_PERIOD,ND,TSD,EMBEDDED_SOLAR_CAPACITY\n"
        "01-JAN-2019,1,23808,25291,12663\n"
    )
    new = (
        "SETTLEMENT_DATE,SETTLEMENT_PERIOD,ND,TSD,EMBEDDED_SOLAR_CAPACITY,FORECAST_ACTUAL_INDICATOR\n"
        '"2026-10-19",1,24000,26000,22000,A\n"2026-10-22",1,25000,27000,22000,F\n'
    )
    (tmp_path / "a.csv").write_text(old)
    (tmp_path / "b.csv").write_text(new)
    target = demand.clean([tmp_path / "a.csv", tmp_path / "b.csv"], tmp_path)
    df = pd.read_parquet(target)
    assert df["settlement_date"].dt.strftime("%Y-%m-%d").tolist() == ["2019-01-01", "2026-10-19"]
    assert df["solar_cap"].tolist() == [12663, 22000]


def test_iso_dates_are_never_read_day_first():
    parsed = demand.parse_dates(pd.Series(['"2025-12-09"', "09-DEC-2025", "09-Dec-25"]))
    assert parsed.dt.strftime("%Y-%m-%d").tolist() == ["2025-12-09"] * 3
    with pytest.raises(ValueError):
        demand.parse_dates(pd.Series(["December 9"]))


def test_yearly_files_are_found_by_name():
    resources = [
        {"name": "Historic Demand Data 2025", "url": "https://x/demanddata_2025.csv"},
        {"name": "Frequently Asked Questions (FAQ)", "url": "https://x/faq.docx"},
    ]
    assert demand.yearly_files(resources) == {2025: "https://x/demanddata_2025.csv"}


def test_neso_forecast_parse():
    csv = (
        "﻿Month,Date,Datetime,Settlement_Period,Demand_Forecast,Demand_Outturn,X\n"
        "Oct,2026-10-19,2026-10-19T23:30:00Z,47,26762,24639,0\n"
    )
    df = neso_forecast.parse(pd.read_csv(io.StringIO(csv), encoding="utf-8-sig"))
    assert df.iloc[0][["neso_forecast", "neso_outturn"]].tolist() == [26762, 24639]


def test_weather_is_population_weighted():
    hourly = lambda t: {  # noqa: E731
        "time": ["2026-01-01T00:00"],
        "temperature_2m": [t],
        "shortwave_radiation": [0],
        "wind_speed_10m": [10],
        "cloud_cover": [50],
    }
    from energyforecast.config import CITIES

    data = [{"hourly": hourly(10.0)}] + [{"hourly": hourly(0.0)} for _ in range(len(CITIES) - 1)]
    out = weather.national(data)
    london = list(CITIES.values())[0][2]
    total = sum(c[2] for c in CITIES.values())
    assert out["temperature_2m"].iloc[0] == pytest.approx(10 * london / total)


def test_holidays_include_2018():
    df = holidays.parse({"england-and-wales": {"events": [{"date": "2026-12-25"}]}})
    assert pd.Timestamp("2018-12-25") in set(df["date"])


def test_elexon_fills_only_the_missing_days():
    from energyforecast.sources import elexon

    days = pd.date_range("2026-09-01", "2026-09-10")
    rows = [(d, p) for d in days for p in range(1, 49)]
    demand_df = pd.DataFrame(rows, columns=["settlement_date", "settlement_period"])
    demand_df["nd"] = 25000.0
    demand_df["start_utc"] = period_start_utc(
        demand_df["settlement_date"], demand_df["settlement_period"]
    )
    more = pd.date_range("2026-09-01", "2026-09-12")
    indo = pd.DataFrame(
        [(d, p) for d in more for p in range(1, 49)],
        columns=["settlement_date", "settlement_period"],
    )
    indo["indo"] = 24000.0 + indo["settlement_period"]  # a steady offset from ND
    demand_df["nd"] = 25000.0 + demand_df["settlement_period"]
    out = elexon.fill_recent(demand_df, indo)
    added = out[out["source"] == "elexon"]
    assert set(added["settlement_date"].dt.day) == {11, 12}
    assert added["nd"].sub(25000 + added["settlement_period"]).abs().max() < 1


def test_weather_months_end_at_the_last_archive_day():
    from datetime import date

    m = weather.months(date(2025, 11, 1), date(2026, 1, 15))
    assert m == [
        (date(2025, 11, 1), date(2025, 11, 30)),
        (date(2025, 12, 1), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 1, 15)),
    ]
