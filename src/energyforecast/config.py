"""Paths, sources and model settings. Paths are relative to the repository root."""

from __future__ import annotations

import os
from pathlib import Path

RAW_DIR = Path(os.getenv("EF_RAW_DIR", "data/raw"))
CLEAN_DIR = Path(os.getenv("EF_CLEAN_DIR", "data/clean"))
ARTIFACT_DIR = Path(os.getenv("EF_ARTIFACT_DIR", "artifacts"))
SITE_DIR = Path(os.getenv("EF_SITE_DIR", "site"))
HISTORY_DIR = Path(os.getenv("EF_HISTORY_DIR", "history"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

USER_AGENT = "uk-energy-forecast (+https://github.com/EgeErenler/uk-energy-forecast)"
TIMEZONE = "Europe/London"

# NESO (National Energy System Operator) open data, through its CKAN API.
NESO_API = "https://api.neso.energy/api/3/action/package_show?id={dataset}"
DEMAND_DATASET = "historic-demand-data"
FIRST_YEAR = 2018
# NESO's own day-ahead forecast for every half hour, with the outturn, from April 2021.
NESO_FORECAST_URL = (
    "https://api.neso.energy/dataset/5281b494-5566-42dd-bf1a-9ce38642fabf/resource/"
    "08e41551-80f8-4e28-a416-ea473a695db9/download/1b-incentive.csv"
)

BANK_HOLIDAYS_URL = "https://www.gov.uk/bank-holidays.json"

# Weather from Open-Meteo: ERA5 history for training, the forecast for tomorrow.
METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Forecasts as they were issued one day earlier. The backtest uses these for the test
# period, so the model sees the same kind of weather information it gets when live.
METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
WEATHER_VARIABLES = ("temperature_2m", "shortwave_radiation", "wind_speed_10m", "cloud_cover")

# Large cities, weighted by population, stand in for "the weather where people use power".
CITIES = {
    "London": (51.507, -0.128, 8.9),
    "Birmingham": (52.486, -1.890, 2.9),
    "Manchester": (53.480, -2.242, 2.8),
    "Leeds": (53.801, -1.549, 1.9),
    "Glasgow": (55.864, -4.252, 1.7),
    "Newcastle": (54.978, -1.618, 1.1),
    "Bristol": (51.455, -2.588, 1.0),
    "Cardiff": (51.481, -3.179, 0.5),
}

# Embedded solar sits mostly in the south and west, not where people live, so sunshine
# is measured at a second set of places spread over the main solar regions.
SOLAR_SITES = {
    "Truro": (50.263, -5.051),
    "Exeter": (50.718, -3.534),
    "Taunton": (51.015, -3.106),
    "Swindon": (51.558, -1.782),
    "Southampton": (50.909, -1.404),
    "Brighton": (50.822, -0.137),
    "Ashford": (51.146, 0.875),
    "Cambridge": (52.205, 0.122),
    "Norwich": (52.630, 1.297),
    "Peterborough": (52.573, -0.241),
    "Lincoln": (53.230, -0.540),
    "Hereford": (52.056, -2.716),
}
SOLAR_VARIABLES = ("shortwave_radiation", "cloud_cover")

# Embedded wind farms are mostly in windy, rural places: Scotland, Wales, the South West
# and the east coast. Wind at 100 m is close to turbine height.
WIND_SITES = {
    "Aberdeenshire": (57.30, -2.40),
    "Highlands": (57.50, -4.30),
    "Borders": (55.50, -2.80),
    "Dumfries": (55.10, -3.70),
    "Cumbria": (54.60, -3.10),
    "North Wales": (53.00, -3.60),
    "Mid Wales": (52.40, -3.60),
    "Cornwall": (50.40, -4.90),
    "Devon": (50.80, -3.90),
    "Lincolnshire": (53.20, -0.20),
    "Norfolk": (52.75, 1.10),
    "Yorkshire": (53.90, -1.20),
}
WIND_VARIABLES = ("wind_speed_100m",)

TARGET = "nd"  # National Demand, in MW

# Training and testing. The test year is never seen during training.
TEST_START = os.getenv("EF_TEST_START", "2025-01-01")
# The first months of COVID-19 changed demand in a way that will not repeat.
EXCLUDE_FROM_TRAINING = (("2020-03-16", "2020-07-31"),)

# The forecast for day D is made on the morning of D-1, when the newest full day of
# demand data is D-2. Full-day demand features are therefore at least two days old.
# The only newer demand used is the night of D-1 (00:00 to 06:00), which is published
# before the forecast runs.
MIN_LAG_DAYS = 2
NIGHT_PERIODS = 12

QUANTILES = (0.1, 0.5, 0.9)
