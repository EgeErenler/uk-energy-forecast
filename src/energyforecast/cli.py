"""energyforecast fetch | train | forecast | site | serve | sample | daily"""

from __future__ import annotations

import argparse
import logging
import sys

from energyforecast import config

SOURCES = ("demand", "elexon", "neso", "holidays", "weather")


def fetch(names) -> None:
    from energyforecast.sources import demand, elexon, holidays, neso_forecast, weather

    raw, clean = config.RAW_DIR, config.CLEAN_DIR
    steps = {
        "demand": lambda: demand.clean(demand.fetch(raw), clean),
        "elexon": lambda: elexon.clean(elexon.fetch(raw), clean),
        "neso": lambda: neso_forecast.clean(neso_forecast.fetch(raw), clean),
        "holidays": lambda: holidays.clean(holidays.fetch(raw), clean),
        "weather": lambda: weather.clean(weather.fetch(raw), clean),
    }
    for name in names:
        logging.info("fetching %s", name)
        steps[name]()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="energyforecast", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download and clean the data")
    f.add_argument("sources", nargs="*", help=f"any of: {', '.join(SOURCES)} (default: all)")
    sub.add_parser("train", help="backtest, then train the final models")
    fc = sub.add_parser("forecast", help="forecast one day (default: tomorrow)")
    fc.add_argument("--day", help="YYYY-MM-DD")
    sub.add_parser("site", help="build the static results page")
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    sub.add_parser("sample", help="write synthetic data for offline work")
    sub.add_parser("daily", help="fetch, train, forecast tomorrow and build the site")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from energyforecast import pipeline

    if args.command == "fetch":
        unknown = set(args.sources) - set(SOURCES)
        if unknown:
            parser.error(f"unknown source(s): {', '.join(sorted(unknown))}")
        fetch(args.sources or SOURCES)
    elif args.command == "train":
        pipeline.train(config.CLEAN_DIR, config.ARTIFACT_DIR)
    elif args.command == "forecast":
        import pandas as pd

        day = pd.Timestamp(args.day).date() if args.day else None
        pipeline.run_forecast(config.CLEAN_DIR, config.ARTIFACT_DIR, config.HISTORY_DIR, day)
    elif args.command == "site":
        from energyforecast.report import build_site

        build_site(config.CLEAN_DIR, config.ARTIFACT_DIR, config.HISTORY_DIR, config.SITE_DIR)
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("energyforecast.api:app", host=args.host, port=args.port)
    elif args.command == "sample":
        from energyforecast.sample import write_sample

        write_sample(config.CLEAN_DIR)
    elif args.command == "daily":
        from energyforecast.report import build_site

        fetch(SOURCES)
        pipeline.train(config.CLEAN_DIR, config.ARTIFACT_DIR)
        pipeline.run_forecast(config.CLEAN_DIR, config.ARTIFACT_DIR, config.HISTORY_DIR)
        build_site(config.CLEAN_DIR, config.ARTIFACT_DIR, config.HISTORY_DIR, config.SITE_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
