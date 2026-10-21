"""Small HTTP helpers with retries."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from energyforecast.config import USER_AGENT

log = logging.getLogger(__name__)
session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def _retry(fn, attempts: int = 4):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except requests.RequestException as exc:
            if attempt == attempts:
                raise
            log.warning("request failed (%s), retrying", exc)
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def get_json(url: str, params: dict | None = None, timeout: int = 120):
    def go():
        r = session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()

    return _retry(go)


def download(url: str, target: Path, timeout: int = 300) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")

    def go():
        with session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        tmp.replace(target)
        return target

    _retry(go)
    log.info("downloaded %s (%.1f MB)", target.name, target.stat().st_size / 1e6)
    return target
