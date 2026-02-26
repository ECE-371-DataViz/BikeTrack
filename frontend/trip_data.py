"""
CitiBike historic trip data downloader and manager.
Dynamically downloads trip data from 2024 to present from CitiBike's S3 bucket.
"""

import io
import logging
import os
import zipfile
from datetime import date

import pandas as pd
import requests

TRIPDATA_BASE_URL = "https://s3.amazonaws.com/tripdata"
_HERE = os.path.dirname(os.path.abspath(__file__))
TRIP_CACHE_DIR = os.path.join(_HERE, "..", "processed_data", "trips")

logger = logging.getLogger(__name__)

TRIP_COLUMNS = [
    "started_at",
    "ended_at",
    "start_station_id",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
]


def get_available_months():
    """Return (year, month) tuples from January 2024 through today."""
    result, year, month = [], 2024, 1
    today = date.today()
    while (year, month) <= (today.year, today.month):
        result.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return result


def _zip_url(year: int, month: int) -> str:
    return f"{TRIPDATA_BASE_URL}/{year}{month:02d}-citibike-tripdata.csv.zip"


def _cache_path(year: int, month: int) -> str:
    os.makedirs(TRIP_CACHE_DIR, exist_ok=True)
    return os.path.join(TRIP_CACHE_DIR, f"{year}{month:02d}-citibike-tripdata.csv.gz")


def download_month(year: int, month: int):
    """
    Download and cache trip data for the given month.
    Returns the local cache path on success, or None if unavailable.
    Skips download if a cache file already exists.
    """
    path = _cache_path(year, month)
    if os.path.exists(path):
        return path

    url = _zip_url(year, month)
    logger.info("Downloading %s", url)
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code == 404:
            logger.debug("No data for %d-%02d", year, month)
            return None
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_files:
                return None
            with zf.open(csv_files[0]) as fh:
                df = pd.read_csv(
                    fh,
                    usecols=lambda c: c in TRIP_COLUMNS,
                    dtype={"start_station_id": str, "end_station_id": str},
                )

        df = df.dropna(
            subset=[
                "start_station_id",
                "end_station_id",
                "started_at",
                "ended_at",
                "start_lat",
                "start_lng",
                "end_lat",
                "end_lng",
            ]
        )
        df = df[df["start_station_id"] != df["end_station_id"]]
        df.to_csv(path, index=False, compression="gzip")
        logger.info("Saved %d trips for %d-%02d", len(df), year, month)
        return path
    except Exception as exc:
        if hasattr(exc, 'response') and exc.response is not None and exc.response.status_code != 404:
            logger.error("HTTP %s downloading %d-%02d: %s", exc.response.status_code, year, month, exc)
        else:
            logger.error("Failed to download %d-%02d: %s", year, month, exc)
        return None


def load_trips(sample_per_month: int = 5000) -> pd.DataFrame:
    """
    Load (downloading if necessary) CitiBike trip data from 2024 to present.
    Returns a DataFrame sorted by started_at with a duration_seconds column.
    Returns an empty DataFrame if no data is available.
    """
    frames = []
    for year, month in get_available_months():
        path = _cache_path(year, month)
        if not os.path.exists(path):
            path = download_month(year, month)
        if not path or not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(
                path,
                compression="gzip",
                dtype={"start_station_id": str, "end_station_id": str},
            )
            if len(df) > sample_per_month:
                df = df.sample(n=sample_per_month, random_state=42)
            frames.append(df)
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["started_at"] = pd.to_datetime(combined["started_at"], errors="coerce")
    combined["ended_at"] = pd.to_datetime(combined["ended_at"], errors="coerce")
    combined = combined.dropna(subset=["started_at", "ended_at"])
    combined["duration_seconds"] = (
        (combined["ended_at"] - combined["started_at"]).dt.total_seconds()
    )
    # Keep sane durations: 1 minute to 3 hours
    combined = combined[
        (combined["duration_seconds"] >= 60) & (combined["duration_seconds"] <= 10800)
    ]
    combined = combined.sort_values("started_at").reset_index(drop=True)
    return combined
