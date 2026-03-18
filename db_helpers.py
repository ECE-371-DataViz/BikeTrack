#!/usr/bin/env python3
"""
Database helper functions for one-time setup tasks.
These functions are only called explicitly and should not run during normal app operation.
"""
import pandas as pd
import requests
import io
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from sqlalchemy import text
from globals import *


def migrate_schema(db_manager):
    """Lightweight schema migration for columns added after initial deployment.

    Creates PostGIS extensions, adds geometry columns, and performs schema updates.
    Safe to call multiple times - uses IF NOT EXISTS/IF NOT ALREADY EXISTS patterns.

    Args:
        db_manager: DBManager instance with engine and Session_eng
    """
    try:
        with db_manager.engine.begin() as conn:
            # Enable PostGIS extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_topology"))

            # Add geom column and spatial index
            conn.execute(
                text(
                    "ALTER TABLE stations ADD COLUMN IF NOT EXISTS geom geometry(Point,4326)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_station_geom ON stations USING GIST (geom)"
                )
            )

            # Populate geom column from lat/lon for existing stations
            conn.execute(
                text(
                    "UPDATE stations SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) "
                    "WHERE geom IS NULL"
                )
            )

            # Existing migrations
            conn.execute(
                text(
                    "ALTER TABLE app_metadata ADD COLUMN IF NOT EXISTS num_trips INTEGER DEFAULT 10"
                )
            )

            # Migrate route table: old schema had station_id as PK with no id column.
            # New schema has autoincrement id PK.
            result = conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'route' AND column_name = 'id'"
                )
            )
            if result.fetchone() is None:
                # Old schema detected — drop and let create_all rebuild
                conn.execute(text("DROP TABLE IF EXISTS route CASCADE"))

            # Historic queue pipeline columns on route table
            conn.execute(
                text(
                    "ALTER TABLE route ADD COLUMN IF NOT EXISTS source_trip_id INTEGER"
                )
            )
            conn.execute(
                text("ALTER TABLE route DROP COLUMN IF EXISTS source_started_at")
            )
            conn.execute(
                text("ALTER TABLE route DROP COLUMN IF EXISTS source_ended_at")
            )
            conn.execute(text("ALTER TABLE route DROP COLUMN IF EXISTS trip_group"))
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_route_source_trip_id ON route (source_trip_id)"
                )
            )
        print("✓ Schema migration completed")
    except Exception as e:
        print(f"Warning during migration: {e}")
        pass  # Table might not exist yet; initial_load() will handle it


def initial_load(db_manager):
    """One-time initialization: create tables, load station data, and populate trip history.

    This should only be called once when setting up a fresh database.
    Creates all ORM tables, loads stations from CSV, fetches short names from GBFS,
    and downloads trip data from S3.

    Args:
        db_manager: DBManager instance
    """
    from postgres_manager import Base

    # Create all tables
    Base.metadata.create_all(db_manager.engine)
    with db_manager.engine.begin() as conn:
        conn.execute(
            text("ALTER TABLE stations ADD COLUMN IF NOT EXISTS short_name TEXT")
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_station_short_name ON stations (short_name)"
            )
        )

    # Load default and first historic metadata for current timestamp:
    with db_manager.Session_eng() as session:
        load_station_data(db_manager, session)
        update_station_short_names(db_manager, session)
        # Ensure app_metadata has a default entry (id=1)
        from postgres_manager import AppMetadata

        res = session.get(AppMetadata, 1)
        if res is None:
            res = AppMetadata(id=1, last_updated=datetime.now(), mode=0)
            session.add(res)
        session.commit()

    # Load any new trip data from the S3 bucket
    download_trip_data(db_manager)
    print("✓ Initial database load completed")


def load_station_data(db_manager, session, csv_path="data_src.csv"):
    """Load station data from CSV file into database.

    Args:
        db_manager: DBManager instance (for access to models if needed)
        session: SQLAlchemy session
        csv_path: Path to station CSV file (default: data_src.csv)
    """
    from postgres_manager import Station, CurrentData

    print(f"Loading station data from {csv_path}...")
    df = pd.read_csv(csv_path)
    loaded = 0
    for _, row in df.iterrows():
        station_id = str(row["station_id"])
        station = Station(
            index=int(row["index"]),
            station_id=station_id,
            short_name=None,
            name=row["matched_name"],
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
        )
        if not session.get(Station, row["index"]):
            session.add(station)
            station.current_data = CurrentData(station_id=station_id)
            loaded += 1
    session.commit()
    print(f"✓ Loaded {loaded} stations into DB")


def update_station_short_names(db_manager, session=None):
    """Load/update station short names from GBFS station_information feed.

    Args:
        db_manager: DBManager instance
        session: SQLAlchemy session (creates one if not provided)

    Returns:
        Number of stations updated
    """
    from postgres_manager import Station

    close_session = False
    if session is None:
        session = db_manager.Session_eng()
        close_session = True

    try:
        short_name_map = _fetch_station_information()
    except Exception as e:
        print(f"Warning: could not fetch GBFS station information for short names: {e}")
        if close_session:
            session.close()
        return 0

    if not short_name_map:
        if close_session:
            session.close()
        return 0

    updated = 0
    stations = session.query(Station).all()
    for station in stations:
        new_short_name = short_name_map.get(station.station_id)
        if new_short_name and station.short_name != new_short_name:
            station.short_name = new_short_name
            updated += 1

    if updated:
        session.commit()
        print(f"✓ Updated short_name for {updated} stations from GBFS")

    if close_session:
        session.close()

    return updated


def _fetch_station_information():
    """Fetch station information feed and return station_id -> short_name mapping.

    Returns:
        Dict mapping station_id to short_name
    """
    info_url = "https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_information.json"
    response = requests.get(info_url, timeout=20)
    response.raise_for_status()
    data = response.json()

    short_names = {}
    for station in data.get("data", {}).get("stations", []):
        station_id = str(station.get("station_id", "")).strip()
        short_name = station.get("short_name")
        if station_id and short_name is not None:
            short_name = str(short_name).strip()
            if short_name:
                short_names[station_id] = short_name
    return short_names


def download_trip_data(db_manager):
    """Download and load *new* Citibike trip-data zip files from the S3 bucket.

    Files already ingested (tracked in ``downloaded_trip_files`` table) are skipped.
    Only loads files from the last year to improve performance.

    Args:
        db_manager: DBManager instance

    Returns:
        Total number of trip rows inserted
    """
    from postgres_manager import DownloadedTripFile

    S3_BUCKET_URL = "https://s3.amazonaws.com/tripdata"
    S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    _RELEVANT_COLS = {
        # New format (Feb 2021+)
        "started_at",
        "ended_at",
        "start_station_id",
        "end_station_id",
        "rideable_type",
    }

    print("Fetching S3 bucket listing …")
    try:
        all_keys = _list_s3_zip_keys(S3_BUCKET_URL, S3_NS)
    except Exception as e:
        print(f"✗ Could not list S3 bucket: {e}")
        return 0

    print(f"Found {len(all_keys)} trip-data zip files in S3 bucket")

    # Filter to only files from the last year
    one_year_ago = datetime.now() - timedelta(days=365)
    recent_files = []
    for key in all_keys:
        # Extract year/date from filename (e.g., "2025-citibike-tripdata.zip" or "202512-citibike-tripdata.zip")
        try:
            date_part = key.split("-citibike")[0]
            if len(date_part) == 4:  # YYYY format
                file_year = int(date_part)
                file_date = datetime(file_year, 1, 1)
            elif len(date_part) == 6:  # YYYYMM format
                file_year = int(date_part[:4])
                file_month = int(date_part[4:])
                file_date = datetime(file_year, file_month, 1)
            else:
                continue
            if file_date >= one_year_ago:
                recent_files.append(key)
        except (ValueError, IndexError):
            continue

    print(f"Loading {len(recent_files)} recent file(s) from the last year")

    # Determine which files still need to be loaded and preload valid station short names
    with db_manager.Session_eng() as session:
        from postgres_manager import Station

        already_loaded = set(
            row[0] for row in session.query(DownloadedTripFile.filename).all()
        )
        valid_short_names = {row[0] for row in session.query(Station.short_name).all()}

    new_files = [k for k in recent_files if k not in already_loaded]

    if not valid_short_names:
        print("✗ No station short names found in DB; skipping trip-data load.")
        return 0

    if not new_files:
        print("All recent trip-data files already loaded.")
        return 0

    print(f"{len(new_files)} new file(s) to download and load")
    total_trips = 0

    for file_key in sorted(new_files):
        file_url = f"{S3_BUCKET_URL}/{file_key}"
        print(f"Downloading {file_key} …")
        try:
            resp = requests.get(file_url, timeout=600)
            resp.raise_for_status()

            trips_in_file = 0
            files_processed = []

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_names = [
                    n
                    for n in zf.namelist()
                    if n.lower().endswith(".csv") and not n.startswith("__MACOSX")
                ]

                for csv_name in csv_names:
                    print(f"  Processing {csv_name} …")
                    with zf.open(csv_name) as csv_file:
                        # Load CSV and only select needed columns upfront
                        reader = pd.read_csv(
                            csv_file,
                            dtype=str,
                            usecols=_RELEVANT_COLS,
                        )
                        df = _parse_trip_chunk(reader, valid_short_names)
                        if not df.empty:
                            # Use pd.to_sql for efficient bulk insertion
                            df.to_sql(
                                "trip_data",
                                db_manager.engine,
                                if_exists="append",
                                index=False,
                            )
                            trips_in_file += len(df)
                            files_processed.append(csv_name)

            # Mark this zip as ingested (separate transaction)
            with db_manager.Session_eng() as session:
                session.add(
                    DownloadedTripFile(
                        filename=file_key,
                        downloaded_at=datetime.now(),
                    )
                )
                session.commit()

            total_trips += trips_in_file
            print(f"  ✓ Loaded {trips_in_file:,} trips from {file_key}")

        except Exception as e:
            print(f"  ✗ Error processing {file_key}: {e}")
            continue

    print(f"✓ Total: loaded {total_trips:,} new trips from {len(new_files)} file(s)")
    return total_trips


def _list_s3_zip_keys(s3_bucket_url, s3_ns):
    """Fetch all .zip keys containing citibike trip data from the S3 bucket.

    Args:
        s3_bucket_url: Base S3 bucket URL
        s3_ns: XML namespace for S3 responses

    Returns:
        List of S3 key names for citibike trip data files
    """
    all_keys = []
    marker = ""
    while True:
        url = s3_bucket_url if not marker else f"{s3_bucket_url}?marker={marker}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        for contents in root.findall(f"{s3_ns}Contents"):
            key = contents.find(f"{s3_ns}Key").text
            all_keys.append(key)

        is_truncated = root.find(f"{s3_ns}IsTruncated")
        if is_truncated is not None and is_truncated.text.lower() == "true":
            marker = all_keys[-1]
        else:
            break

    return [k for k in all_keys if k.endswith(".zip") and "citibike-tripdata" in k]


def _parse_trip_chunk(chunk, valid_short_names):
    """Normalize a CSV chunk (old *or* new Citibike format) for trip_data table.

    Returns a dataframe with columns: started_at, ended_at, start_station_id,
    end_station_id, rideable_type (ready for pd.to_sql insertion).

    Args:
        chunk: DataFrame loaded from CSV
        valid_short_names: Set of valid station short names to filter by

    Returns:
        Normalized DataFrame ready for database insertion
    """
    # Convert datetime columns
    chunk["started_at"] = pd.to_datetime(chunk["started_at"], errors="coerce")
    chunk["ended_at"] = pd.to_datetime(chunk["ended_at"], errors="coerce")

    # Drop rows with any missing values in any column
    chunk = chunk.dropna()

    # Keep only rows whose start/end short names are present in stations table
    station_mask = chunk["start_station_id"].isin(valid_short_names) & chunk[
        "end_station_id"
    ].isin(valid_short_names)
    chunk = chunk[station_mask]

    return chunk
