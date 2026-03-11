#!/usr/bin/env python3
"""
PostgreSQL Manager for BikeTrack using SQLAlchemy ORM
Loads station data from data_src.csv and periodically updates with GBFS status
"""

import pandas as pd
import requests
import time
import sys
import io
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from sqlalchemy import create_engine, ForeignKey, Index, select, text
from sqlalchemy import func
from sqlalchemy.orm import (
    aliased,
    sessionmaker,
    relationship,
    DeclarativeBase,
    Mapped,
    mapped_column,
    joinedload,
)
from typing import List, Optional
from sqlalchemy import URL
from sqlalchemy.orm import sessionmaker, relationship, DeclarativeBase
from globals import *
import math
import random


def haversine(lat1, lon1, lat2, lon2):
    """Returns distance in meters between two lat/lon points"""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def point_to_segment_distance(lat, lon, lat1, lon1, lat2, lon2):
    """Project point onto segment, return minimum distance (in meters)"""
    x0, y0 = lon, lat
    x1, y1 = lon1, lat1
    x2, y2 = lon2, lat2
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return haversine(lat, lon, lat1, lon1)
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx*dx + dy*dy)
    t = max(0, min(1, t))
    proj_lon = x1 + t * dx
    proj_lat = y1 + t * dy
    return haversine(lat, lon, proj_lat, proj_lon)


# SQLAlchemy ORM Base
class Base(DeclarativeBase):
    pass


class Station(Base):
    """Core station information"""

    __tablename__ = "stations"
    index: Mapped[int] = mapped_column(primary_key=True)
    station_id: Mapped[str] = mapped_column(unique=True)
    short_name: Mapped[Optional[str]] = mapped_column(nullable=True)
    name: Mapped[str]
    latitude: Mapped[float]
    longitude: Mapped[float]
    __table_args__ = (
        Index("idx_station_point", "station_id"),
        Index("idx_station_short_name", "short_name"),
    )

    # Relationships
    current_data: Mapped[Optional["CurrentData"]] = relationship(
        "CurrentData",
        back_populates="station",
        uselist=False,
        cascade="all, delete-orphan",
    )
    routes: Mapped[List["Route"]] = relationship(
        "Route", back_populates="station", cascade="all, delete-orphan"
    )

    def to_dict(self):
        """Convert to dictionary"""
        result = {
            "index": self.index,
            "station_id": self.station_id,
            "short_name": self.short_name,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }
        if self.current_data:
            result.update(
                {
                    "bikes_available": self.current_data.bikes_available,
                    "docks_available": self.current_data.docks_available,
                    "ebikes_available": self.current_data.ebikes_available,
                    "last_updated": self.current_data.last_updated,
                    "update_timestamp": self.current_data.last_updated.timestamp(),  # For driver.py compatibility
                    # Note: prev_* fields not tracked in current schema - driver.py LED blinking may not work as expected
                    "prev_bikes_available": 0,  # Driver.py compatibility - would need schema update for real tracking
                    "prev_ebikes_available": 0,  # Driver.py compatibility - would need schema update for real tracking
                }
            )
        return result


class CurrentData(Base):
    """Real-time station status data"""

    __tablename__ = "current_data"
    station_id: Mapped[str] = mapped_column(
        ForeignKey("stations.station_id"), primary_key=True
    )
    bikes_available: Mapped[int] = mapped_column(nullable=True, default=0)
    docks_available: Mapped[int] = mapped_column(nullable=True, default=0)
    ebikes_available: Mapped[int] = mapped_column(nullable=True, default=0)
    last_updated: Mapped[datetime] = mapped_column(nullable=False,default=datetime.now)
    # Relationship
    station: Mapped["Station"] = relationship("Station", back_populates="current_data")

class TripData(Base):
    """Historical trip data from Citibike S3 bucket"""
    __tablename__ = "trip_data"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(nullable=False)
    ended_at: Mapped[datetime] = mapped_column(nullable=False)
    start_station_id: Mapped[Optional[str]] = mapped_column(nullable=True)
    end_station_id: Mapped[Optional[str]] = mapped_column(nullable=True)
    rideable_type: Mapped[Optional[str]] = mapped_column(nullable=True)
    __table_args__ = (
        Index("idx_trip_started", "started_at"),
        Index("idx_trip_start_station", "start_station_id"),
        Index("idx_trip_end_station", "end_station_id"),
    )


class DownloadedTripFile(Base):
    """Tracks which S3 trip data files have already been downloaded and loaded"""
    __tablename__ = "downloaded_trip_files"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(unique=True, nullable=False)
    downloaded_at: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)


class Route(Base):
    """Route display data for both route-finder and historic playback.

    For route-finder mode: trip_group=0, appear_at=0.0, lifetime=-1.0 (permanent).
    For historic mode: each trip gets a unique trip_group, with appear_at
    (seconds from playback start) and lifetime (seconds visible).
    """

    __tablename__ = "route"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    station_id: Mapped[str] = mapped_column(
        ForeignKey("stations.station_id"), nullable=False
    )
    color: Mapped[str] = mapped_column(nullable=False)
    trip_group: Mapped[int] = mapped_column(default=0)
    appear_at: Mapped[float] = mapped_column(default=0.0)
    lifetime: Mapped[float] = mapped_column(default=-1.0)

    # Relationship
    station: Mapped["Station"] = relationship("Station", back_populates="routes")


class HistoricTrip(Base):
    """Queued trip data for historic playback on the LED driver.

    The web app writes all trips here; the driver reads them in order,
    manages which 10 are active, and cycles through them.
    """

    __tablename__ = "historic_trips"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    position: Mapped[int] = mapped_column(nullable=False)  # 0-based queue order
    color: Mapped[str] = mapped_column(nullable=False)
    duration_seconds: Mapped[float] = mapped_column(nullable=False)
    start_station_id: Mapped[str] = mapped_column(nullable=False)
    end_station_id: Mapped[str] = mapped_column(nullable=False)
    start_station_name: Mapped[Optional[str]] = mapped_column(nullable=True)
    end_station_name: Mapped[Optional[str]] = mapped_column(nullable=True)
    rideable_type: Mapped[Optional[str]] = mapped_column(nullable=True)


class AppMetadata(Base):
    """Application metadata to speed up interactions"""

    __tablename__ = "app_metadata"

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    last_updated: Mapped[datetime] = mapped_column()
    viewing_timestamp: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    speed: Mapped[int] = mapped_column(default=1)
    mode: Mapped[int] = mapped_column(default=0)
    num_trips: Mapped[int] = mapped_column(default=10)


class DBManager:
    def __init__(self, host, port, dbname, user, password):

        self.database_url = URL.create(
            "postgresql", username=user, password=password, host=host, port=port, database=dbname
        )
        self.engine = create_engine(self.database_url, echo=False)
        self.Session_eng = sessionmaker(bind=self.engine)
        self._migrate_schema()
        print(f"Connected to PostgreSQL at {host}:{port}/{dbname}")

    def _migrate_schema(self):
        """Lightweight schema migration for columns added after initial deployment."""
        try:
            with self.engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE app_metadata ADD COLUMN IF NOT EXISTS num_trips INTEGER DEFAULT 10"
                ))
                # Migrate route table: old schema had station_id as PK with no id column.
                # New schema has autoincrement id PK + trip_group/appear_at/lifetime.
                result = conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'route' AND column_name = 'trip_group'"
                ))
                if result.fetchone() is None:
                    # Old schema detected — drop and let create_all rebuild
                    conn.execute(text("DROP TABLE IF EXISTS route CASCADE"))
        except Exception:
            pass  # Table might not exist yet; initial_load() will handle it

    def get_metadata(self, session=None):
        """Retrieve application metadata"""
        with self.Session_eng() as session:
            meta = session.get(AppMetadata, 1)
            return meta

    def update_metadata(
        self, session=None, in_type=None, speed=None, viewing_timestamp=None, num_trips=None
    ):
        """
        Update the metadata table with mode, last_updated, viewing_timestamp, speed, and num_trips.
        Args:
            session: SQLAlchemy session (optional)
            in_type: Mode label (optional)
            viewing_timestamp: Timestamp being viewed (optional)
            speed: Speed value (optional)
            num_trips: Number of historic trips (optional)
        """
        if session is None:
            with self.Session_eng() as session:
                meta = session.get(AppMetadata, 1)
                meta.last_updated = datetime.now()
                meta.mode = in_type if in_type else meta.mode
                if speed is not None:
                    meta.speed = speed
                if viewing_timestamp is not None:
                    meta.viewing_timestamp = viewing_timestamp
                if num_trips is not None:
                    meta.num_trips = num_trips
                session.commit()
                return meta
        meta = session.get(AppMetadata, 1)
        meta.last_updated = datetime.now()
        meta.mode = in_type if in_type else meta.mode
        if speed is not None:
            meta.speed = speed
        if viewing_timestamp is not None:
            meta.viewing_timestamp = viewing_timestamp
        if num_trips is not None:
            meta.num_trips = num_trips
        return meta

    def is_update(self, prev_timestamp):
        """Check if there are any station updates since the given timestamp"""
        with self.Session_eng() as session:
            meta = session.get(AppMetadata, 1)
            if meta.last_updated > prev_timestamp:
                return False, meta.mode
            return True, meta.mode

    def initial_load(self):
        """Create all tables if they don't exist"""
        Base.metadata.create_all(self.engine)
        with self.engine.begin() as conn:
            conn.execute(text("ALTER TABLE stations ADD COLUMN IF NOT EXISTS short_name TEXT"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_station_short_name ON stations (short_name)"))
        # Load default and first historic metadata for current timestamp:
        with self.Session_eng() as session:
            self.load_station_data(session)
            self.update_station_short_names(session)
            # Ensure app_metadata has a default entry (id=1)
            res = session.get(AppMetadata, 1)
            if res is None:
                res = AppMetadata(id=1, last_updated=datetime.now(), mode=0)
                session.add(res)
            session.commit()
        # Load any new trip data from the S3 bucket
        self.get_tripdata()


    def load_station_data(self, session, csv_path="data_src.csv"):
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
        print(f"✓ Loaded {loaded} stations into DB")

    def _fetch_station_information(self):
        """Fetch station information feed and return station_id -> short_name mapping."""
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

    def update_station_short_names(self, session):
        """Load/update station short names from GBFS station_information feed."""
        try:
            short_name_map = self._fetch_station_information()
        except Exception as e:
            print(f"Warning: could not fetch GBFS station information for short names: {e}")
            return 0

        if not short_name_map:
            return 0

        updated = 0
        stations = session.query(Station).all()
        for station in stations:
            new_short_name = short_name_map.get(station.station_id)
            if new_short_name and station.short_name != new_short_name:
                station.short_name = new_short_name
                updated += 1

        if updated:
            print(f"✓ Updated short_name for {updated} stations from GBFS")
        return updated

    def update_stations(self, url):
        """Fetch GBFS data and update station statuses"""
        updated = 0
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching station status: {e}")
        timestamp = datetime.now()
        with self.Session_eng() as session:
            for station_data in data["data"]["stations"]:
                station_id = station_data["station_id"]
                # Get station and update its current data
                station = session.query(Station).filter(Station.station_id == station_id).first()
                if not station or not station.current_data:
                    continue
                current = station.current_data
                current.bikes_available = station_data["num_bikes_available"]
                current.docks_available = station_data["num_docks_available"]
                current.ebikes_available = station_data["num_ebikes_available"]
                current.last_updated = timestamp
                updated += 1

            self.update_metadata(session)
            session.commit()
            return updated

    def get_station(self, station_id):
        """Get a single station with current data"""
        with self.Session_eng() as session:
            station = session.get(Station, station_id)
            return station.to_dict()

    def get_all_stations(self):
        """Get all stations with current data"""
        with self.Session_eng() as session:
            stations = session.query(Station).all()
            return [station.to_dict() for station in stations]

    def set_route_stations(self, route_stations):
        """Write route-finder stations (permanent, lifetime=-1)."""
        with self.Session_eng() as session:
            session.query(Route).delete()
            for station_data in route_stations:
                route = Route(
                    station_id=station_data["station_id"],
                    color=station_data.get("color", "#ffffff"),
                    trip_group=0,
                    appear_at=0.0,
                    lifetime=-1.0,
                )
                session.add(route)
            self.update_metadata(session, in_type=ROUTE)
            session.commit()
        print(f"✓ Stored {len(route_stations)} route stations in PostgreSQL")

    def get_route_stations(self):
        """Get permanent route stations (lifetime=-1) and their colors.
        Used by driver route_mode()."""
        with self.Session_eng() as session:
            routes = (
                session.query(Route)
                .filter(Route.lifetime < 0)
                .join(Station)
                .order_by(Station.latitude.asc())
                .all()
            )
            return {route.station_id: route.color for route in routes}

    def get_historic_route_data(self):
        """Get all historic route entries grouped by trip_group.
        Returns list of dicts with station_id, color, trip_group, appear_at, lifetime."""
        with self.Session_eng() as session:
            routes = (
                session.query(Route)
                .filter(Route.lifetime >= 0)
                .order_by(Route.trip_group, Route.id)
                .all()
            )
            result = []
            for r in routes:
                station = session.query(Station).filter(Station.station_id == r.station_id).first()
                if station is None:
                    continue
                stn = station.to_dict()
                result.append({
                    "station_id": r.station_id,
                    "index": stn["index"],
                    "color": r.color,
                    "trip_group": r.trip_group,
                    "appear_at": r.appear_at,
                    "lifetime": r.lifetime,
                })
            return result

    # 50 distinct colors for historic trip rendering
    HISTORIC_COLORS = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
        "#FF8000", "#FF0080", "#80FF00", "#00FF80", "#0080FF", "#8000FF",
        "#FF4000", "#FF0040", "#40FF00", "#00FF40", "#0040FF", "#4000FF",
        "#FFC000", "#FF00C0", "#C0FF00", "#00FFC0", "#00C0FF", "#C000FF",
        "#FF6600", "#FF0066", "#66FF00", "#00FF66", "#0066FF", "#6600FF",
        "#FFAA00", "#FF00AA", "#AAFF00", "#00FFAA", "#00AAFF", "#AA00FF",
        "#FF3333", "#33FF33", "#3333FF", "#FFFF33", "#FF33FF", "#33FFFF",
        "#FF6633", "#33FF66", "#6633FF", "#FFCC33", "#CC33FF", "#33FFCC",
        "#FF9933", "#9933FF",
    ]
    HISTORIC_STATION_THRESHOLD = 200  # meters

    def prepare_historic_routes(self):
        """Read HISTORIC metadata, select trips, compute stations on each
        trip's path, assign colors and timing, and write everything to the
        route table so the driver can render it.

        The timing model:
        - All initial num_trips trips appear at t=0 with lifetime = duration/speed.
        - When a trip slot expires, the next replacement appears at
          appear_at = previous_appear_at + previous_lifetime.
        Each slot independently chains replacements ~5 deep.
        """
        meta = self.get_metadata()
        start_dt = meta.viewing_timestamp
        speed = meta.speed or 100
        num_trips = meta.num_trips or 10

        if not start_dt:
            print("prepare_historic_routes: no viewing_timestamp, skipping")
            return False

        print(f"Preparing historic routes: start={start_dt}, speed={speed}x, trips={num_trips}")

        # Build station lookup
        all_stations = self.get_all_stations()
        station_map = {s["station_id"]: s for s in all_stations}

        # Color management
        palette = list(self.HISTORIC_COLORS)
        random.shuffle(palette)
        used_colors = set()

        def pick_color():
            for c in palette:
                if c not in used_colors:
                    used_colors.add(c)
                    return c
            return "#FFFFFF"

        def release_color(c):
            used_colors.discard(c)

        def get_path_station_ids(trip):
            """Get station IDs along the straight-line path of a trip, sorted start→end."""
            path = [
                [trip["start_lat"], trip["start_lon"]],
                [trip["end_lat"], trip["end_lon"]],
            ]
            stations = self.get_stations_on_path(path, self.HISTORIC_STATION_THRESHOLD)
            if not stations:
                return []
            slat, slon = trip["start_lat"], trip["start_lon"]
            elat, elon = trip["end_lat"], trip["end_lon"]
            dx, dy = elon - slon, elat - slat
            if dx != 0 or dy != 0:
                stations.sort(key=lambda s: (s["longitude"] - slon) * dx + (s["latitude"] - slat) * dy)
            return [s["station_id"] for s in stations]

        # Select initial batch
        trips = self.get_random_trips_in_window(start_dt, window_minutes=60, limit=num_trips)
        if not trips:
            print("prepare_historic_routes: no trips found, switching to LIVE")
            self.update_metadata(in_type=LIVE)
            return False

        REPLACEMENTS_PER_SLOT = 5
        route_entries = []
        trip_group_counter = 0

        for trip in trips:
            station_ids = get_path_station_ids(trip)
            if not station_ids:
                continue

            color = pick_color()
            lifetime = max(trip["duration_seconds"] / speed, 3.0)
            appear_at = 0.0

            for sid in station_ids:
                route_entries.append(Route(
                    station_id=sid, color=color,
                    trip_group=trip_group_counter,
                    appear_at=appear_at, lifetime=lifetime,
                ))

            # Chain replacements for this slot
            current_end_dt = trip["ended_at"]
            current_appear = appear_at + lifetime
            release_color(color)

            for _ in range(REPLACEMENTS_PER_SLOT):
                rep_trips = self.get_random_trips_in_window(
                    current_end_dt, window_minutes=60, limit=1
                )
                if not rep_trips:
                    break
                rep = rep_trips[0]
                rep_sids = get_path_station_ids(rep)
                if not rep_sids:
                    break
                rep_color = pick_color()
                rep_lifetime = max(rep["duration_seconds"] / speed, 3.0)

                trip_group_counter += 1
                for sid in rep_sids:
                    route_entries.append(Route(
                        station_id=sid, color=rep_color,
                        trip_group=trip_group_counter,
                        appear_at=current_appear, lifetime=rep_lifetime,
                    ))

                current_end_dt = rep["ended_at"]
                current_appear += rep_lifetime
                release_color(rep_color)

            trip_group_counter += 1

        if not route_entries:
            print("prepare_historic_routes: no valid routes, switching to LIVE")
            self.update_metadata(in_type=LIVE)
            return False

        # Write atomically
        with self.Session_eng() as session:
            session.query(Route).delete()
            session.add_all(route_entries)
            session.commit()

        print(f"✓ Prepared {len(route_entries)} historic route entries "
              f"across {trip_group_counter} trip groups")
        return True

    def clear_route(self, set_live=True):
        """Clear route stations"""
        with self.Session_eng() as session:
            session.query(Route).delete()
            if set_live:
                self.update_metadata(session, in_type=LIVE)
            session.commit()

        print("✓ Cleared route stations")



    def get_stations_by_distance(
        self, latitude, longitude, limit=None, filter_type=None
    ) -> List[dict]:

        with self.Session_eng() as session:
            # Calculate Euclidean distance in lat/lon space
            # For more accurate geographic distance, use sqrt((lat1-lat2)^2 + (lon1-lon2)^2)
            # Note: This is approximate; for precise distances, use PostGIS ST_Distance
            distance_expr = func.sqrt(
                func.pow(Station.latitude - latitude, 2)
                + func.pow(Station.longitude - longitude, 2)
            ).label("distance")

            query = session.query(Station, distance_expr)

            ##Get the closest one with bikes or docks if specified

            if filter_type == "bikes":
                query = query.join(CurrentData).where(CurrentData.bikes_available > 0)
            elif filter_type == "docks":
                query = query.join(CurrentData).where(CurrentData.docks_available > 0)
            elif filter_type == "ebikes":
                query = query.join(CurrentData).where(CurrentData.ebikes_available > 0)
            query = query.order_by(distance_expr)

            if limit:
                query = query.limit(limit)

            results = query.all()
            return_list = [station.to_dict() for station, _ in results]
            return return_list

    def get_stations_on_path(self, path, threshold_meters):
        """
        Return all stations within threshold_meters of any segment in the path.
        path: list of (lat, lon) tuples
        threshold_meters: distance threshold
        """
        # Compute bounding box for initial filter
        lats = [pt[0] for pt in path]
        lons = [pt[1] for pt in path]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Calculate lat/lon margin from threshold_meters using haversine
        # Approximate: 1 degree latitude ≈ 111km, adjust longitude by latitude
        R = 6371000  # Earth radius in meters
        lat_margin = (threshold_meters / R) * (180 / math.pi)
        
        # Longitude margin varies with latitude - use average latitude for approximation
        avg_lat = (min_lat + max_lat) / 2
        lon_margin = (threshold_meters / (R * math.cos(math.radians(avg_lat)))) * (180 / math.pi)
        
        min_lat -= lat_margin
        max_lat += lat_margin
        min_lon -= lon_margin
        max_lon += lon_margin

        with self.Session_eng() as session:
            # Initial bounding box filter
            stations = session.query(Station).filter(
                Station.latitude >= min_lat,
                Station.latitude <= max_lat,
                Station.longitude >= min_lon,
                Station.longitude <= max_lon
            ).all()
            result = []
            for station in stations:
                station_lat = station.latitude
                station_lon = station.longitude
                # Check all path segments: once we suceed, break
                # Prevents doublechecking a station that's already met the threshold
                for i in range(len(path) - 1):
                    lat1, lon1 = path[i]
                    lat2, lon2 = path[i+1]
                    dist = point_to_segment_distance(station_lat, station_lon, lat1, lon1, lat2, lon2)
                    if dist < threshold_meters:
                        result.append(station.to_dict())
                        break
            return result

    def get_all_station_status(self):
        """Get status dict for all stations (for app.py compatibility)"""
        with self.Session_eng() as session:
            stations = (
                session.query(Station).options(joinedload(Station.current_data)).all()
            )
            status_dict = {}
            for station in stations:
                if station.current_data:
                    status_dict[station.station_id] = {
                        "bikes_available": station.current_data.bikes_available,
                        "docks_available": station.current_data.docks_available,
                        "ebikes_available": station.current_data.ebikes_available,
                        "is_renting": True,  # Assume always renting for now
                        "is_returning": True,  # Assume always returning for now
                    }
            return status_dict

    def save_route(self, route_stats, selected_route=None):
        """Write route stations and their colors to PostgreSQL

        Args:
            route_stats: List of route statistics with station data
            selected_route: Index of selected route (None for all routes)
        """
        with self.Session_eng() as session:
            # Clear existing route data
            session.query(Route).delete()

            # Collect all stations to write using vectorized approach
            stations_to_write = []
            for idx, stat in enumerate(route_stats):
                should_include = (selected_route is None) or (selected_route == idx)
                if (
                    should_include
                    and "route_stations" in stat
                    and not stat["route_stations"].empty
                ):
                    color = stat["color"]
                    df = stat["route_stations"]
                    # Vectorized conversion to avoid iterrows()
                    stations_to_write.extend(
                        {"station_id": str(sid), "color": color}
                        for sid in df["station_id"]
                    )

            # Bulk insert using bulk_insert_mappings for better performance
            if stations_to_write:
                session.bulk_insert_mappings(Route, stations_to_write)
                self.update_metadata(session, in_type=ROUTE)
                session.commit()
                print(f"✓ Wrote {len(stations_to_write)} route stations to PostgreSQL")
                return len(stations_to_write)
            return 0

    def ping(self):
        """Check if PostgreSQL connection is alive"""
        try:
            with self.Session_eng() as session:
                session.execute(select(1))
            return True
        except Exception:
            return False

    def get_route_update_timestamp(self):
        """Get the timestamp of the last route update
        Returns the last_updated timestamp when mode is ROUTE
        """
        with self.Session_eng() as session:
            meta = session.get(AppMetadata, 1)
            if meta and meta.mode == ROUTE:
                return meta.last_updated.timestamp()
            return 0.0

    # ---- Historic trip queue for driver playback ----

    def set_historic_trips(self, trips):
        """Write the full trip queue for the driver to consume.

        Args:
            trips: list of dicts, each with keys:
                position, color, duration_seconds,
                start_station_id, end_station_id,
                start_station_name, end_station_name, rideable_type
        """
        with self.Session_eng() as session:
            session.query(HistoricTrip).delete()
            for t in trips:
                session.add(HistoricTrip(
                    position=t["position"],
                    color=t["color"],
                    duration_seconds=t["duration_seconds"],
                    start_station_id=t["start_station_id"],
                    end_station_id=t["end_station_id"],
                    start_station_name=t.get("start_station_name"),
                    end_station_name=t.get("end_station_name"),
                    rideable_type=t.get("rideable_type"),
                ))
            session.commit()
        print(f"✓ Stored {len(trips)} historic trips for driver playback")

    def get_historic_trips(self):
        """Read the full trip queue ordered by position."""
        with self.Session_eng() as session:
            rows = (
                session.query(HistoricTrip)
                .order_by(HistoricTrip.position)
                .all()
            )
            return [
                {
                    "position": r.position,
                    "color": r.color,
                    "duration_seconds": r.duration_seconds,
                    "start_station_id": r.start_station_id,
                    "end_station_id": r.end_station_id,
                    "start_station_name": r.start_station_name,
                    "end_station_name": r.end_station_name,
                    "rideable_type": r.rideable_type,
                }
                for r in rows
            ]

    def clear_historic_trips(self):
        """Clear the historic trip queue."""
        with self.Session_eng() as session:
            session.query(HistoricTrip).delete()
            session.commit()

    def get_random_trips_in_window(self, start_time, window_minutes=2, limit=50):
        """Get random trips starting within a time window, with station coordinates.

        Args:
            start_time: datetime for the window start
            window_minutes: width of the time window in minutes
            limit: maximum number of trips to return

        Returns:
            List of dicts with trip info and start/end station coordinates.
        """
        window_end = start_time + timedelta(minutes=window_minutes)
        StartStation = aliased(Station)
        EndStation = aliased(Station)

        with self.Session_eng() as session:
            trips = (
                session.query(TripData, StartStation, EndStation)
                .join(StartStation, StartStation.short_name == TripData.start_station_id)
                .join(EndStation, EndStation.short_name == TripData.end_station_id)
                .filter(
                    TripData.started_at >= start_time,
                    TripData.started_at < window_end,
                )
                .order_by(func.random())
                .limit(limit)
                .all()
            )

            result = []
            for trip, start_stn, end_stn in trips:
                duration = (trip.ended_at - trip.started_at).total_seconds()
                if duration <= 0:
                    continue
                result.append({
                    "trip_id": trip.id,
                    "started_at": trip.started_at,
                    "ended_at": trip.ended_at,
                    "duration_seconds": duration,
                    "rideable_type": trip.rideable_type,
                    "start_station_id": start_stn.station_id,
                    "start_station_name": start_stn.name,
                    "start_lat": start_stn.latitude,
                    "start_lon": start_stn.longitude,
                    "end_station_id": end_stn.station_id,
                    "end_station_name": end_stn.name,
                    "end_lat": end_stn.latitude,
                    "end_lon": end_stn.longitude,
                })

            return result

    # ---- Trip data ingestion from S3 bucket ----

    S3_BUCKET_URL = "https://s3.amazonaws.com/tripdata"
    S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

    def _list_s3_zip_keys(self):
        """Fetch all .zip keys containing citibike trip data from the S3 bucket."""
        all_keys = []
        marker = ""
        while True:
            url = self.S3_BUCKET_URL if not marker else f"{self.S3_BUCKET_URL}?marker={marker}"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            for contents in root.findall(f"{self.S3_NS}Contents"):
                key = contents.find(f"{self.S3_NS}Key").text
                all_keys.append(key)

            is_truncated = root.find(f"{self.S3_NS}IsTruncated")
            if is_truncated is not None and is_truncated.text.lower() == "true":
                marker = all_keys[-1]
            else:
                break

        return [k for k in all_keys if k.endswith(".zip") and "citibike-tripdata" in k]

    # Columns we care about across both CSV formats
    _RELEVANT_COLS = {
        # New format (Feb 2021+)
        "started_at", "ended_at", "start_station_id", "end_station_id",
        "rideable_type",
    }

    @staticmethod
    def _parse_trip_chunk(chunk, valid_short_names):
        """Normalize a CSV chunk (old *or* new Citibike format) for trip_data table.
        Returns a dataframe with columns: started_at, ended_at, start_station_id, 
        end_station_id, rideable_type (ready for pd.to_sql insertion).
        """
        # Convert datetime columns
        chunk["started_at"] = pd.to_datetime(chunk["started_at"], errors="coerce")
        chunk["ended_at"] = pd.to_datetime(chunk["ended_at"], errors="coerce")
        
        # Drop rows with any missing values in any column
        chunk = chunk.dropna()

        # Keep only rows whose start/end short names are present in stations table
        station_mask = (
            chunk["start_station_id"].isin(valid_short_names)
            & chunk["end_station_id"].isin(valid_short_names)
        )
        chunk = chunk[station_mask]
        
        return chunk

    def get_tripdata(self):
        """Download and load *new* Citibike trip-data zip files from the S3
        bucket into the ``trip_data`` table.  Files that have already been
        ingested (tracked in ``downloaded_trip_files``) are skipped.
        
        Only loads files from the last year to improve performance.

        Returns the total number of trip rows inserted.
        """
        print("Fetching S3 bucket listing …")
        try:
            zip_keys = self._list_s3_zip_keys()
        except Exception as e:
            print(f"✗ Could not list S3 bucket: {e}")
            return 0
        print(f"Found {len(zip_keys)} trip-data zip files in S3 bucket")

        # Filter to only files from the last year
        one_year_ago = datetime.now() - timedelta(days=365)
        recent_files = []
        for key in zip_keys:
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
        with self.Session_eng() as session:
            already_loaded = set(
                row[0] for row in session.query(DownloadedTripFile.filename).all()
            )
            valid_short_names = {
                row[0] for row in session.query(Station.short_name).all()
            }
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
            file_url = f"{self.S3_BUCKET_URL}/{file_key}"
            print(f"Downloading {file_key} …")
            try:
                resp = requests.get(file_url, timeout=600)
                resp.raise_for_status()

                trips_in_file = 0
                files_processed = []

                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    csv_names = [
                        n for n in zf.namelist()
                        if n.lower().endswith(".csv")
                        and not n.startswith("__MACOSX")
                    ]

                    for csv_name in csv_names:
                        print(f"  Processing {csv_name} …")
                        with zf.open(csv_name) as csv_file:
                            # Load CSV and only select needed columns upfront
                            reader = pd.read_csv(
                                csv_file,
                                dtype=str,
                                usecols=self._RELEVANT_COLS,
                            )
                            df = self._parse_trip_chunk(reader, valid_short_names)
                            if not df.empty:
                                # Use pd.to_sql for efficient bulk insertion
                                df.to_sql(
                                    "trip_data",
                                    self.engine,
                                    if_exists="append",
                                    index=False,
                                )
                                trips_in_file += len(df)
                                files_processed.append(csv_name)

                # Mark this zip as ingested (separate transaction)
                with self.Session_eng() as session:
                    session.add(DownloadedTripFile(
                        filename=file_key,
                        downloaded_at=datetime.now(),
                    ))
                    session.commit()

                total_trips += trips_in_file
                print(f"  ✓ Loaded {trips_in_file:,} trips from {file_key}")

            except Exception as e:
                print(f"  ✗ Error processing {file_key}: {e}")
                continue

        print(f"✓ Total: loaded {total_trips:,} new trips from {len(new_files)} file(s)")
        return total_trips


if __name__ == "__main__":
    # GBFS URL
    #G Update interval in seconds
    manager = DBManager(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)
    manager.initial_load()
    last_s3 = datetime.now()
    last_historic_params = None  # Track (viewing_timestamp, speed, num_trips) to avoid re-preparing
    station_status_url = (
        "https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_status.json"
    )
    trip_data = "https://s3.amazonaws.com/tripdata"
    manager.update_metadata(in_type=LIVE)

    while True:
        try:
            # Check if historic mode was requested
            meta = manager.get_metadata()
            if meta.mode == HISTORIC:
                current_params = (meta.viewing_timestamp, meta.speed, meta.num_trips)
                if current_params != last_historic_params:
                    print("Historic mode detected — preparing routes...")
                    manager.prepare_historic_routes()
                    last_historic_params = current_params
            else:
                last_historic_params = None

            num_updated = manager.update_stations(url=station_status_url)
            print(f'Updated {num_updated} stations at {datetime.now().strftime("%H:%M:%S")}')
            time.sleep(UPDATE_INTERVAL)
        except Exception as e:
            print(f"Error updating stations... Waiting to see if issue resolves: {e}")
            time.sleep(UPDATE_INTERVAL)
        if (datetime.now() - last_s3).total_seconds() >= TRIP_UPDATE_INTERVAL:
            manager.get_tripdata()
            last_s3 = datetime.now()