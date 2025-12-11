#!/usr/bin/env python3
"""
PostgreSQL Manager for BikeTrack using SQLAlchemy ORM
Loads station data from data_src.csv and periodically updates with GBFS status
"""

import pandas as pd
import requests
import time
import sys
from datetime import datetime, timedelta
from sqlalchemy import create_engine, ForeignKey, Index, select
from sqlalchemy import func
from sqlalchemy.orm import (
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
    name: Mapped[str]
    latitude: Mapped[float]
    longitude: Mapped[float]
    __table_args__ = (Index("idx_station_point", "station_id"),)

    # Relationships
    current_data: Mapped[Optional["CurrentData"]] = relationship(
        "CurrentData",
        back_populates="station",
        uselist=False,
        cascade="all, delete-orphan",
    )
    historic_data: Mapped[List["HistoricData"]] = relationship(
        "HistoricData", back_populates="station", cascade="all, delete-orphan"
    )
    route: Mapped[Optional["Route"]] = relationship(
        "Route", back_populates="station", uselist=False, cascade="all, delete-orphan"
    )

    def to_dict(self):
        """Convert to dictionary"""
        result = {
            "index": self.index,
            "station_id": self.station_id,
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


class HistoricData(Base):
    """Historical data snapshots every 15 minutes"""

    __tablename__ = "historic_data"

    id: Mapped[int] = mapped_column(primary_key=True)
    station_id: Mapped[str] = mapped_column(
        ForeignKey("stations.station_id"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(nullable=False)
    bikes_available: Mapped[int]
    docks_available: Mapped[int]
    ebikes_available: Mapped[int]
    # Relationship
    station: Mapped["Station"] = relationship("Station", back_populates="historic_data")

    # Index for efficient queries
    __table_args__ = (Index("idx_historic_timestamp", "station_id", "timestamp"),)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "station_id": self.station_id,
            "timestamp": self.timestamp,
            "bikes_available": self.bikes_available,
            "docks_available": self.docks_available,
            "ebikes_available": self.ebikes_available,
        }


class Route(Base):
    """Current route display data"""

    __tablename__ = "route"

    station_id: Mapped[str] = mapped_column(
        ForeignKey("stations.station_id"), primary_key=True
    )
    color: Mapped[str] = mapped_column(nullable=False)

    # Relationship
    station: Mapped["Station"] = relationship("Station", back_populates="route")


class AppMetadata(Base):
    """Application metadata to speed up interactions"""

    __tablename__ = "app_metadata"

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    last_updated: Mapped[datetime] = mapped_column()
    viewing_timestamp: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    speed: Mapped[int] = mapped_column(default=1)
    mode: Mapped[int] = mapped_column(default=0)


class DBManager:
    def __init__(self, host, port, dbname, user, password):

        self.database_url = URL.create(
            "postgresql", username=user, password=password, host=host, port=port, database=dbname
        )
        self.engine = create_engine(self.database_url, echo=False)
        self.Session_eng = sessionmaker(bind=self.engine)
        print(f"Connected to PostgreSQL at {host}:{port}/{dbname}")

    def get_metadata(self, session=None):
        """Retrieve application metadata"""
        with self.Session_eng() as session:
            meta = session.get(AppMetadata, 1)
            return meta

    def update_metadata(
        self, session=None, in_type=None, viewing_timestamp=None, speed=None
    ):
        """
        Update the metadata table with mode, last_updated, viewing_timestamp, and speed.
        Args:
            session: SQLAlchemy session (optional)
            type: Mode label (optional)
            viewing_timestamp: Timestamp being viewed (optional)
            speed: Speed value (optional)
        """
        if session is None:
            with self.Session_eng() as session:
                meta = session.get(AppMetadata, 1)
                meta.last_updated = datetime.now()
                print("Updating metadata:", {
                    "mode": in_type,
                    "viewing_timestamp": viewing_timestamp,
                    "speed": speed
                })
                meta.mode = in_type if in_type else meta.mode
                if viewing_timestamp is not None:
                    meta.viewing_timestamp = viewing_timestamp
                if speed is not None:
                    meta.speed = speed
                session.commit()

                return meta
        meta = session.get(AppMetadata, 1)
        meta.last_updated = datetime.now()
        meta.mode = in_type if in_type else meta.mode
        if viewing_timestamp is not None:
            meta.viewing_timestamp = viewing_timestamp
        if speed is not None:
            meta.speed = speed
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
        # Load default and first historic metadata for current timestamp:
        with self.Session_eng() as session:
            self.load_station_data(session)
            # Ensure app_metadata has a default entry (id=1)
            res = session.get(AppMetadata, 1)
            if res is None:
                res = AppMetadata(id=1, last_updated=datetime.now(), mode=0)
                session.add(res)
            session.commit()

    def load_station_data(self, session, csv_path="data_src.csv"):
        print(f"Loading station data from {csv_path}...")
        df = pd.read_csv(csv_path)
        loaded = 0
        for _, row in df.iterrows():
            station_id = str(row["station_id"])
            station = Station(
                index=int(row["index"]),
                station_id=station_id,
                name=row["matched_name"],
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
            )
            if not session.get(Station, row["index"]):
                session.add(station)
                station.current_data = CurrentData(station_id=station_id)
                loaded += 1
        print(f"✓ Loaded {loaded} stations into DB")

    def update_stations(self, url, archive=False):
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
                # Archive historic data if this is a relevant timestamp
                if archive:
                    historic = HistoricData(
                        station_id=station_id,
                        timestamp=timestamp,
                        bikes_available=current.bikes_available,
                        docks_available=current.docks_available,
                        ebikes_available=current.ebikes_available,
                    )
                    session.add(historic)

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
        with self.Session_eng() as session:
            session.query(Route).delete()
            # Add new routes
            for station_data in route_stations:
                route = Route(
                    station_id=station_data["station_id"],
                    color=station_data.get("color", "#ffffff"),
                )
                session.add(route)
            self.update_metadata(session, type=ROUTE)
            session.commit()
        print(f"✓ Stored {len(route_stations)} route stations in PostgreSQL")

    def get_route_stations(self):
        """Get all route stations and their colors"""
        with self.Session_eng() as session:
            routes = (
                session.query(Route)
                .join(Station)
                .order_by(
                    Station.latitude.asc()
                )  # south (low latitude) first, north last
                .all()
            )
            # Dict preserves insertion order; latitude ascending yields south→north
            # Return mapping of station_id -> color (hex string)
            return {route.station_id: route.color for route in routes}

    def clear_route(self):
        """Clear route stations"""
        with self.Session_eng() as session:
            session.query(Route).delete()
            self.update_metadata(session, type=LIVE)
            session.commit()

        print("✓ Cleared route stations")

    def get_closest_artifact(self, timestamp):
        """Get all station snapshots for the timestamp closest to the given timestamp"""
        with self.Session_eng() as session:
            # Get all distinct timestamps from the database
            all_timestamps = session.query(HistoricData.timestamp).distinct().all()            
            if not all_timestamps:
                return []
            
            # Find the closest timestamp using Python
            closest_timestamp = min(
                all_timestamps, 
                key=lambda ts: abs((ts - timestamp).total_seconds())
            )
            
            print((f"Closest timestamp found: {closest_timestamp}"))
            # Fetch all historic snapshots for that timestamp
            snapshots = (
                session.query(HistoricData)
                .filter(HistoricData.timestamp == closest_timestamp)
                .all()
            )

            return [x.to_dict() for x in snapshots]

    def get_timestamp_range(self):
        """Get the min and max timestamps from historic_data"""
        with self.Session_eng() as session:
            result = session.query(
                func.min(HistoricData.timestamp), func.max(HistoricData.timestamp)
            ).first()
            if result and len(result) == 2:
                return {"min": result[0], "max": result[1]}
            return None


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

            # Collect all stations to write
            stations_to_write = []

            for idx, stat in enumerate(route_stats):
                # Only include stations from selected route, or all routes if none selected
                should_include = (selected_route is None) or (selected_route == idx)

                if (
                    should_include
                    and "route_stations" in stat
                    and not stat["route_stations"].empty
                ):
                    color = stat["color"]

                    for _, srow in stat["route_stations"].iterrows():
                        stations_to_write.append(
                            {"station_id": str(srow["station_id"]), "color": color}
                        )

            # Write to database
            if stations_to_write:
                for station_data in stations_to_write:
                    route = Route(
                        station_id=station_data["station_id"],
                        color=station_data["color"],
                    )
                    session.add(route)

                self.update_metadata(session, type=ROUTE)
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


if __name__ == "__main__":
    # GBFS URL
    #G Update interval in seconds
    manager = DBManager(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)
    manager.initial_load()
    last_archive = datetime.now()
    station_status_url = (
        "https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_status.json"
    )
    while True:
        time_diff = (datetime.now() - last_archive).total_seconds() / 60.0
        archive = time_diff >= HISTORY_PERIOD
        if archive:
            last_archive = datetime.now()
            print(f"Archiving historic data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
        try:
            num_updated = manager.update_stations(
                url=station_status_url, archive=archive
            )
            print(f'Updated {num_updated} stations at {datetime.now().strftime("%H:%M:%S")}')
            time.sleep(UPDATE_INTERVAL)
        except Exception as e:
            print(f"Error updating stations... Waiting to see if issue resolves: {e}")
            time.sleep(30)