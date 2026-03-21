#!/usr/bin/env python3
"""
PostgreSQL Manager for BikeTrack using SQLAlchemy ORM
Handles database connections and runtime queries for station data and routes.

For one-time setup operations (migrations, initial load, trip data ingestion),
see db_helpers.py instead.
"""
import requests
import time
from datetime import datetime, timedelta
from sqlalchemy import create_engine, ForeignKey, Index, select, func, cast
from sqlalchemy.dialects.postgresql import array
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
from geoalchemy2 import Geometry, Geography
from globals import *
import time


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
    # PostGIS geometry column for efficient spatial queries
    geom: Mapped[Optional["Geometry"]] = mapped_column(
        Geometry("POINT", srid=4326), nullable=True
    )
    __table_args__ = (
        Index("idx_station_point", "station_id"),
        Index("idx_station_short_name", "short_name"),
        Index("idx_station_geom", "geom", postgresql_using="gist"),
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
                    "update_timestamp": self.current_data.last_updated.timestamp(),
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
    last_updated: Mapped[datetime] = mapped_column(nullable=False, default=datetime.now)
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
    """Tracks Downloaded S3 trip data files"""

    __tablename__ = "downloaded_trip_files"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(unique=True, nullable=False)
    downloaded_at: Mapped[datetime] = mapped_column(
        nullable=False, default=datetime.now
    )


class Route(Base):
    """Core route rendering system
    Expresses how routes should appear, what colors, etc.
    Route display data for both route-finder and historic playback.
    """

    __tablename__ = "route"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    station_id: Mapped[str] = mapped_column(
        ForeignKey("stations.station_id"), nullable=False
    )
    color: Mapped[str] = mapped_column(nullable=False)
    appear_at: Mapped[float] = mapped_column(default=0.0)
    lifetime: Mapped[float] = mapped_column(default=-1.0)
    source_trip_id: Mapped[Optional[int]] = mapped_column(nullable=True)
    # Relationship
    station: Mapped["Station"] = relationship("Station", back_populates="routes")


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
            "postgresql",
            username=user,
            password=password,
            host=host,
            port=port,
            database=dbname,
        )
        self.engine = create_engine(self.database_url, echo=False)
        self.Session_eng = sessionmaker(bind=self.engine)
        print(f"Connected to PostgreSQL at {host}:{port}/{dbname}")

    def is_update(self, prev_timestamp):
        """Check if there are any station updates since the given timestamp"""
        with self.Session_eng() as session:
            meta = session.get(AppMetadata, 1)
            if meta.last_updated > prev_timestamp:
                return False, meta.mode
            return True, meta.mode

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
                station = (
                    session.query(Station)
                    .filter(Station.station_id == station_id)
                    .first()
                )
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

    def get_metadata(self):
        """Retrieve application metadata"""
        with self.Session_eng() as session:
            meta = session.get(AppMetadata, 1)
            return meta

    def update_metadata(
        self,
        session=None,
        in_type=None,
        speed=None,
        viewing_timestamp=None,
        num_trips=None):
        
        close = False
        if session is None:
            session = self.Session_eng()
            close = True
        meta = session.get(AppMetadata, 1)
        meta.last_updated = datetime.now()
        meta.mode = in_type if in_type else meta.mode
        if speed is not None:
            meta.speed = speed
        if viewing_timestamp is not None:
            meta.viewing_timestamp = viewing_timestamp
        if num_trips is not None:
            meta.num_trips = num_trips
        if close:
            session.commit()
            session.close()
        return meta

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
        """Write route-finder stations with sequential appear_at times for animation.
        
        Stations appear sequentially from nearest to farthest from the route start.
        Uses PostGIS distance ordering only (no Python distance fallback).
        """
        if not route_stations:
            return
        
        with self.Session_eng() as session:
            session.query(Route).delete()
            now_ts = time.time()
            animation_duration = 5.0  # 5 seconds total animation

            station_ids = [s["station_id"] for s in route_stations]
            start_station_id = station_ids[0]
            input_order = {sid: i for i, sid in enumerate(station_ids)}

            start_geom_subquery = (
                session.query(Station.geom)
                .filter(
                    Station.station_id == start_station_id,
                    Station.geom.isnot(None),
                )
                .scalar_subquery()
            )

            start_station_exists = (
                session.query(Station.station_id)
                .filter(
                    Station.station_id == start_station_id,
                    Station.geom.isnot(None),
                )
                .first()
            )

            if start_station_exists is None:
                raise ValueError(
                    f"Missing PostGIS geometry for start station {start_station_id}"
                )

            distance_rows = (
                session.query(
                    Station.station_id,
                    func.ST_Distance(
                        cast(Station.geom, Geography),
                        cast(start_geom_subquery, Geography),
                    ).label("distance_m"),
                )
                .filter(
                    Station.station_id.in_(station_ids),
                    Station.geom.isnot(None),
                )
                .all()
            )

            if len(distance_rows) < len(set(station_ids)):
                missing = sorted(
                    set(station_ids) - {station_id for station_id, _ in distance_rows}
                )
                raise ValueError(
                    f"Missing PostGIS geometry for route stations: {', '.join(missing)}"
                )

            sorted_station_ids = [
                station_id
                for station_id, _ in sorted(
                    distance_rows,
                    key=lambda t: (float(t[1]), input_order.get(t[0], 10**9)),
                )
            ]

            route_by_station_id = {
                station_data["station_id"]: station_data for station_data in route_stations
            }
            sorted_stations = [
                route_by_station_id[sid]
                for sid in sorted_station_ids
                if sid in route_by_station_id
            ]

            num_stations = len(sorted_stations)
            for idx, station_data in enumerate(sorted_stations):
                # Space out appear_at times so route grows from start to finish
                appear_at = now_ts + (idx / num_stations) * animation_duration if num_stations > 1 else now_ts
                
                route = Route(
                    station_id=station_data["station_id"],
                    color=station_data.get("color", "#ffffff"),
                    appear_at=appear_at,
                    lifetime=animation_duration,
                )
                session.add(route)
            self.update_metadata(session, in_type=ROUTE)
            session.commit()
        print(f"✓ Stored {len(route_stations)} route stations in PostgreSQL with sequential animation")

    def get_route_rows(self):
        """Fetch active route rows joined with station index for rendering."""
        with self.Session_eng() as session:
            query = session.query(Route, Station.index).join(
                Station, Station.station_id == Route.station_id
            )
            return query.order_by(Route.source_trip_id, Route.id).all()

    def get_route_groups(self):
        """Return grouped active routes for rendering and playback."""
        rows = self.get_route_rows()
        grouped = {}
        for route, index in rows:
            group_key = (
                f"trip:{route.source_trip_id}"
                if route.source_trip_id is not None
                else f"route:{route.id}"
            )
            if group_key not in grouped:
                grouped[group_key] = {
                    "group_key": group_key,
                    "maps_to_trip": route.source_trip_id is not None,
                    "trip_id": route.source_trip_id,
                    "color": route.color,
                    "appear_at": route.appear_at,
                    "lifetime": route.lifetime,
                    "indices": [],
                    "station_ids": [],
                }
            if 0 <= index < N_LEDS:
                grouped[group_key]["indices"].append(index)
            grouped[group_key]["station_ids"].append(route.station_id)
        return list(grouped.values())

    def get_route_station_map(self):
        """Return station_id -> color map from active route rows."""
        station_map = {}
        for group in self.get_route_groups():
            for station_id in group["station_ids"]:
                if station_id not in station_map:
                    station_map[station_id] = group["color"]
        return station_map

    def _get_path_station_ids(self, trip):
        """Get station IDs on a trip path sorted from start to end using PostGIS."""
        with self.Session_eng() as session:
            start_point = func.ST_SetSRID(
                func.ST_MakePoint(trip["start_lon"], trip["start_lat"]), 4326
            )
            end_point = func.ST_SetSRID(
                func.ST_MakePoint(trip["end_lon"], trip["end_lat"]), 4326
            )
            linestring_geom = func.ST_SetSRID(
                func.ST_MakeLine(array([start_point, end_point])),
                4326,
            )
            linestring_geog = cast(linestring_geom, Geography)
            locate_expr = func.ST_LineLocatePoint(linestring_geom, Station.geom)
            rows = (
                session.query(Station.station_id, locate_expr.label("path_pos"))
                .filter(
                    Station.geom.isnot(None),
                    func.ST_DWithin(
                        cast(Station.geom, Geography),
                        linestring_geog,
                        HISTORIC_STATION_THRESHOLD,
                    ),
                )
                .order_by(locate_expr)
                .all()
            )
            return [station_id for station_id, _ in rows]

    def clear_route(self, set_live=True):
        """Clear all active route rows."""
        with self.Session_eng() as session:
            session.query(Route).delete()
            if set_live:
                self.update_metadata(session, in_type=LIVE)
            session.commit()

    def load_trips(self, timestamp, N_trips):
        """Queue N historic trips around a timestamp into the route table."""
        if timestamp is None:
            return 0

        target_count = max(0, int(N_trips or 0))
        if target_count == 0:
            return 0

        meta = self.get_metadata()
        speed = (meta.speed if meta and meta.speed else 100) or 100
        now_ts = time.time()

        candidate_trips = self.get_random_trips_in_window(timestamp)

        if not candidate_trips:
            return 0

        with self.Session_eng() as session:
            existing_trip_ids = {
                row[0]
                for row in session.query(Route.source_trip_id)
                .filter(Route.lifetime >= 0, Route.source_trip_id.isnot(None))
                .all()
            }
            loaded = 0
            if not HISTORIC_COLORS:
                color_cycle = ["#FFFFFF"]
            else:
                color_cycle = HISTORIC_COLORS

            for trip in candidate_trips:
                if loaded >= target_count:
                    break
                if trip["trip_id"] in existing_trip_ids:
                    continue

                station_ids = self._get_path_station_ids(trip)
                if not station_ids:
                    continue

                # Assign highly distinct colors before repeating.
                color = color_cycle[loaded % len(color_cycle)]
                lifetime = max(trip["duration_seconds"] / speed, 3.0)

                # Space out station appearances to animate route growth from start to finish
                num_stations = len(station_ids)
                for station_idx, station_id in enumerate(station_ids):
                    # Each station appears at a different time for animation
                    # appear_at = now_ts + (station_idx / num_stations) * lifetime
                    appear_at = now_ts + (station_idx / num_stations) * lifetime if num_stations > 1 else now_ts
                    
                    session.add(
                        Route(
                            station_id=station_id,
                            color=color,
                            appear_at=appear_at,
                            lifetime=lifetime,
                            source_trip_id=trip["trip_id"],
                        )
                    )

                existing_trip_ids.add(trip["trip_id"])
                loaded += 1

            if loaded:
                self.update_metadata(
                    session,
                    in_type=HISTORIC,
                    viewing_timestamp=timestamp,
                    num_trips=(
                        meta.num_trips if meta and meta.num_trips else target_count
                    ),
                )
            session.commit()
            return loaded

    def load_trips_with_gmaps_paths(self, trips_with_paths):
        """Load historic trips using pre-calculated Google Maps paths.
        
        This finds stations along the actual Google Maps bicycling route,
        not a direct line between start and end.
        
        Args:
            trips_with_paths: List of trip dicts with:
                - trip_id, start_lat, start_lon, end_lat, end_lon, duration_seconds, ...
                - path: list of [lat, lon] coordinates from Google Maps route
        """
        if not trips_with_paths:
            return 0
        
        meta = self.get_metadata()
        speed = (meta.speed if meta and meta.speed else 100) or 100
        now_ts = time.time()
        
        with self.Session_eng() as session:
            existing_trip_ids = {
                row[0]
                for row in session.query(Route.source_trip_id)
                .filter(Route.lifetime >= 0, Route.source_trip_id.isnot(None))
                .all()
            }
            loaded = 0
            if not HISTORIC_COLORS:
                color_cycle = ["#FFFFFF"]
            else:
                color_cycle = HISTORIC_COLORS
            
            for trip in trips_with_paths:
                if trip.get("trip_id") in existing_trip_ids:
                    continue
                
                # Get stations along the Google Maps path (not direct line)
                path = trip.get("path", [])
                if not path or len(path) < 2:
                    continue
                
                # get_stations_on_path returns list of station dicts, extract the IDs
                stations_info = self.get_stations_on_path(path, HISTORIC_STATION_THRESHOLD)
                if not stations_info:
                    continue
                station_ids = [s["station_id"] for s in stations_info]
                
                # Assign highly distinct colors before repeating.
                color = color_cycle[loaded % len(color_cycle)]
                lifetime = max(trip.get("duration_seconds", 0) / speed, 3.0)
                
                # Space out station appearances to animate route growth from start to finish
                num_stations = len(station_ids)
                for station_idx, station_id in enumerate(station_ids):
                    appear_at = now_ts + (station_idx / num_stations) * lifetime if num_stations > 1 else now_ts
                    
                    session.add(
                        Route(
                            station_id=station_id,
                            color=color,
                            appear_at=appear_at,
                            lifetime=lifetime,
                            source_trip_id=trip.get("trip_id"),
                        )
                    )
                
                existing_trip_ids.add(trip.get("trip_id"))
                loaded += 1
            
            if loaded:
                self.update_metadata(
                    session,
                    in_type=HISTORIC,
                    viewing_timestamp=meta.viewing_timestamp,
                    num_trips=(
                        meta.num_trips if meta and meta.num_trips else loaded
                    ),
                )
            session.commit()
            return loaded

    def remove_trip(self, source_trip_id):
        """Remove one historic trip (identified by source_trip_id) from route."""
        if source_trip_id is None:
            return 0
        with self.Session_eng() as session:
            deleted = (
                session.query(Route)
                .filter(Route.source_trip_id == int(source_trip_id))
                .delete()
            )
            session.commit()
            return deleted

    def get_stations_by_distance(
        self, latitude, longitude, limit=None, filter_type=None
    ) -> List[dict]:
        """Get stations sorted by distance using PostGIS.

        This uses ST_Distance for accurate geographic distance calculation,
        making it much faster than the previous Euclidean approximation.
        """
        with self.Session_eng() as session:
            # Use PostGIS geography distance so result is in meters.
            point_geog = cast(
                func.ST_SetSRID(func.ST_MakePoint(longitude, latitude), 4326),
                Geography,
            )
            distance_expr = func.ST_Distance(
                cast(Station.geom, Geography),
                point_geog,
            ).label("distance_m")

            query = session.query(Station, distance_expr).filter(Station.geom.isnot(None))

            # Get the closest one with bikes or docks if specified
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
        Uses PostGIS ST_DWithin for highly optimized distance queries.

        path: list of (lat, lon) tuples
        threshold_meters: distance threshold in meters
        """
        with self.Session_eng() as session:
            if not path or len(path) < 2:
                return []

            # Build path geometry from numeric points to avoid WKT parsing issues.
            line_points = [
                func.ST_SetSRID(func.ST_MakePoint(lon, lat), 4326)
                for lat, lon in path
            ]
            linestring_geom = func.ST_SetSRID(
                func.ST_MakeLine(array(line_points)),
                4326,
            )
            linestring_geog = cast(linestring_geom, Geography)
            locate_expr = func.ST_LineLocatePoint(linestring_geom, Station.geom)

            # Use ST_DWithin for a distance buffer query
            # This is much faster than checking each station manually
            # ST_DWithin works in meters when using geography type
            stations = (
                session.query(Station)
                .where(
                    Station.geom.isnot(None),
                    func.ST_DWithin(
                        cast(Station.geom, Geography),
                        linestring_geog,
                        threshold_meters,
                    )
                )
                .order_by(locate_expr)
                .all()
            )

            result = [station.to_dict() for station in stations]
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
        """Write route stations and their colors to PostgreSQL with sequential animation.
        
        Stations appear sequentially to create a growing animation effect.

        Args:
            route_stats: List of route statistics with station data
            selected_route: Index of selected route (None for all routes)
        """
        with self.Session_eng() as session:
            # Clear existing route data
            session.query(Route).delete()

            # Collect all stations to write with sequential appear_at times
            now_ts = time.time()
            stations_to_write = []
            animation_duration = 5.0  # 5 seconds total animation
            
            for idx, stat in enumerate(route_stats):
                should_include = (selected_route is None) or (selected_route == idx)
                if (
                    should_include
                    and "route_stations" in stat
                    and not stat["route_stations"].empty
                ):
                    color = stat["color"]
                    df = stat["route_stations"]
                    num_stations = len(df)
                    
                    # Vectorized conversion with sequential appear_at times
                    for station_idx, sid in enumerate(df["station_id"]):
                        # Space out appear_at times so route grows from start to finish
                        appear_at = now_ts + (station_idx / num_stations) * animation_duration if num_stations > 1 else now_ts
                        stations_to_write.append({
                            "station_id": str(sid),
                            "color": color,
                            "appear_at": appear_at,
                            "lifetime": animation_duration,
                        })

            # Bulk insert using bulk_insert_mappings for better performance
            if stations_to_write:
                session.bulk_insert_mappings(Route, stations_to_write)
                self.update_metadata(session, in_type=ROUTE)
                session.commit()
                print(f"✓ Wrote {len(stations_to_write)} route stations to PostgreSQL with sequential animation")
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

    def get_random_trips_in_window(self, timestamp):
        """Get random trips around a timestamp using globally configured windows.

        Returns:
            List of dicts with trip info and start/end station coordinates.
        """
        StartStation = aliased(Station)
        EndStation = aliased(Station)

        meta = self.get_metadata()
        target = max(1, int((meta.num_trips if meta else 1) or 1))
        limit = min(
            HISTORIC_TRIP_MAX_QUERY_LIMIT,
            target * HISTORIC_TRIP_CANDIDATE_MULTIPLIER,
        )

        with self.Session_eng() as session:
            trips = []
            for minutes in HISTORIC_TRIP_WINDOWS_MINUTES:
                window_start = timestamp - timedelta(minutes=minutes)
                window_end = timestamp + timedelta(minutes=minutes)
                trips = (
                    session.query(TripData, StartStation, EndStation)
                    .join(
                        StartStation,
                        StartStation.short_name == TripData.start_station_id,
                    )
                    .join(EndStation, EndStation.short_name == TripData.end_station_id)
                    .filter(
                        TripData.started_at >= window_start,
                        TripData.started_at < window_end,
                    )
                    .order_by(func.random())
                    .limit(limit)
                    .all()
                )
                if trips:
                    break

            result = []
            for trip, start_stn, end_stn in trips:
                duration = (trip.ended_at - trip.started_at).total_seconds()
                if duration <= 0:
                    continue
                result.append(
                    {
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
                    }
                )

            return result

    def get_trips_by_ids(self, trip_ids):
        """Get trip details for an explicit set of trip IDs."""
        if not trip_ids:
            return []

        StartStation = aliased(Station)
        EndStation = aliased(Station)

        with self.Session_eng() as session:
            rows = (
                session.query(TripData, StartStation, EndStation)
                .join(
                    StartStation,
                    StartStation.short_name == TripData.start_station_id,
                )
                .join(EndStation, EndStation.short_name == TripData.end_station_id)
                .filter(TripData.id.in_(trip_ids))
                .all()
            )

            result = []
            for trip, start_stn, end_stn in rows:
                duration = (trip.ended_at - trip.started_at).total_seconds()
                if duration <= 0:
                    continue
                result.append(
                    {
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
                    }
                )

            return result


if __name__ == "__main__":
    # For one-time setup tasks, use db_helpers:
    # - migrate_schema() for schema migrations
    # - initial_load() for creating tables and loading initial data
    # - download_trip_data() for loading S3 trip data

    from db_helpers import migrate_schema

    manager = DBManager(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)
    # Uncomment to run migrations: migrate_schema(manager)
    # Uncomment to do initial setup: db_helpers.initial_load(manager)

    manager.update_metadata(in_type=LIVE)
    last_s3 = datetime.now()
    station_status_url = "https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_status.json"

    while True:
        try:
            num_updated = manager.update_stations(url=station_status_url)
            print(
                f'Updated {num_updated} stations at {datetime.now().strftime("%H:%M:%S")}'
            )
            time.sleep(UPDATE_INTERVAL)
        except Exception as e:
            print(f"Error updating stations... Waiting to see if issue resolves: {e}")
            time.sleep(UPDATE_INTERVAL)
