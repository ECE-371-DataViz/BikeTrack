#!/usr/bin/env python3
"""
PostgreSQL Manager for BikeTrack using SQLAlchemy ORM
Loads station data from data_src.csv and periodically updates with GBFS status
"""

import pandas as pd
import requests
import time
import sys
from datetime import datetime
from sqlalchemy import create_engine, ForeignKey, Index, select
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker, relationship, DeclarativeBase, Mapped, mapped_column, joinedload
from typing import List, Optional
from sqlalchemy import URL
from sqlalchemy.orm import sessionmaker, relationship, DeclarativeBase
from globals import *

# SQLAlchemy ORM Base
class Base(DeclarativeBase):
    pass

class Station(Base):
    """Core station information"""
    __tablename__ = 'stations'
    index: Mapped[int] = mapped_column(primary_key=True)
    station_id: Mapped[str] = mapped_column(unique=True)
    name: Mapped[str]
    latitude: Mapped[float]
    longitude: Mapped[float]
    __table_args__ = (
        Index('idx_station_point', 'station_id'),
    )

    # Relationships
    current_data: Mapped[Optional['CurrentData']] = relationship(
        'CurrentData', back_populates="station", uselist=False, cascade="all, delete-orphan")
    historic_data: Mapped[List['HistoricData']] = relationship(
        "HistoricData", back_populates="station", cascade="all, delete-orphan")
    route: Mapped[Optional['Route']] = relationship("Route", back_populates="station",
                         uselist=False, cascade="all, delete-orphan")

    def to_dict(self):
        """Convert to dictionary"""
        result = {
            'index': self.index,
            'station_id': self.station_id,
            'name': self.name,
            'latitude': self.latitude,
            'longitude': self.longitude,
        }
        if self.current_data:
            result.update({
                'bikes_available': self.current_data.bikes_available,
                'docks_available': self.current_data.docks_available,
                'ebikes_available': self.current_data.ebikes_available,
                'last_updated': self.current_data.last_updated,
                'update_timestamp': self.current_data.last_updated.timestamp(),  # For driver.py compatibility
                # Note: prev_* fields not tracked in current schema - driver.py LED blinking may not work as expected
                'prev_bikes_available': 0,  # Driver.py compatibility - would need schema update for real tracking
                'prev_ebikes_available': 0,  # Driver.py compatibility - would need schema update for real tracking
            })
        return result


class CurrentData(Base):
    """Real-time station status data"""
    __tablename__ = 'current_data'
    station_id: Mapped[str] = mapped_column(ForeignKey('stations.station_id'), primary_key=True)
    bikes_available: Mapped[int]
    docks_available: Mapped[int]
    ebikes_available: Mapped[int]
    last_updated: Mapped[datetime] = mapped_column(nullable=False)
    # Relationship
    station: Mapped['Station'] = relationship('Station', back_populates="current_data")


class HistoricData(Base):
    """Historical data snapshots every 15 minutes"""
    __tablename__ = 'historic_data'

    id: Mapped[int] = mapped_column(primary_key=True)
    station_id: Mapped[str] = mapped_column(ForeignKey('stations.station_id'), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(nullable=False)
    bikes_available: Mapped[int]
    docks_available: Mapped[int]
    ebikes_available: Mapped[int]
    # Relationship
    station: Mapped['Station'] = relationship("Station", back_populates="historic_data")

    # Index for efficient queries
    __table_args__ = (
        Index('idx_historic_timestamp', 'station_id', 'timestamp'),
    )
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'station_id': self.station_id,
            'timestamp': self.timestamp,
            'bikes_available': self.bikes_available,
            'docks_available': self.docks_available,
            'ebikes_available': self.ebikes_available,
        }


class Route(Base):
    """Current route display data"""
    __tablename__ = 'route'

    station_id: Mapped[str] = mapped_column(ForeignKey('stations.station_id'), primary_key=True)
    color: Mapped[str] = mapped_column(nullable=False)

    # Relationship
    station: Mapped['Station'] = relationship("Station", back_populates="route")

class AppMetadata(Base):
    """Application metadata to speed up interactions"""
    __tablename__ = 'app_metadata'

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    last_updated: Mapped[datetime] = mapped_column()
    viewing_timestamp: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    speed: Mapped[int] = mapped_column(default=1)
    mode: Mapped[int] = mapped_column(default=0)

class DBManager:
    def __init__(self, host, port, dbname, user, password):
        
        self.database_url = URL.create(
            "postgresql",
            username=user,
            password=password,
            host=host,
            port=port,
            database=dbname
        )
        self.engine = create_engine(self.database_url, echo=False)
        self.Session_eng = sessionmaker(bind=self.engine)
        print(f"Connected to PostgreSQL at {host}:{port}/{dbname}")
    
    def get_metadata(self, session=None):
        """Retrieve application metadata"""
        with self.Session_eng() as session:
            meta = session.get(AppMetadata, 1)
            return meta

    def update_metadata(self, session=None, type=None, viewing_timestamp=None, speed=None):
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
                meta.mode = type if type else meta.mode
                if viewing_timestamp is not None:
                    meta.viewing_timestamp = viewing_timestamp
                if speed is not None:
                    meta.speed = speed
                session.commit()
                return meta
        meta = session.get(AppMetadata, 1)
        meta.last_updated = datetime.now()
        meta.mode = type if type else meta.mode
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
        #Load default and first historic metadata for current timestamp: 
        with self.Session_eng() as session:
            self.load_station_data(session)
            # Ensure app_metadata has a default entry (id=1)
            res = session.get(AppMetadata, 1)
            if res is None:
                res = AppMetadata(id=1, last_updated=datetime.now(), mode=0)
                session.add(res)
            session.commit()
            
    def load_station_data(self, session, csv_path='data_src.csv'):
        print(f"Loading station data from {csv_path}...")
        df = pd.read_csv(csv_path)
        loaded = 0
        for _, row in df.iterrows():
            station_id = str(row['station_id'])
            station = Station(
                index=int(row['index']),
                station_id=station_id,
                name=row['matched_name'],
                latitude=float(row['latitude']),
                longitude=float(row['longitude'])
            )
            session.add(station)
            station.current_data = CurrentData(station_id=station_id)
            loaded += 1
        print(f"✓ Loaded {loaded} stations into DB")


    def update_stations(self, archive=False):
        """Fetch GBFS data and update station statuses"""
        updated = 0
        try:
            response = requests.get(STATION_STATUS_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching station status: {e}")
        timestamp = datetime.now()
        with self.Session_eng() as session:
            for station_data in data["data"]["stations"]:
                station_id = station_data["station_id"]
                # Get station and update its current data
                station = session.get(Station, station_id)
                if not station or not station.current_data:
                    continue
                current = station.current_data
                current.bikes_available = station_data.get('num_bikes_available', 0)
                current.docks_available = station_data.get('num_docks_available', 0)
                current.ebikes_available = station_data.get('num_ebikes_available', 0)
                current.last_updated = timestamp
                updated += 1
                # Archive historic data if this is a relevant timestamp
                if archive:
                    historic = HistoricData(
                        station_id=station_id,
                        timestamp=timestamp,
                        bikes_available=current.bikes_available,
                        docks_available=current.docks_available,
                        ebikes_available=current.ebikes_available
                    )
                    session.add(historic)
            
            self.update_metadata(session)
            session.commit()
        
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
                    station_id=station_data['station_id'],
                    color=station_data.get('color', '#ffffff')
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
                .order_by(Station.latitude.asc())  # south (low latitude) first, north last
                .all()
            )
            # Dict preserves insertion order; latitude ascending yields south→north
            return {route.station_id: route.color for route in routes}

    def clear_route(self):
        """Clear route stations"""
        with self.Session_eng() as session:
            session.query(Route).delete()
            self.update_metadata(session, type=LIVE)
            session.commit()

        print("✓ Cleared route stations")

    def get_closest_artifact(self, timestamp):
        with self.Session_eng() as session:
            closest_timestamp_subquery = (
                session.query(HistoricData.timestamp)
                .distinct()
                .order_by(func.abs(func.extract('epoch', HistoricData.timestamp - timestamp)))
                .limit(1)
                .scalar_subquery()
            )            
            # Fetch all historic snapshots for that closest timestamp
            snapshots = (
                session.query(HistoricData)
                .options(joinedload(HistoricData.station))
                .filter(HistoricData.timestamp == closest_timestamp_subquery)
                .all()
            )
            
            if not snapshots:
                return []
            
            return [x.to_dict() for x in snapshots]

    def get_timestamp_range(self):
        """Get the min and max timestamps from historic_data"""
        with self.Session_eng() as session:
            result = session.query(
                func.min(HistoricData.timestamp),
                func.max(HistoricData.timestamp)
            ).first()
            if result and len(result) == 2:
                return {'min': result[0], 'max': result[1]}
            return None

    def get_all_stations_basic(self):
        """Get all stations with basic location info (no current data)"""
        with self.Session_eng() as session:
            stations = session.query(Station).all()
            return [{
                'station_id': s.station_id,
                'name': s.name,
                'latitude': s.latitude,
                'longitude': s.longitude,
                'index': s.index
            } for s in stations]

    def get_stations_by_distance(self, latitude, longitude, limit=None, filter_type=None) -> List[dict]:

        with self.Session_eng() as session:
            # Calculate Euclidean distance in lat/lon space
            # For more accurate geographic distance, use sqrt((lat1-lat2)^2 + (lon1-lon2)^2)
            # Note: This is approximate; for precise distances, use PostGIS ST_Distance
            distance_expr = func.sqrt(
                func.pow(Station.latitude - latitude, 2) + 
                func.pow(Station.longitude - longitude, 2)
            ).label('distance')
            
            query = (
                session.query(Station, distance_expr))
            
            ##Get the closest one with bikes or docks if specified

            if filter_type == 'bikes':
                query = query.join(CurrentData).where(CurrentData.bikes_available > 0)
            elif filter_type == 'docks':
                query = query.join(CurrentData).where(CurrentData.docks_available > 0)
            elif filter_type == 'ebikes':
                query = query.join(CurrentData).where(CurrentData.ebikes_available > 0)
            query = query.order_by(distance_expr)
            
            if limit:
                query = query.limit(limit)            
            
            results = query.all()
            return_list = [station.to_dict() for station, _ in results]
            return return_list

    def get_stations_near_segment(self, lat1, lon1, lat2, lon2, threshold_degrees):
        with self.Session_eng() as session:
            # Create bounding box for quick filtering
            min_lat = min(lat1, lat2) - threshold_degrees
            max_lat = max(lat1, lat2) + threshold_degrees
            min_lon = min(lon1, lon2) - threshold_degrees
            max_lon = max(lon1, lon2) + threshold_degrees
            
            # Vector from segment start to end
            dx = lon2 - lon1
            dy = lat2 - lat1
            length_sq = dx * dx + dy * dy
            
            # Project station onto segment: t = dot(station - start, segment) / length_sq
            # Clamp t to [0, 1] to stay on segment
            t_expr = func.greatest(
                0.0,
                func.least(
                    1.0,
                    ((Station.longitude - lon1) * dx + (Station.latitude - lat1) * dy) / 
                    func.greatest(length_sq, 0.0001)  # Avoid division by zero
                )
            )
            
            # Closest point on segment
            closest_lat = lat1 + t_expr * dy
            closest_lon = lon1 + t_expr * dx
            
            # Distance from station to closest point (Euclidean approximation)
            dist_to_segment = func.sqrt(
                func.pow(Station.latitude - closest_lat, 2) + 
                func.pow(Station.longitude - closest_lon, 2)
            )
            
            # Query stations in bounding box and within threshold distance
            query = (
                session.query(Station)
                .filter(
                    Station.latitude >= min_lat,
                    Station.latitude <= max_lat,
                    Station.longitude >= min_lon,
                    Station.longitude <= max_lon,
                    dist_to_segment <= threshold_degrees
                )
            )
            
            stations = query.all()
            return [station.to_dict() for station in stations]

    def get_all_station_status(self):
        """Get status dict for all stations (for app.py compatibility)"""
        with self.Session_eng() as session:
            stations = session.query(Station).options(joinedload(Station.current_data)).all()
            status_dict = {}
            for station in stations:
                if station.current_data:
                    status_dict[station.station_id] = {
                        'bikes_available': station.current_data.bikes_available,
                        'docks_available': station.current_data.docks_available,
                        'ebikes_available': station.current_data.ebikes_available,
                        'is_renting': True,  # Assume always renting for now
                        'is_returning': True,  # Assume always returning for now
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
                
                if should_include and 'route_stations' in stat and not stat['route_stations'].empty:
                    color = stat['color']
                    
                    for _, srow in stat['route_stations'].iterrows():
                        stations_to_write.append({
                            'station_id': str(srow['station_id']),
                            'color': color
                        })
            
            # Write to database
            if stations_to_write:
                for station_data in stations_to_write:
                    route = Route(
                        station_id=station_data['station_id'],
                        color=station_data['color']
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
        
if __name__ == '__main__':
    # GBFS URL
    STATION_STATUS_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"

    # Update interval in seconds
    UPDATE_INTERVAL = 30
    HISTORY_PERIOD = 5 # Minutes between historic data snapshots
    manager = DBManager(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)
    manager.initial_load()
    last_archive = datetime.now()
    while True:
        time_diff = (datetime.now() - last_archive).total_seconds() / 60.0
        manager.update_stations(archive= time_diff >= HISTORY_PERIOD)
        time.sleep(UPDATE_INTERVAL)