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

# Metadata types
LIVE=0
ROUTE=1
HISTORIC=2

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

    ##Used to figure out how quickly the main app should update its display
    speed: Mapped[int]
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
    
    def update_metadata(self, session=None, type=None):
        if session is None:
            with self.Session_eng() as session:
                meta = session.get(AppMetadata, 1)
                meta.last_updated = datetime.now()
                meta.mode = type if type is not None else meta.mode
                session.commit()
                return    
        meta = session.get(AppMetadata, 1)
        meta.last_updated = datetime.now()
        meta.mode = type if type is not None else meta.mode
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
            routes = session.query(Route).all()
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
                # Get distinct snapshot timestamps across all historic entries
                rows = session.query(HistoricData.timestamp).distinct().all()
                if not rows:
                    return {}
                # Find the closest timestamp to the requested one
                closest_ts = min((r[0] for r in rows), key=lambda t: abs((t - timestamp).total_seconds()))
                # Fetch all historic snapshots for that chosen timestamp, loading the Station relationship
                snapshots = (
                    session.query(HistoricData)
                    .options(joinedload(HistoricData.station))
                    .filter(HistoricData.timestamp == closest_ts)
                    .all()
                )

                result = [x.to_dict() for x in snapshots]
                return result

    def get_timestamp_range(self):
        """Get the min and max timestamps from historic_data"""
        with self.Session_eng() as session:
            result = session.query(
                func.min(HistoricData.timestamp),
                func.max(HistoricData.timestamp)
            ).first()
            if result and result[0] and result[1]:
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

    def write_route_stations(self, route_stats, selected_route=None):
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

def main():
    """Main function to run the PostgreSQL manager"""
    # Backwards-compatible alias: PostgresStationManager is a thin wrapper around DBManager
    class PostgresStationManager(DBManager):
        pass

    manager = PostgresStationManager()

    # Load initial data
    manager.load_station_data()

    print(f"\nStarting periodic updates every {UPDATE_INTERVAL} seconds...")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            manager.update_station_status()
            time.sleep(UPDATE_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n✓ Stopped PostgreSQL manager")
        sys.exit(0)


if __name__ == '__main__':
    # Database connection settings
    DB_HOST = 'localhost'
    DB_PORT = 5432
    DB_NAME = 'biketrack'
    DB_USER = 'biketrack_user'
    DB_PASSWORD = 'password'

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