#!/usr/bin/env python3
"""
Redis Manager for BikeTrack
Loads station data from data_src.csv and periodically updates with GBFS status
"""

import redis
import pandas as pd
import requests
import json
import time
import sys
from datetime import datetime

# Redis connection settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# GBFS URL
STATION_STATUS_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"

# Update interval in seconds
UPDATE_INTERVAL = 5


class RedisStationManager:
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB):
        """Initialize Redis connection"""
        self.redis_client = redis.Redis(
            host=host, port=port, db=db, decode_responses=True
        )
        print(f"Connected to Redis at {host}:{port}")

    def load_station_data(self, csv_path="data_src.csv"):
        """Load station data from CSV into Redis"""
        print(f"Loading station data from {csv_path}...")

        df = pd.read_csv(csv_path)
        loaded = 0

        for _, row in df.iterrows():
            station_id = str(row["station_id"])
            station_data = {
                "index": int(row["index"]),
                "station_id": station_id,
                "name": row["matched_name"],
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "bikes_available": 0,
                "docks_available": 0,
                "ebikes_available": 0,
                "is_renting": 1,
                "is_returning": 1,
                "is_updated": False,
                "last_updated": datetime.now().isoformat(),
            }

            # Store as JSON in Redis with key pattern: station:{station_id}
            self.redis_client.set(f"station:{station_id}", json.dumps(station_data))

            # Add to index set for quick lookup
            self.redis_client.sadd("station:all", station_id)

            loaded += 1

        print(f"✓ Loaded {loaded} stations into Redis")
        return loaded

    def update_station_status(self):
        """Fetch GBFS data and update station statuses in Redis"""
        try:
            response = requests.get(STATION_STATUS_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching station status: {e}")
            return

        updated = 0
        for station in data["data"]["stations"]:
            station_id = station["station_id"]

            # Check if this station exists in our dataset
            if not self.redis_client.sismember("station:all", station_id):
                continue

            # Get existing station data
            station_key = f"station:{station_id}"
            station_json = self.redis_client.get(station_key)

            if station_json:
                station_data = json.loads(station_json)

                # Check if status actually changed
                old_bikes = station_data.get("bikes_available", 0)
                old_docks = station_data.get("docks_available", 0)
                old_ebikes = station_data.get("ebikes_available", 0)

                new_bikes = station.get("num_bikes_available", 0)
                new_docks = station.get("num_docks_available", 0)
                new_ebikes = station.get("num_ebikes_available", 0)

                # Set is_updated flag if any value changed
                status_changed = (
                    old_bikes != new_bikes
                    or old_docks != new_docks
                    or old_ebikes != new_ebikes
                )

                # Store previous values before updating (for comparison in driver.py)
                if status_changed:
                    station_data["prev_bikes_available"] = old_bikes
                    station_data["prev_docks_available"] = old_docks
                    station_data["prev_ebikes_available"] = old_ebikes
                    # Store update timestamp as Unix timestamp (float)
                    station_data["update_timestamp"] = time.time()
                    updated += 1

                # Update status fields
                station_data["bikes_available"] = new_bikes
                station_data["docks_available"] = new_docks
                station_data["ebikes_available"] = new_ebikes
                station_data["is_renting"] = station.get("is_renting", 1)
                station_data["is_returning"] = station.get("is_returning", 1)
                station_data["last_updated"] = datetime.now().isoformat()

                # Save back to Redis
                self.redis_client.set(station_key, json.dumps(station_data))

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if updated > 0:
            print(f"[{timestamp}] Updated {updated} stations")
        return updated

    def get_station(self, station_id):
        """Get a single station's data"""
        station_json = self.redis_client.get(f"station:{station_id}")
        if station_json:
            return json.loads(station_json)
        return None

    def get_all_stations(self):
        """Get all stations data"""
        station_ids = self.redis_client.smembers("station:all")
        stations = []
        for station_id in station_ids:
            station_data = self.get_station(station_id)
            if station_data:
                stations.append(station_data)

        return stations

    def set_route_stations(self, route_stations):
        """Store route stations with their colors"""
        # Clear existing route data
        self.redis_client.delete("route:stations")

        # Store each station with its color
        for station_data in route_stations:
            station_id = station_data["station_id"]
            color = station_data.get("color", "#ffffff")

            # Store in hash: route:stations -> {station_id: color}
            self.redis_client.hset("route:stations", station_id, color)

        print(f"✓ Stored {len(route_stations)} route stations in Redis")

    def get_route_stations(self):
        """Get all route stations and their colors"""
        return self.redis_client.hgetall("route:stations")

    def get_route_update_timestamp(self):
        """Get the timestamp of the last route update"""
        ts = self.redis_client.get("route:update_timestamp")
        return float(ts) if ts else 0.0

    def check_clear_all_flag(self):
        """Check if all LEDs should be cleared (returns True if clear signal exists)"""
        return self.redis_client.get("route:clear_all") == "1"

    def clear_clear_all_flag(self):
        """Clear the clear_all flag after driver.py has cleared LEDs"""
        self.redis_client.delete("route:clear_all")

    def clear_route_stations(self):
        """Clear route stations data and signal driver to clear all LEDs"""
        self.redis_client.delete("route:stations")
        # Set timestamp with special marker for clearing
        self.redis_client.set("route:update_timestamp", str(time.time()))
        self.redis_client.set("route:clear_all", "1")
        print("✓ Cleared route stations")

    def write_route_stations(self, route_stats, selected_route=None):
        """Write route stations and their colors to Redis

        Args:
            route_stats: List of route statistics with station data
            selected_route: Index of selected route (None for all routes)
        """
        # Clear existing route data
        self.redis_client.delete("route:stations")

        # Set timestamp to indicate new route has been written
        self.redis_client.set("route:update_timestamp", str(time.time()))

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

        # Write to Redis as hash: route:stations -> {station_id: color}
        if stations_to_write:
            for station in stations_to_write:
                self.redis_client.hset(
                    "route:stations", station["station_id"], station["color"]
                )

            print(f"✓ Wrote {len(stations_to_write)} route stations to Redis")
            return len(stations_to_write)
        return 0

    def get_all_station_status(self):
        """Get status dict for all stations (for app.py compatibility)"""
        station_ids = self.redis_client.smembers("station:all")
        status_dict = {}

        for station_id in station_ids:
            station_json = self.redis_client.get(f"station:{station_id}")
            if station_json:
                station_data = json.loads(station_json)
                status_dict[station_id] = {
                    "bikes_available": station_data.get("bikes_available", 0),
                    "docks_available": station_data.get("docks_available", 0),
                    "ebikes_available": station_data.get("ebikes_available", 0),
                    "is_renting": station_data.get("is_renting", 1) == 1,
                    "is_returning": station_data.get("is_returning", 1) == 1,
                }

        return status_dict

    def get_stations_by_ids(self, station_ids):
        """Get multiple stations by their IDs

        Args:
            station_ids: List or set of station IDs

        Returns:
            Dict mapping station_id to station data
        """
        result = {}
        for station_id in station_ids:
            station_data = self.get_station(station_id)
            if station_data:
                result[station_id] = station_data
        return result

    def ping(self):
        """Check if Redis connection is alive"""
        return self.redis_client.ping()

    def clear_all_data(self):
        """Clear all station data from Redis (use with caution!)"""
        station_ids = self.redis_client.smembers("station:all")

        for station_id in station_ids:
            self.redis_client.delete(f"station:{station_id}")

        self.redis_client.delete("station:all")
        self.redis_client.delete("route:stations")

        print("✓ Cleared all data from Redis")


def main():
    """Main function to run the Redis manager"""
    manager = RedisStationManager()

    # Load initial data
    manager.load_station_data()

    print(f"\nStarting periodic updates every {UPDATE_INTERVAL} seconds...")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            manager.update_station_status()
            time.sleep(UPDATE_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n✓ Stopped Redis manager")
        sys.exit(0)


if __name__ == "__main__":
    main()
