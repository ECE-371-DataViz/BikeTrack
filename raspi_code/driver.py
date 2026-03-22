
import csv
from datetime import datetime, timedelta
import time
from postgres_manager import DBManager
from globals import *

if PI:
    import board
    import neopixel

    LEDS = neopixel.NeoPixel(board.D18, N_LEDS, brightness=0.1, auto_write=False)
else:

    class MOCK_LEDS:
        def __init__(self, num_leds):
            self.leds = [(0, 0, 0)] * num_leds

        def __setitem__(self, index, color):
            if 0 <= index < len(self.leds):
                self.leds[index] = color

        def fill(self, color):
            self.leds = [color] * len(self.leds)

        def show(self):
            pass
            # print("LED colors updated (simulation):")

    LEDS = MOCK_LEDS(N_LEDS)


def load_logo(csv_path, led_array):
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"])
            r = int(row["r"])
            g = int(row["g"])
            b = int(row["b"])
            led_array[idx] = (r, g, b)


def hex_to_rgb(color_hex, default_color=COLOR_MAP["white"]):
    """Convert a hex color string (e.g., '#0077be') to an RGB tuple."""
    stripped = color_hex.lstrip("#")
    if len(stripped) != 6:
        return default_color
    try:
        return tuple(int(stripped[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        print("Invalid hex color:", color_hex)
        return default_color


# PostgreSQL manager instance
db_manager = DBManager(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)


def init_live():
    stations = db_manager.get_all_stations()
    base_state = {}
    for station in stations:
        station_id = station["station_id"]
        base_state[station_id] = station
        index = station["index"]
        color = get_color(station)
        LEDS[index] = color
    return base_state


def get_color(station):
    # Turn off LED if station has no bikes and no docks (out of service)
    if station["bikes_available"] == 0 and station["docks_available"] == 0:
        return (0, 0, 0)

    base = COLOR_MAP["red"]
    brightness = station["docks_available"] / 25.5
    # Green if more than 25% of bikes are ebikes
    if (
        station["bikes_available"] > 0
        and (station["ebikes_available"] / station["bikes_available"]) > 0.25
    ):
        base = COLOR_MAP["green"]
        brightness = station["ebikes_available"] / (
            station["bikes_available"] + station["docks_available"]
        )
    elif station["bikes_available"] > 0:
        base = COLOR_MAP["blue"]
        brightness = station["bikes_available"] / (
            station["bikes_available"] + station["docks_available"]
        )
    brightness = min(brightness + 0.1, 1.0)
    return [int(b * brightness) for b in base]


def diff(current, new):
    if current["ebikes_available"] < new["ebikes_available"]:
        return COLOR_MAP["green"]
    if current["bikes_available"] < new["bikes_available"]:
        return COLOR_MAP["blue"]
    if current["docks_available"] < new["docks_available"]:
        return COLOR_MAP["red"]
    return None


def blink(update_list):
    for _ in range(NUM_BLINKS):
        for position in update_list:
            color = update_list[position][0]
            LEDS[position] = color
        LEDS.show()
        time.sleep(BLINK_DURATION / 2)
        for position in update_list:
            LEDS[position] = COLOR_MAP["blank"]
        LEDS.show()
        # clear_all_leds()
        time.sleep(BLINK_DURATION / 2)


def live_mode(current_state):
    print("Entering live mode...")
    stations = db_manager.get_all_stations()
    update_list = {}
    for station in stations:
        station_id = station["station_id"]
        blink_color = diff(current_state[station_id], station)
        if blink_color:
            position = station["index"]
            update_list[position] = (blink_color, station)
        current_state[station_id] = station
    print("Updating", len(update_list), "stations")
    blink(update_list)
    for station in current_state:
        station_data = current_state[station]
        position = station_data["index"]
        color = get_color(station_data)
        LEDS[position] = color
    LEDS.show()
    return current_state


def render_grouped_routes(groups, animated_keys=None, step_delay=0.0):
    """Render grouped route data returned by DBManager.get_route_groups()."""
    animated_keys = animated_keys or set()
    clear_all_leds()
    for g in groups:
        color = hex_to_rgb(g["color"], COLOR_MAP["white"])
        animate = g["group_key"] in animated_keys
        for idx in g["indices"]:
            LEDS[idx] = color
            if animate:
                LEDS.show()
                time.sleep(step_delay)
    LEDS.show()


##Assume completely dark backdrop
def route_mode():
    """Route mode with time-aware rendering.
    
    Routes animate by showing stations sequentially based on their appear_at times.
    Polls for active stations and re-renders each cycle.
    """

    print("Entering route mode...")
    clear_all_leds()
    rendered_stations = set()  # Track which stations we've shown before for animation
    
    while True:
        # Check if mode has changed
        meta = db_manager.get_metadata()
        if meta.mode != ROUTE:
            print("Route mode: mode changed externally, exiting")
            break
        
        # Get all route groups
        all_groups = db_manager.get_route_groups()
        if not all_groups:
            print("Route mode: route map is empty, switching to LIVE mode")
            db_manager.update_metadata(in_type=LIVE)
            break
        
        # Get only currently active routes (based on current time and appear_at)
        now_ts = time.time()
        active_groups = {}
        newly_animated = set()
        
        for g in all_groups:
            # Check which stations in this group should be visible now
            active_indices = []
            active_station_ids = []
            
            rows = db_manager.get_route_rows()
            for route, index in rows:
                # Match routes to groups by trip_id or route id
                group_matches = (
                    (route.source_trip_id == g.get("trip_id")) or
                    (route.source_trip_id is None and g.get("group_key", "").startswith("route:"))
                )
                
                if group_matches and route.appear_at <= now_ts < route.appear_at + route.lifetime:
                    station_id = route.station_id
                    # Track newly appearing stations for animation
                    if station_id not in rendered_stations:
                        newly_animated.add(station_id)
                        rendered_stations.add(station_id)
                    
                    if 0 <= index < N_LEDS:
                        active_indices.append(index)
                    active_station_ids.append(station_id)
            
            if active_indices:
                active_groups[g["group_key"]] = {
                    "group_key": g["group_key"],
                    "indices": active_indices,
                    "color": g["color"],
                }
        
        # Render active groups with animation for newly appeared stations
        if active_groups:
            clear_all_leds()
            for group_key, g in active_groups.items():
                color = hex_to_rgb(g["color"], COLOR_MAP["white"])
                animate = group_key in newly_animated or not rendered_stations  # Animate new stations
                for idx in g["indices"]:
                    LEDS[idx] = color
                    if animate:
                        LEDS.show()
                        time.sleep(0.05)
            LEDS.show()
        else:
            # No active stations currently visible
            clear_all_leds()

        time.sleep(0.2)  # Poll for updates frequently
    print("Exiting route mode...")

def historic_mode():
    """Historic playback loop.

    Uses datetime-based trip start scheduling with per-station offsets so each
    trip grows from start to finish. When a trip ends, it is removed and a new
    trip is loaded immediately.
    """
    print("Entering historic mode...")
    meta = db_manager.get_metadata()
    seed_timestamp = meta.viewing_timestamp or datetime.now()
    seed_count = max(1, int(meta.num_trips or 1))
    playback_speed = max(1, int(meta.speed or 1))
    replacement_cursor = seed_timestamp

    db_manager.clear_route(set_live=False)
    loaded = db_manager.load_trips(seed_timestamp, seed_count)
    if loaded == 0:
        print("Historic mode: no trips available to load, switching to LIVE")
        db_manager.update_metadata(in_type=LIVE)
        return

    trip_start_times = {}
    rendered_station_keys = set()

    while True:
        meta = db_manager.get_metadata()
        if meta.mode != HISTORIC:
            print("Historic mode: mode changed externally, exiting")
            print("Exiting historic mode...")
            break

        rows = db_manager.get_route_rows()
        historic_rows = [
            (route, index) for route, index in rows if route.source_trip_id is not None
        ]

        if not historic_rows:
            print("Historic mode: route queue empty, switching to LIVE")
            db_manager.update_metadata(in_type=LIVE)
            break

        # Build per-trip structure once per poll for performance.
        trips = {}
        for route, index in historic_rows:
            trip_id = int(route.source_trip_id)
            if trip_id not in trips:
                trips[trip_id] = {
                    "color": route.color,
                    "lifetime": float(route.lifetime or 0.0),
                    "min_appear": float(route.appear_at or 0.0),
                    "rows": [],
                }
            trips[trip_id]["min_appear"] = min(
                trips[trip_id]["min_appear"], float(route.appear_at or 0.0)
            )
            trips[trip_id]["lifetime"] = max(
                trips[trip_id]["lifetime"], float(route.lifetime or 0.0)
            )
            trips[trip_id]["rows"].append((route, index))

        now_dt = datetime.now()
        active_pixels = []
        ended_trips = []
        newly_visible = set()

        for trip_id, trip in trips.items():
            if trip_id not in trip_start_times:
                trip_start_times[trip_id] = now_dt

            elapsed = (now_dt - trip_start_times[trip_id]).total_seconds()
            lifetime = max(0.0, trip["lifetime"])

            if elapsed >= lifetime:
                ended_trips.append((trip_id, trip))
                continue

            for route, index in sorted(
                trip["rows"], key=lambda x: float(x[0].appear_at or 0.0)
            ):
                station_offset = float(route.appear_at or 0.0) - trip["min_appear"]
                station_offset = max(0.0, station_offset)
                if elapsed < station_offset:
                    continue
                if not (0 <= index < N_LEDS):
                    continue

                station_key = (trip_id, route.station_id)
                if station_key not in rendered_station_keys:
                    newly_visible.add(station_key)
                    rendered_station_keys.add(station_key)

                active_pixels.append((
                    index,
                    hex_to_rgb(trip["color"], COLOR_MAP["white"]),
                    station_key,
                ))

        clear_all_leds()
        for index, color, station_key in active_pixels:
            LEDS[index] = color
            if station_key in newly_visible:
                LEDS.show()
                time.sleep(0.01)
        LEDS.show()

        for trip_id, trip in ended_trips:
            flash_indices = []
            for route, index in trip["rows"]:
                if 0 <= index < N_LEDS:
                    flash_indices.append(index)

            if flash_indices:
                flash_color = hex_to_rgb(trip["color"], COLOR_MAP["white"])
                for _ in range(3):
                    for idx in flash_indices:
                        LEDS[idx] = flash_color
                    LEDS.show()
                    time.sleep(0.15)
                    for idx in flash_indices:
                        LEDS[idx] = COLOR_MAP["blank"]
                    LEDS.show()
                    time.sleep(0.15)

            db_manager.remove_trip(trip_id)
            trip_start_times.pop(trip_id, None)
            rendered_station_keys = {
                key for key in rendered_station_keys if key[0] != trip_id
            }

            replacement_cursor = replacement_cursor + timedelta(
                seconds=max(1.0, max(0.0, trip["lifetime"]) * playback_speed)
            )
            db_manager.load_trips(replacement_cursor, 1)

        time.sleep(0.1)
    print("Exiting historic mode...")


def clear_all_leds():
    """Clear all LEDs to black"""
    # print("Clearing all LEDs...")
    LEDS.fill(COLOR_MAP["blank"])
    LEDS.show()


## Blinks them, and then leaves them on the last color
if __name__ == "__main__":
    clear_all_leds()
    logo_path = "./image_builder/cu_logo_leds_lower.csv"
    load_logo(logo_path, LEDS)
    LEDS.show()
    start = datetime.now()
    print("Loading stations from PostgreSQL...")
    init_live()
    print("✓ Stations loaded!")
    delta = datetime.now() - start
    ## Keep logo up for 10 seconds
    time.sleep(10 - delta.total_seconds() if delta.total_seconds() < 10 else 0)
    LEDS.show()
    db_state = db_manager.get_metadata()
    mode = db_state.mode

    station_states = db_manager.get_all_station_status()

    while True:
        # try:
        s_time = time.time()
        db_state = db_manager.get_metadata()
        if db_state.mode != mode:
            print("Mode changed from", mode, "to", db_state.mode)
            mode = db_state.mode
            clear_all_leds()
        if mode == HISTORIC:
            historic_mode()
            # Re-read mode after historic_mode returns (it blocks until done)
            db_state = db_manager.get_metadata()
            mode = db_state.mode
            clear_all_leds()
            continue
        elif mode == LIVE:
            station_states = live_mode(station_states)
        elif mode == ROUTE:
            route_mode()
        time_dormant = max(0, UPDATE_RATE - (time.time() - s_time))
        time.sleep(time_dormant)
    # except Exception as e:
    #     print("Error in main loop: Changing system behavior to live mode", e)
    #     db_manager.update_metadata(in_type=LIVE)
    #     time.sleep(5)
