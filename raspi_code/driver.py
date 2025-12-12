###
# Pi specific imports
import board
import neopixel

##
from datetime import datetime, timedelta
import time
from postgres_manager import DBManager
from globals import *

# Constants
COLOR_MAP = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "blank": (0, 0, 0),
}


# The LED behavior depends on the driver itself...
# LEDS = [0] * 700  # placeholder for LED strip
NUM_BLINKS = 5
BLINK_DURATION = 1
UPDATE_RATE = 1  # Seconds between update

N_LEDS = 665
LEDS = neopixel.NeoPixel(board.D18, N_LEDS, brightness=0.1, auto_write=False)

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
    LEDS.show()
    return base_state


def get_color(station):
    # Turn off LED if station has no bikes and no docks (out of service)
    if station["bikes_available"] == 0 and station["docks_available"] == 0:
        return (0, 0, 0)
    
    base = COLOR_MAP["red"]
    brightness = min(station["docks_available"]+30, 25.5) * 10
    # Green if more than 25% of bikes are ebikes
    if station["bikes_available"] > 0 and (station["ebikes_available"] / station["bikes_available"]) > 0.25:
        base = COLOR_MAP["green"]
        brightness = min(station["bikes_available"]+30, 25.5) * 10
    elif station["bikes_available"] > 0:
        base = COLOR_MAP["blue"]
        brightness = min(station["bikes_available"] + 30, 25.5) * 10
    return [int(b * brightness / 255) for b in base]


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


##Assume completely dark backdrop
def route_mode():
    # get_route_stations() returns a dict mapping station_id -> color_hex
    route_map = db_manager.get_route_stations()
    if not route_map:
        print("Route mode: route map is empty, switching to LIVE mode")
        db_manager.update_metadata(in_type=LIVE)
        return
    # Build a station lookup table by station_id for quick access
    all_stations = db_manager.get_all_stations()
    
    station_map = {s["station_id"]: s for s in all_stations}
    clear_all_leds()
    for station_id, color_hex in route_map.items():
        station = station_map.get(station_id)
        if not station:
            print(f"Warning: station_id {station_id} not found in station map")
            continue
        index = station.get("index")
        if index is None or not isinstance(index, int) or index < 0 or index >= N_LEDS:
            print(f"Warning: invalid index for station {station_id}: {index}")
            continue

        color = hex_to_rgb(color_hex, COLOR_MAP["white"])
        LEDS[index] = color
        LEDS.show()
        # Slowly light up each LED from bottom to top
        time.sleep(0.1)
    # Ensure final state is visible
    LEDS.show()

def historic_mode(current_state, timestamp):
    stations = db_manager.get_artifact(timestamp)
    if len(stations) == 0:
        print("No historic data for timestamp, moving to live", timestamp)
        db_manager.update_metadata(in_type=LIVE)
        return current_state
    for station in stations:
        station_data = current_state[station]
        position = station_data["index"]
        color = get_color(station_data)
        LEDS[position] = color
    print("Historic mode: updated LEDs for timestamp", timestamp)
    LEDS.show()
    return current_state


def clear_all_leds():
    """Clear all LEDs to black"""
    # print("Clearing all LEDs...")
    LEDS.fill(COLOR_MAP["blank"])
    LEDS.show()

## Blinks them, and then leaves them on the last color
if __name__ == "__main__":
    print("Loading stations from PostgreSQL...")
    init_live()
    clear_all_leds()
    mode_matcher = {LIVE: live_mode, ROUTE: route_mode, HISTORIC: historic_mode}
    db_state = db_manager.get_metadata()
    mode = db_state.mode
    station_states = db_manager.get_all_station_status()
    starting_timestamp = datetime.now()
    ticks = 0
    # Historic playback state
    historic_timestamps = []
    historic_start_index = None
    while True:
        try:
            s_time = time.time()
            db_state = db_manager.get_metadata()
            if db_state.mode != mode:
                print("Mode changed from", mode, "to", db_state.mode)
                mode = db_state.mode
                clear_all_leds()
                # reset historic play state when mode changes
                historic_timestamps = []
                historic_start_index = None
                ticks = 0
            if mode == HISTORIC:
                print("In Historic Mode")
                # When the viewing timestamp changes (or we have no timestamps yet),
                # load the full list of available historic timestamps and find the
                # starting index to begin playback from.
                if db_state.viewing_timestamp != starting_timestamp or not historic_timestamps:
                    starting_timestamp = db_state.viewing_timestamp
                    ticks = 0
                    historic_timestamps = db_manager.get_timestamps()
                    if not historic_timestamps:
                        print("No historic timestamps available, switching to live")
                        db_manager.update_metadata(in_type=LIVE)
                        continue
                    # Find the first timestamp >= requested starting timestamp
                    start_idx = next((i for i, t in enumerate(historic_timestamps) if t >= starting_timestamp), None)
                    if start_idx is None:
                        print("Starting timestamp beyond available range, switching to live")
                        db_manager.update_metadata(in_type=LIVE)
                        continue
                    historic_start_index = start_idx

                cur_idx = historic_start_index + ticks
                # If we run past the end of available timestamps, switch back to live
                if cur_idx >= len(historic_timestamps):
                    print("Reached end of historic timestamps, switching to live")
                    db_manager.update_metadata(in_type=LIVE)
                    continue

                timestamp = historic_timestamps[cur_idx]
                historic_mode(station_states, timestamp)
                ticks += 1
                time.sleep(db_state.speed)
            else:
                if mode == LIVE:
                    station_states = live_mode(station_states)
                elif mode == ROUTE:
                    route_mode()
                time_dormant = max(0, UPDATE_RATE - (time.time() - s_time))
                time.sleep(time_dormant)
        except Exception as e:
            print("Error in main loop: Changing system behavior to live mode", e)
            db_manager.update_metadata(in_type=LIVE)
            time.sleep(5)