###
# Pi specific imports
PI = False
if PI:
    import board
    import neopixel

##
import csv
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
if PI:
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
            print("LED colors updated (simulation):")
    LEDS = MOCK_LEDS(N_LEDS)

def load_logo(csv_path, led_array):
    """
    Load LED color values from a CSV and set them in the provided LED array.
    Does not loop or call show().
    Args:
        csv_path: Path to the CSV file (index,r,g,b columns required)
        led_array: NeoPixel or similar array, modified in-place
    """
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['index'])
            r = int(row['r'])
            g = int(row['g'])
            b = int(row['b'])
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
    if station["bikes_available"] > 0 and (station["ebikes_available"] / station["bikes_available"]) > 0.25:
        base = COLOR_MAP["green"]
        brightness = station["ebikes_available"]/ (station["bikes_available"] + station["docks_available"])        
    elif station["bikes_available"] > 0:
        base = COLOR_MAP["blue"]
        brightness = station["bikes_available"] / (station["bikes_available"] + station["docks_available"])
    brightness = min(brightness+0.1, 1.0)
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
        print("Warning: No historic data for timestamp", timestamp)
        return current_state
    clear_all_leds()
    for station in stations:
        position = station["index"]
        color = get_color(station)
        LEDS[position] = color
        current_state[station["station_id"]] = station
    LEDS.show()
    return current_state


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
    mode_matcher = {LIVE: live_mode, ROUTE: route_mode, HISTORIC: historic_mode}
    db_state = db_manager.get_metadata()
    mode = db_state.mode

    station_states = db_manager.get_all_station_status()

    # Historic display state: track the last timestamp rendered to LEDs
    last_historic_timestamp = None
    
    while True:
        # try:
            s_time = time.time()
            db_state = db_manager.get_metadata()
            if db_state.mode != mode:
                print("Mode changed from", mode, "to", db_state.mode)
                mode = db_state.mode
                clear_all_leds()
                # reset historic state when mode changes
                last_historic_timestamp = None
            if mode == HISTORIC:
                current_viewing_timestamp = db_state.viewing_timestamp
                if current_viewing_timestamp != last_historic_timestamp:
                    print("Viewing historic timestamp:", current_viewing_timestamp)
                    if current_viewing_timestamp:
                        station_states = historic_mode(station_states, current_viewing_timestamp)
                        last_historic_timestamp = current_viewing_timestamp
                time_dormant = max(0, UPDATE_RATE - (time.time() - s_time))
                time.sleep(time_dormant)
            else:
                if mode == LIVE:
                    station_states = live_mode(station_states)
                elif mode == ROUTE:
                    route_mode()
                time_dormant = max(0, UPDATE_RATE - (time.time() - s_time))
                time.sleep(time_dormant)
        # except Exception as e:
        #     print("Error in main loop: Changing system behavior to live mode", e)
        #     db_manager.update_metadata(in_type=LIVE)
        #     time.sleep(5)