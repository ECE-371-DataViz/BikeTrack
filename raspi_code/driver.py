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
N_LEDS = 655
LEDS = neopixel.NeoPixel(board.D18, N_LEDS, brightness=0.05, auto_write=False)

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

# The LED behavior depends on the driver itself...
# LEDS = [0] * 700  # placeholder for LED strip
NUM_BLINKS = 3
BLINK_DURATION = 0.5
UPDATE_RATE = 1  # Seconds between update


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
    
    base = COLOR_MAP["red"] / 255
    brightness = int(min(station["docks_available"], 25.5) * 10)
    # Green if more than 10% of bikes are ebikes
    if station["bikes_available"] > 0 and (station["ebikes_available"] / station["bikes_available"]) > 0.1:
        base = COLOR_MAP["green"] / 255
        brightness = int(min(station["bikes_available"], 25.5) * 10)
    elif station["bikes_available"] > 0:
        base = COLOR_MAP["blue"] / 255
        brightness = int(min(station["bikes_available"], 25.5) * 10)
    return base * brightness


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
        clear_all_leds()
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
    stations = db_manager.get_route_stations()
    for station in stations:
        color = hex_to_rgb(station["color"], COLOR_MAP["white"])
        index = station["index"]
        LEDS[index] = color
        ##Slowly light up each LED from bottom to top
        time.sleep(0.1)


def historic_mode(current_state, timestamp):
    stations = db_manager.get_closest_artifact(timestamp)
    update_list = {}
    for station in stations:
        station_id = station["station_id"]
        blink_color = diff(current_state[station_id], station)
        if blink_color:
            position = station["index"]
            update_list[position] = blink_color
        current_state[station_id] = station
    blink(update_list)
    for station in current_state:
        station_data = current_state[station]
        position = station_data["index"]
        color = get_color(station_data)
        LEDS[position] = color
    LEDS.show()
    return current_state


def clear_all_leds():
    """Clear all LEDs to black"""
    LEDS.fill(COLOR_MAP["blank"])
    LEDS.show()

## Blinks them, and then leaves them on the last color
if __name__ == "__main__":
    print("Loading stations from PostgreSQL...")
    init_live()
    clear_all_leds()
    mode_matcher = {LIVE: live_mode, ROUTE: route_mode, HISTORIC: historic_mode}
    state = db_manager.get_metadata()
    mode = state.mode
    current_state = db_manager.get_all_station_status()
    timestamp = datetime.now()
    while True:
        s_time = time.time()
        state = db_manager.get_metadata()
        if state.mode != mode:
            mode = state.mode
            clear_all_leds()
        if mode == HISTORIC:
            if state.viewing_timestamp != timestamp:
                timestamp = state.viewing_timestamp
                historic_mode(state, timestamp)
                timestamp += timedelta(minutes=HISTORY_PERIOD)
                time.sleep(state.speed)
        else:
            if mode == LIVE:
                current_state = live_mode(current_state)
            elif mode == ROUTE:
                route_mode()
            time_dormant = max(0, UPDATE_RATE - (time.time() - s_time))
            time.sleep(time_dormant)
