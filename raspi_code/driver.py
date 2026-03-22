
import csv
from datetime import datetime
import time
from postgres_manager import DBManager
from globals import *

if PI:
    import board
    import neopixel

    LEDS = neopixel.NeoPixel(
        board.D18, N_LEDS, brightness=0.1, auto_write=False)
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

# PostgreSQL manager instance
db_manager = DBManager(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)


def clear_all_leds():
    """Clear all LEDs to black"""
    # print("Clearing all LEDs...")
    LEDS.fill(COLOR_MAP["blank"])
    LEDS.show()


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
        return tuple(int(stripped[i: i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        print("Invalid hex color:", color_hex)
        return default_color


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


# New Render pipeline
# We need color, appear_at, and the station_index for core rendering.
# For historic mode, we'll also need the source_trip_id:
# if the station marks the end of a trip, We should load another station into the database.
# Speed is a multiplier on time progression.
    #  default should be such that 30 minute route takes 1 minute, so 30X.
# Appear at is the raw seconds it should take to appear relative to start.
# As such, we can just multiply the render time by start to get the appear_at time.

HISTORIC_LINGER_SECONDS = 5


def _render_at(station_row):
    return station_row["trip_start_at"] + station_row["appear_at"]


def render_routes(route_stations, trip_time, cur_indx):
    # Assume route stations is ordered by render time
    while cur_indx < len(route_stations) and _render_at(route_stations[cur_indx]) <= trip_time:
        station = route_stations[cur_indx]
        index = station["index"]
        if index < 0 or index >= N_LEDS:
            cur_indx += 1
            continue
        color = hex_to_rgb(station["color"])
        LEDS[index] = color
        cur_indx += 1
    LEDS.show()
    return cur_indx


def route_mode(meta):
    speed = meta.speed or 30
    # Assume get_route_rows returns stations ordered by appear_at time
    stations = db_manager.get_route_rows()
    clear_all_leds()
    start_time = time.time()
    cur_indx = 0
    while cur_indx < len(stations):
        now = time.time()
        trip_time = (now - start_time) * (speed)
        cur_indx = render_routes(stations, trip_time, cur_indx)
    # Keep the final route displayed for 10 seconds after completion
    time.sleep(5)


def historic_mode(meta):
    speed = meta.speed or 30
    base_viewing_timestamp = meta.viewing_timestamp or datetime.now()
    start_time = time.time()

    while True:
        now = time.time()
        trip_time = (now - start_time) * speed
        mode, speed, stations = db_manager.historic_tick(
            base_viewing_timestamp,
            trip_time,
            HISTORIC_LINGER_SECONDS,
        )
        if mode != HISTORIC:
            break
        if not stations:
            clear_all_leds()
            time.sleep(0.05)
            continue

        clear_all_leds()
        render_routes(stations, trip_time, 0)
        time.sleep(0.05)


# Blinks them, and then leaves them on the last color
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
    # Keep logo up for 10 seconds
    time.sleep(10 - delta.total_seconds() if delta.total_seconds() < 10 else 0)
    LEDS.show()
    meta = db_manager.get_metadata()
    mode = meta.mode

    station_states = db_manager.get_all_station_status()

    while True:
        # try:
        s_time = time.time()
        meta = db_manager.get_metadata()
        if meta.mode != mode:
            print("Mode changed from", mode, "to", meta.mode)
            mode = meta.mode
            clear_all_leds()
        if mode == HISTORIC:
            historic_mode(meta)
            continue
        elif mode == LIVE:
            station_states = live_mode(station_states)
            time_dormant = max(0, UPDATE_RATE - (time.time() - s_time))
            time.sleep(time_dormant)
        elif mode == ROUTE:
            route_mode(meta)
            continue
    # except Exception as e:
    #     print("Error in main loop: Changing system behavior to live mode", e)
    #     db_manager.update_metadata(in_type=LIVE)
    #     time.sleep(5)
