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
    clear_all_leds()
    animated_keys = animated_keys or set()
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
    groups = db_manager.get_route_groups()
    if not groups:
        print("Route mode: route map is empty, switching to LIVE mode")
        db_manager.update_metadata(in_type=LIVE)
        return
    animated_keys = {g["group_key"] for g in groups}
    render_grouped_routes(groups, animated_keys=animated_keys, step_delay=0.1)


def historic_mode():
    """Historic playback loop using DB queue operations.

    Startup:
    1) clear existing historic rows
    2) load N trips around metadata timestamp via load_trips()

    Runtime:
    - render active trips from route table
    - when a mapped trip expires, remove_trip(source_trip_id)
    - load one replacement trip to keep queue size stable
    """
    meta = db_manager.get_metadata()
    seed_timestamp = meta.viewing_timestamp or datetime.now()
    seed_count = max(1, int(meta.num_trips or 1))
    playback_speed = max(1, int(meta.speed or 1))
    replacement_cursor = seed_timestamp

    clear_all_leds()
    db_manager.clear_route(set_live=False)
    loaded = db_manager.load_trips(seed_timestamp, seed_count)
    if loaded == 0:
        print("Historic mode: no trips available to load, switching to LIVE")
        db_manager.update_metadata(in_type=LIVE)
        return

    rendered_groups = set()

    while True:
        meta = db_manager.get_metadata()
        if meta.mode != HISTORIC:
            print("Historic mode: mode changed externally, exiting")
            return

        now_ts = time.time()
        groups = db_manager.get_route_groups()
        if not groups:
            print("Historic mode: route queue empty, switching to LIVE")
            db_manager.update_metadata(in_type=LIVE)
            return

        # Animate only newly loaded trips.
        newly_rendered = set()
        for g in groups:
            group_key = g["group_key"]
            if group_key in rendered_groups:
                continue
            newly_rendered.add(group_key)
            rendered_groups.add(group_key)

        if newly_rendered:
            render_grouped_routes(groups, animated_keys=newly_rendered, step_delay=0.05)

        expired_groups = [
            g
            for g in groups
            if g.get("maps_to_trip")
            and now_ts >= float(g["appear_at"]) + float(g["lifetime"])
        ]

        for g in expired_groups:
            group_key = g["group_key"]
            flash_indices = g["indices"]
            flash_color = hex_to_rgb(g["color"], COLOR_MAP["white"])

            # Flash the ending trip before replacement.
            for _ in range(3):
                for idx in flash_indices:
                    LEDS[idx] = flash_color
                LEDS.show()
                time.sleep(0.25)
                for idx in flash_indices:
                    LEDS[idx] = COLOR_MAP["blank"]
                LEDS.show()
                time.sleep(0.25)

            db_manager.remove_trip(g.get("trip_id"))
            rendered_groups.discard(group_key)

            replacement_cursor = replacement_cursor + timedelta(
                seconds=max(1.0, float(g["lifetime"]) * playback_speed)
            )
            db_manager.load_trips(replacement_cursor, 1)

        if expired_groups:
            groups = db_manager.get_route_groups()
            render_grouped_routes(groups)

        time.sleep(0.5)


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
