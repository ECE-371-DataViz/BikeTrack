
import csv
from datetime import datetime, timedelta
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


def clear_all_leds(show=True):
    """Clear all LEDs to black"""
    # print("Clearing all LEDs...")
    LEDS.fill(COLOR_MAP["blank"])
    if show:
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
FADE_DURATION_SECONDS = 0.1


def _render_at(station_row):
    return station_row["trip_start_at"] + station_row["appear_at"]


def _trip_complete_at(station_row):
    return station_row["trip_start_at"] + station_row["lifetime"]


def _historic_snapshot():
    stations = [
        row
        for row in db_manager.get_route_rows()
        if row.get("source_trip_id") is not None
    ]
    trip_state = {}
    for row in stations:
        trip_id = row.get("source_trip_id")
        if trip_id is None:
            continue
        entry = trip_state.setdefault(
            trip_id,
            {
                "complete_at": _trip_complete_at(row),
                "indices": set(),
            },
        )
        entry["complete_at"] = max(entry["complete_at"], _trip_complete_at(row))
        idx = row.get("index", -1)
        if 0 <= idx < N_LEDS:
            entry["indices"].add(idx)
    print(f"[SNAPSHOT] Loaded {len(stations)} stations across {len(trip_state)} trips")
    for trip_id, state in trip_state.items():
        print(f"  Trip {trip_id}: {len(state['indices'])} LEDs, completes at {state['complete_at']:.2f}s")
    return stations, trip_state


def render_visible_instant(route_stations, trip_time):
    """Render all currently visible route rows immediately (no fade)."""
    clear_all_leds(show=False)
    next_index = 0
    visible_count = 0
    for idx, station in enumerate(route_stations):
        if _render_at(station) > trip_time:
            next_index = idx
            break
        led_index = station.get("index", -1)
        if 0 <= led_index < N_LEDS:
            LEDS[led_index] = hex_to_rgb(station["color"])
            visible_count += 1
        next_index = idx + 1
    LEDS.show()
    print(f"[RESTORE] Instantly rendered {visible_count} visible stations at trip_time={trip_time:.2f}s")
    return next_index


def fade_leds(led_updates, speed=FADE_DURATION_SECONDS, steps=5):
    """Fade a list of (index, target_color) updates in over `speed` seconds."""
    if not led_updates:
        return

    steps = max(1, int(steps))
    step_delay = max(0.0, float(speed)) / steps
    print(f"[FADE] Starting fade for {len(led_updates)} LEDs over {speed:.3f}s in {steps} steps")

    start_colors = {}
    for index, _ in led_updates:
        if not (0 <= index < N_LEDS):
            continue
        if PI:
            start_colors[index] = tuple(int(c) for c in LEDS[index])
        else:
            start_colors[index] = tuple(int(c) for c in LEDS.leds[index])

    for step in range(1, steps + 1):
        t = step / steps
        for index, target_color in led_updates:
            if index not in start_colors:
                continue
            start_color = start_colors[index]
            blended = (
                int(start_color[0] + (target_color[0] - start_color[0]) * t),
                int(start_color[1] + (target_color[1] - start_color[1]) * t),
                int(start_color[2] + (target_color[2] - start_color[2]) * t),
            )
            LEDS[index] = blended
        LEDS.show()
        if step_delay > 0:
            time.sleep(step_delay)


def render_routes(route_stations, trip_time, cur_indx, fade_speed=FADE_DURATION_SECONDS):
    # Assume route stations is ordered by render time
    led_updates = []
    initial_indx = cur_indx
    while cur_indx < len(route_stations) and _render_at(route_stations[cur_indx]) <= trip_time:
        station = route_stations[cur_indx]
        index = station["index"]
        if index < 0 or index >= N_LEDS:
            cur_indx += 1
            continue
        color = hex_to_rgb(station["color"])
        led_updates.append((index, color))
        cur_indx += 1

    if led_updates:
        print(f"[RENDER] trip_time={trip_time:.2f}s: fading {len(led_updates)} new LEDs (indices {initial_indx}-{cur_indx})")
        fade_leds(led_updates, speed=fade_speed)
    return cur_indx


def route_mode(meta):
    speed = meta.speed or 30
    # Assume get_route_rows returns stations ordered by appear_at time
    stations = db_manager.get_route_rows()
    print(f"[ROUTE_MODE] Starting with {len(stations)} stations, speed={speed}x")
    clear_all_leds()
    start_time = time.time()
    cur_indx = 0
    last_print_time = 0
    while cur_indx < len(stations):
        now = time.time()
        trip_time = (now - start_time) * (speed)
        cur_indx = render_routes(stations, trip_time, cur_indx)
        # Debug print every 2 seconds of real time
        if now - last_print_time > 2:
            print(f"[ROUTE_MODE] Progress: {cur_indx}/{len(stations)} stations ({100*cur_indx/len(stations):.1f}%), trip_time={trip_time:.2f}s")
            last_print_time = now
    print(f"[ROUTE_MODE] Complete! All {len(stations)} stations rendered. Lingering for 5s...")
    # Keep the final route displayed for 5 seconds after completion
    time.sleep(5)


def historic_mode(meta):
    print(f"[HISTORIC_MODE] Initializing with speed={meta.speed}x, viewing_timestamp={meta.viewing_timestamp}")
    speed = meta.speed or 30
    base_viewing_timestamp = meta.viewing_timestamp or datetime.now()
    tracked_viewing_timestamp = meta.viewing_timestamp
    tracked_num_trips = meta.num_trips
    tracked_last_updated = meta.last_updated
    start_time = time.time()
    linger_virtual = HISTORIC_LINGER_SECONDS * speed

    stations, trip_state = _historic_snapshot()
    if not stations:
        print("[HISTORIC_MODE] ERROR: No stations loaded from snapshot. Clearing and exiting.")
        clear_all_leds()
        return

    print(f"[HISTORIC_MODE] Initial snapshot ready: {len(stations)} stations, linger_virtual={linger_virtual:.2f}s")
    clear_all_leds()
    cur_indx = 0
    frame_count = 0
    last_print_time = 0
    last_meta_check_time = 0.0
    meta_check_interval = 0.25
    while True:
        now = time.time()
        trip_time = (now - start_time) * speed

        if now - last_meta_check_time >= meta_check_interval:
            refreshed_meta = db_manager.get_metadata()
            last_meta_check_time = now

            if refreshed_meta.mode != HISTORIC:
                print(f"[HISTORIC_MODE] Mode changed to {refreshed_meta.mode}. Exiting historic mode.")
                break

            refreshed_last_updated = refreshed_meta.last_updated
            tracked_last_updated_safe = tracked_last_updated or datetime.min
            if refreshed_last_updated and refreshed_last_updated > tracked_last_updated_safe:
                prev_speed = speed
                speed = refreshed_meta.speed or speed
                linger_virtual = HISTORIC_LINGER_SECONDS * speed

                should_reset = (
                    refreshed_meta.viewing_timestamp != tracked_viewing_timestamp
                    or refreshed_meta.num_trips != tracked_num_trips
                )

                tracked_last_updated = refreshed_last_updated
                tracked_viewing_timestamp = refreshed_meta.viewing_timestamp
                tracked_num_trips = refreshed_meta.num_trips

                if should_reset:
                    base_viewing_timestamp = refreshed_meta.viewing_timestamp or datetime.now()
                    start_time = now
                    trip_time = 0.0

                    stations, trip_state = _historic_snapshot()
                    if not stations:
                        print("[HISTORIC_MODE] ERROR: No stations after metadata reset. Clearing and exiting.")
                        clear_all_leds()
                        break

                    clear_all_leds(show=False)
                    cur_indx = render_visible_instant(stations, trip_time)
                    print(
                        "[HISTORIC_MODE] Metadata changed during playback. "
                        "Reset timeline and refreshed snapshot."
                    )
                    continue

                if speed != prev_speed:
                    print(
                        f"[HISTORIC_MODE] Playback speed changed to {speed}x "
                        f"(linger_virtual={linger_virtual:.2f}s)."
                    )

        cur_indx = render_routes(stations, trip_time, cur_indx)
        frame_count += 1

        # Debug print every 3 seconds of real time
        if now - last_print_time > 3:
            active_trips = len(trip_state)
            print(f"[HISTORIC_MODE] Frame {frame_count}, trip_time={trip_time:.2f}s, {active_trips} active trips, cur_indx={cur_indx}")
            last_print_time = now

        ended_trip_id = None
        ended_state = None
        for trip_id, state in trip_state.items():
            if trip_time >= state["complete_at"] + linger_virtual:
                ended_trip_id = trip_id
                ended_state = state
                break

        if ended_trip_id is None:
            time.sleep(0.01)
            continue

        fade_out_updates = [(idx, (0, 0, 0)) for idx in ended_state["indices"] if 0 <= idx < N_LEDS]
        if fade_out_updates:
            print(
                f"[HISTORIC_MODE] Trip {ended_trip_id} complete at {trip_time:.2f}s. "
                "Applying standard 0.1s fade-out..."
            )
            fade_leds(fade_out_updates, speed=FADE_DURATION_SECONDS)

        print(f"[HISTORIC_MODE] Trip {ended_trip_id} fade complete at {trip_time:.2f}s. Removing and loading replacement...")
        db_manager.remove_trip(ended_trip_id)

        replacement_start = (
            ended_state["complete_at"]
            + linger_virtual
            + FADE_DURATION_SECONDS
        )
        replacement_timestamp = (
            base_viewing_timestamp + timedelta(seconds=replacement_start)
        )
        print(f"[HISTORIC_MODE] Loading replacement trip at timestamp {replacement_timestamp}...")
        db_manager.load_trips(
            replacement_timestamp,
            1,
            start_at_seconds=replacement_start,
        )

        refreshed_meta = db_manager.get_metadata()
        if refreshed_meta.mode != HISTORIC:
            print(f"[HISTORIC_MODE] Mode changed to {refreshed_meta.mode}. Exiting historic mode.")
            break
        speed = refreshed_meta.speed or speed
        tracked_last_updated = refreshed_meta.last_updated
        tracked_viewing_timestamp = refreshed_meta.viewing_timestamp
        tracked_num_trips = refreshed_meta.num_trips
        linger_virtual = HISTORIC_LINGER_SECONDS * speed
        print(f"[HISTORIC_MODE] Updated speed={speed}x")

        stations, trip_state = _historic_snapshot()
        if not stations:
            print("[HISTORIC_MODE] ERROR: No stations in refreshed snapshot. Clearing and exiting.")
            clear_all_leds()
            break

        print(f"[HISTORIC_MODE] Refreshed snapshot: {len(stations)} stations, instantaneously rendering...")
        # After DB refresh, restore current visible state instantly and only fade
        # genuinely new stations from this point onward.
        cur_indx = render_visible_instant(stations, trip_time)


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
            try:
                historic_mode(meta)
            except Exception as e:
                print("Error in historic mode:", e)
            continue
        elif mode == LIVE:
            station_states = live_mode(station_states)
            time_dormant = max(0, UPDATE_RATE - (time.time() - s_time))
            time.sleep(time_dormant)
        elif mode == ROUTE:
            try:
                route_mode(meta)
            except Exception as e:
                print("Error in route mode:", e)
            continue
    # except Exception as e:
    #     print("Error in main loop: Changing system behavior to live mode", e)
    #     db_manager.update_metadata(in_type=LIVE)
    #     time.sleep(5)
