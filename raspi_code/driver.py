###
# Pi specific imports
##
import time
from redis_manager import RedisStationManager

# Constants
COLOR_MAP = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255)
}
UPDATE_LIST = []

##The LED behavior depends on the driver itself...
LEDS = []
NUM_BLINKS = 3
BLINK_DURATION = 0.5
UPDATE_RATE = 1  # Seconds between update

# Redis manager instance
redis_manager = RedisStationManager()

# Timestamp tracking
last_checked_timestamp = 0.0
last_route_timestamp = 0.0

def load_stations_from_redis():
    """Verify Redis connection and station data"""
    stations = redis_manager.get_all_stations()
    count = len(stations)
    print(f"Verified {count} stations in Redis")
    return count

def get_route_stations():
    """Get route stations and their colors from Redis"""
    return redis_manager.get_route_stations()

def get_data():
    """Get updated station data from Redis using timestamp-based tracking
    
    Returns:
        tuple: (success: bool, clear_all: bool)
    """

    # Check if all LEDs should be cleared
    if redis_manager.check_clear_all_flag():
        print("Clearing all LEDs")
        redis_manager.clear_clear_all_flag()
        last_route_timestamp = 0.0  # Reset route tracking
        return (True, True)  # Success, and clear all LEDs
    
    # Get current route update timestamp
    route_timestamp = redis_manager.get_route_update_timestamp()
    
    # Check if route has been updated since we last checked
    if route_timestamp > last_route_timestamp:
        print(f"New route detected (timestamp: {route_timestamp})")
        last_route_timestamp = route_timestamp
        
        # Get route stations
        route_stations = get_route_stations()
        
        # Add route stations to update list
        for station_id, color in route_stations.items():
            station_data = redis_manager.get_station(station_id)
            if station_data:
                position = station_data['index']
                color_rgb = COLOR_MAP.get(color, COLOR_MAP["white"])
                UPDATE_LIST.append((position, color_rgb))
        
        return (True, True)  # Success, and clear all LEDs first
    
    # Normal operation - check for station updates using timestamps
    route_stations = get_route_stations()
    current_time = time.time()
    
    # Update route stations with white color (always show these)
    for station_id in route_stations.keys():
        station_data = redis_manager.get_station(station_id)
        if station_data:
            position = station_data['index']
            UPDATE_LIST.append((position, COLOR_MAP["white"]))
    
    # Get all stations and check for updates since last check
    all_stations = redis_manager.get_all_stations()
    
    for station_data in all_stations:
        station_id = station_data['station_id']
        
        # Skip route stations - already handled
        if station_id in route_stations:
            continue
        
        # Check if station was updated after our last check
        update_ts = station_data.get('update_timestamp', 0.0)
        if update_ts > last_checked_timestamp:
            position = station_data['index']
            bikes_available = station_data.get('bikes_available', 0)
            ebikes_available = station_data.get('ebikes_available', 0)
            
            # Get previous values to determine color
            prev_bikes = station_data.get('prev_bikes_available', bikes_available)
            prev_ebikes = station_data.get('prev_ebikes_available', ebikes_available)
            
            # Determine color based on change
            if bikes_available < prev_bikes:
                # Bikes decreased - someone took a bike (red)
                UPDATE_LIST.append((position, COLOR_MAP["red"]))
            elif bikes_available > prev_bikes:
                # Bikes increased - someone returned a bike
                if ebikes_available > prev_ebikes:
                    # Ebike returned (green)
                    UPDATE_LIST.append((position, COLOR_MAP["green"]))
                else:
                    # Regular bike returned (blue)
                    UPDATE_LIST.append((position, COLOR_MAP["blue"]))
    
    # Update our last checked timestamp to current time
    last_checked_timestamp = current_time
    
    return (True, False)  # Success, normal operation

def clear_all_leds():
    """Clear all LEDs to black"""
    for i in range(len(LEDS)):
        LEDS[i] = (0, 0, 0)

##Blinks them, and then leaves them on the last color
def update_leds(route_render=False):
    """Update LEDs with new colors
    
    Args:
        route_render: If True, this is a route render - show progressive build from south to north
    """
    if route_render:
        # Route rendering mode: clear all LEDs first
        clear_all_leds()
        time.sleep(0.1)
        
        # Sort UPDATE_LIST by latitude (south to north) for progressive rendering
        # Get station data for each LED to sort by latitude
        stations_with_positions = []
        for (led_idx, color_rgb) in UPDATE_LIST:
            # Find the station by index to get its latitude
            all_stations = redis_manager.get_all_stations()
            for station_data in all_stations:
                if station_data['index'] == led_idx:
                    stations_with_positions.append({
                        'led_idx': led_idx,
                        'color': color_rgb,
                        'latitude': station_data['latitude']
                    })
                    break
        
        # Sort by latitude (ascending = south to north)
        stations_with_positions.sort(key=lambda x: x['latitude'])
        
        # Render progressively from south to north
        for station_info in stations_with_positions:
            LEDS[station_info['led_idx']] = station_info['color']
            time.sleep(0.5)  # 500ms delay between each LED Update        
        UPDATE_LIST.clear()
    else:
        # Normal blinking mode for station updates
        for _ in range(NUM_BLINKS):
            for (led_idx, j) in UPDATE_LIST:
                LEDS[led_idx] = (0, 0, 0)
            time.sleep(BLINK_DURATION / 2)
            for (led_idx, j) in UPDATE_LIST:
                LEDS[led_idx] = j
            time.sleep(BLINK_DURATION / 2)
            #Leave them on the last color, though this behavior could be customized

    UPDATE_LIST.clear()

if __name__ == '__main__':
    print("Loading stations from Redis...")
    load_stations_from_redis()
    
    print(f"Starting LED update loop (every {UPDATE_RATE} seconds)...")
    print("Press Ctrl+C to stop\n")
    
    while True:
        s_time = time.time()
        success, is_new_route = get_data()
        if success:
            update_leds(clear_first=is_new_route)
        time_dormant = max(0, UPDATE_RATE - (time.time() - s_time))
        time.sleep(time_dormant)

