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
IN_ROUTE_MODE = False  # Track if we're currently displaying a route

##The LED behavior depends on the driver itself...
LEDS = [0] * 1000
NUM_BLINKS = 3
BLINK_DURATION = 0.5
UPDATE_RATE = 1  # Seconds between update

# Redis manager instance
redis_manager = RedisStationManager()


def load_stations_from_redis():
    """Verify Redis connection and station data"""
    stations = redis_manager.get_all_stations()
    count = len(stations)
    print(f"Verified {count} stations in Redis")
    return count

def get_route_stations():
    """Get route stations and their colors from Redis"""
    return redis_manager.get_route_stations()

def get_data(last_update_ts, last_route_ts):
    """Get updated station data from Redis using timestamp-based tracking
    
    Args:
        last_update_ts: Last timestamp we checked for station updates
        last_route_ts: Last timestamp we checked for route updates
    
    Returns:
        tuple: (success: bool, is_route_update: bool, new_update_ts: float, new_route_ts: float)
    """
    global IN_ROUTE_MODE
    # Check if all LEDs should be cleared
    if redis_manager.check_clear_all_flag():
        print("Clearing all LEDs - returning to blinking mode")
        redis_manager.clear_clear_all_flag()
        IN_ROUTE_MODE = False  # Exit route mode
        # Don't add anything to UPDATE_LIST - just clear
        # Return True for route_update to trigger clear behavior
        return (True, True, time.time(), 0.0)  # Reset route tracking
    
    # Get current route update timestamp from Redis
    current_route_ts = redis_manager.get_route_update_timestamp()
    
    # Check if route has been updated since we last checked
    if current_route_ts > last_route_ts:
        print(f"New route detected (timestamp: {current_route_ts})")
        
        # Get route stations
        route_stations = get_route_stations()
        
        if len(route_stations) == 0:
            # Empty route means we're clearing - return to blinking mode
            print("Empty route - returning to blinking mode")
            IN_ROUTE_MODE = False
            return (True, True, time.time(), current_route_ts)
        
        # We have a route - enter route mode
        IN_ROUTE_MODE = True
        
        # Add route stations to update list
        print(f"Processing {len(route_stations)} route stations from Redis")
        stations_found = 0
        stations_missing = 0
        
        for station_id, color in route_stations.items():
            station_data = redis_manager.get_station(station_id)
            if station_data:
                position = station_data['index']
                # Convert hex color to RGB (colors from app.py are hex codes like #0077be)
                # Default to white if conversion fails
                if color.startswith('#'):
                    try:
                        hex_color = color.lstrip('#')
                        r = int(hex_color[0:2], 16)
                        g = int(hex_color[2:4], 16)
                        b = int(hex_color[4:6], 16)
                        color_rgb = (r, g, b)
                    except:
                        color_rgb = COLOR_MAP["white"]
                else:
                    color_rgb = COLOR_MAP.get(color, COLOR_MAP["white"])
                
                UPDATE_LIST.append((position, color_rgb))
                stations_found += 1
            else:
                stations_missing += 1
        
        print(f"Route stations - Found: {stations_found}, Missing: {stations_missing}")
        
        # Return with updated route timestamp and route_update=True
        return (True, True, time.time(), current_route_ts)
    
    # If we're in route mode, don't process blinking updates for station changes
    if IN_ROUTE_MODE:
        # Just maintain the current route display, no blinking updates
        return (True, False, last_update_ts, last_route_ts)
    
    # Normal blinking mode - check for station updates using timestamps
    current_time = time.time()
    
    # Get all stations and check for updates since last check
    all_stations = redis_manager.get_all_stations()
    
    for station_data in all_stations:
        station_id = station_data['station_id']
        
        # Check if station was updated after our last check
        update_ts = station_data.get('update_timestamp', 0.0)
        if update_ts > last_update_ts:
            position = station_data['index']
            bikes_available = station_data.get('bikes_available', 0)
            ebikes_available = station_data.get('ebikes_available', 0)
            
            # Get previous values to determine color
            prev_bikes = station_data.get('prev_bikes_available', bikes_available)
            prev_ebikes = station_data.get('prev_ebikes_available', ebikes_available)
            # Determine color based on change
            color = COLOR_MAP["red"]
            if ebikes_available > prev_ebikes:
                # Ebike returned (green)
                color = COLOR_MAP["green"]
            elif bikes_available > prev_bikes:
                # Regular bike returned (blue)
                color = COLOR_MAP["blue"]
            UPDATE_LIST.append((position, color, bikes_available, ebikes_available))
    
    # Return normal operation - route_update=False
    return (True, False, current_time, last_route_ts)

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
        print(f"Rendering route with {len(UPDATE_LIST)} stations")
        # Sort UPDATE_LIST by latitude (south to north) for progressive rendering
        # Get station data for each LED to sort by latitude
        stations_with_positions = []
        for (led_idx, color_rgb, *_) in UPDATE_LIST:
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
    
    elif len(UPDATE_LIST) > 0:
        print(f"Blinking mode: Updating {len(UPDATE_LIST)} stations")
        # Normal blinking mode for station updates
        for _ in range(NUM_BLINKS):
            for (led_idx, j, *_) in UPDATE_LIST:
                LEDS[led_idx] = (0, 0, 0)
            time.sleep(BLINK_DURATION / 2)
            for (led_idx, j, *_) in UPDATE_LIST:
                LEDS[led_idx] = j
            time.sleep(BLINK_DURATION / 2)
        for (led_idx, j, num_bikes, num_ebikes) in UPDATE_LIST:
            brightness = max(num_bikes, 25.5) * 10   
            if num_ebikes > 0:
                LEDS[led_idx] = brightness * (0,1,0) 
            else:
                LEDS[led_idx] = brightness * (0,0,1)
            print(f"LED {led_idx}: Bikes={num_bikes}, Ebikes={num_ebikes}")
        UPDATE_LIST.clear()

if __name__ == '__main__':
    print("Loading stations from Redis...")
    load_stations_from_redis()
    
    print(f"Starting LED update loop (every {UPDATE_RATE} seconds)...")
    print("Press Ctrl+C to stop\n")
    
    # Initialize timestamps
    last_update_timestamp = 0.0
    last_route_timestamp = 0.0
    
    while True:
        s_time = time.time()
        success, is_route_update, last_update_timestamp, last_route_timestamp = get_data(last_update_timestamp, last_route_timestamp)
        if success:
            update_leds(route_render=is_route_update)
        time_dormant = max(0, UPDATE_RATE - (time.time() - s_time))
        time.sleep(time_dormant)

