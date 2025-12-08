import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import googlemaps
import polyline
from datetime import datetime

import sys
import os
from postgres_manager import DBManager
from globals import *

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

@st.cache_resource
def get_db_manager():
    """Get PostgreSQL manager instance"""
    try:
        # Database connection settings
        DB_HOST = os.getenv('DB_HOST', 'localhost')
        DB_PORT = int(os.getenv('DB_PORT', 5432))
        DB_NAME = os.getenv('DB_NAME', 'biketrack')
        DB_USER = os.getenv('DB_USER', 'biketrack_user')
        DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
        
        manager = DBManager(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)
        if manager.ping():
            return manager
        else:
            st.error("Could not connect to PostgreSQL")
            return None
    except Exception as e:
        st.error(f"Could not connect to PostgreSQL: {e}")
        return None

@st.cache_data(ttl=30, show_spinner='Loading CitiBike Stations from PostgreSQL...')
def load_citibike_stations():
    db_manager = get_db_manager()
    if not db_manager:
        return pd.DataFrame(columns=['station_id', 'latitude', 'longitude'])
    
    stations = db_manager.get_all_stations()    
    df = pd.DataFrame([
        {
            'station_id': s['station_id'],
            'latitude': s['latitude'],
            'longitude': s['longitude'],
            'name': s.get('name', ''),
            'index': s.get('index', 0)
        }
        for s in stations
    ])
    
    st.success(f"✓ Loaded {len(df):,} CitiBike stations from PostgreSQL")
    return df

@st.cache_data(ttl=30, show_spinner='Fetching Live Station Status from PostgreSQL...')
def fetch_gbfs_station_status():
    """Fetch live station status from PostgreSQL
    Returns a dict mapping station_id to availability data
    """
    db_manager = get_db_manager()
    if not db_manager:
        return {}
    
    return db_manager.get_all_station_status()

def find_closest_station(coords, type='bikes'):
    db_manager = get_db_manager()
    lat, lon = coords
    
    # Get closest station with specified filter
    stations = db_manager.get_stations_by_distance(lat, lon, filter_type=type)
    if not stations:
        return None
    station = stations[0]
    distance_meters = haversine_distance(lat, lon, station['latitude'], station['longitude'])
    station['distance'] = distance_meters
    return station

@st.cache_data(show_spinner='Filtering Docks Along Path...')
def filter_points_along_path(path, threshold=10):
    db_manager = get_db_manager()
    
    path_array = np.array(path)
    
    # Calculate average latitude for degree conversion
    avg_lat = path_array[:, 0].mean()
    
    # Convert threshold from meters to degrees (approximate)
    threshold_degrees = threshold / (111000 * np.cos(np.radians(avg_lat)))

    stations_dict = {}
    
    for i in range(len(path_array) - 1):
        lat1, lon1 = path_array[i]
        lat2, lon2 = path_array[i + 1]
        
        # Get stations near this segment from DB
        segment_stations = db_manager.get_stations_near_segment(
            lat1, lon1, lat2, lon2, threshold_degrees
        )
        
        # Add to our collection (deduplicating by station_id)
        for station in segment_stations:
            station_id = station['station_id']
            if station_id not in stations_dict:
                stations_dict[station_id] = station
        
    # Convert to DataFrame for compatibility with downstream code
    return pd.DataFrame(list(stations_dict.values()))

@st.cache_data(show_spinner='Getting Coordinates...')
def geocode(address: str):
    geolocator = Nominatim(user_agent="streamlit-route-app", timeout=10)
    loc = geolocator.geocode(address)
    if loc is None:
        return None
    return (loc.latitude, loc.longitude)

@st.cache_resource
def get_gmaps_client():
    return googlemaps.Client(key=GOOGLE_MAPS)

def add_routes_to_map(m, directions, selected_route=None, station_threshold=50, route_type='bikes'):
    db_manager = get_db_manager()
    features = directions["features"]
    route_names = ["Route 1", "Route 2", "Route 3"]
    all_paths = []
    for feature in features:
        geom = feature.get("geometry", {})
        coords_list = geom.get("coordinates", [])
        path = [(lat, lon) for lon, lat in coords_list]
        all_paths.append(path)
    route_stats = []
    for idx, path in enumerate(all_paths):
        folium.PolyLine(
            path,
            color="#ffffff",  # Route lines always white
            weight=4,
            opacity=0.9
        ).add_to(m)
        # Get start and end stations using DB-side filtering
        start_coords = path[0]
        end_coords = path[-1]
        if route_type == 'ebikes':
            start_candidates = db_manager.get_stations_by_distance(start_coords[0], start_coords[1], limit=1, filter_type='ebikes')
        else:
            start_candidates = db_manager.get_stations_by_distance(start_coords[0], start_coords[1], limit=1, filter_type='bikes')
        end_candidates = db_manager.get_stations_by_distance(end_coords[0], end_coords[1], limit=1, filter_type='docks')
        route_stations = []
        if start_candidates:
            start_station = start_candidates[0]
            start_station['color'] = '#2ecc71' if route_type == 'ebikes' else '#0077be'
            route_stations.append(start_station)
        # Add intermediate stations (white)
        mid_stations = filter_points_along_path(path, threshold=station_threshold)
        for station in mid_stations:
            # Avoid duplicating start/end
            if (not start_candidates or station['station_id'] != start_candidates[0]['station_id']) and \
               (not end_candidates or station['station_id'] != end_candidates[0]['station_id']):
                station['color'] = '#ffffff'
                route_stations.append(station)
        if end_candidates:
            end_station = end_candidates[0]
            end_station['color'] = '#e74c3c'
            route_stations.append(end_station)
        route_stats.append({
            'route_name': route_names[idx],
            'color': '#ffffff',
            'station_count': len(route_stations),
            'route_stations': route_stations,
        })
        show_route = (selected_route is None) or (selected_route == idx)
        if show_route and len(route_stations) > 0:
            for station in route_stations:
                folium.CircleMarker(
                    location=[station['latitude'], station['longitude']],
                    radius=5,
                    color=station['color'],
                    fill=True,
                    fill_color=station['color'],
                    fill_opacity=0.9,
                    opacity=0.9,
                    weight=1,
                    popup=f"Station: {station.get('station_name', station.get('name', station.get('station_id', '')))}"
                ).add_to(m)
    return route_stats

def write_route_stations_to_db(route_stats, selected_route=None):
    db_manager = get_db_manager()
    # Flatten all route_stations for DB writing
    all_stations = []
    for route in route_stats:
        for station in route['route_stations']:
            all_stations.append({
                'station_id': station['station_id'],
                'color': station['color']
            })
    db_manager.save_route(all_stations, selected_route)

def add_all_stations_to_map(m, station_list):
    """Add all stations to map with optional limiting for performance"""
    if not station_list:
        return 0, []
    display_stations = station_list
    for row in display_stations:
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color='#3388ff',
            fill=True,
            fill_color='#3388ff',
            fill_opacity=0.7,
            opacity=0.9,
            weight=1,
            popup=f"Station: {row['station_name']}"
        ).add_to(m)
    return len(display_stations), display_stations

def set_run_true():
    st.session_state['run'] = True

def clear_route():
    """Clear the route from PostgreSQL and reset UI state"""
    st.session_state['run'] = False
    st.session_state['selected_route'] = None
    st.session_state['zoom_to_station'] = None
    
    # Clear route data from PostgreSQL
    db_manager = get_db_manager()
    db_manager.clear_route()
    st.success("✓ Route cleared - LEDs will return to normal status mode")

def reset_run():
    st.session_state['click_origin'] = None
    st.session_state['click_destination'] = None
    st.session_state['click_explore'] = None
    st.session_state['run'] = False
    # legacy tree filter removed
    st.session_state['zoom_to_station'] = None
    st.session_state['selected_route'] = None  # Clear route selection on reset

def reset_run_keep_points():
    """Reset run state but keep clicked origin/destination points"""
    st.session_state['run'] = False
    st.session_state['zoom_to_station'] = None
    st.session_state['selected_route'] = None  # Clear route selection on reset

@st.cache_data(show_spinner='Getting Directions...')
def get_directions(o, d, num_routes):
    """Get directions using Google Maps Directions API
    
    Args:
        o: Origin tuple (lat, lon)
        d: Destination tuple (lat, lon)
        num_routes: Number of alternative routes to request
    
    Returns:
        GeoJSON-formatted dict with routes, or None on error
    """
    travel_mode = 'bicycling'
    client = get_gmaps_client()
    
    # Google Maps expects (lat, lng) format
    origin = f"{o[0]},{o[1]}"
    destination = f"{d[0]},{d[1]}"
    
    # Request directions with alternatives
    result = client.directions(
        origin=origin,
        destination=destination,
        mode=travel_mode,
        alternatives=(num_routes > 1)
    )
    
    if not result:
        st.error("No routes found between the specified locations.")
        return None
    
    # Limit to requested number of routes
    routes_to_use = result[:num_routes]
    
    # Convert Google Maps response to GeoJSON format (similar to OpenRouteService)
    features = []
    for route in routes_to_use:
        # Decode the overview polyline
        encoded_polyline = route['overview_polyline']['points']
        decoded_coords = polyline.decode(encoded_polyline)
        
        # Convert to GeoJSON format: [lon, lat] pairs
        coordinates = [[lon, lat] for lat, lon in decoded_coords]
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            },
            "properties": {
                "summary": route.get('summary', ''),
                "distance": route['legs'][0]['distance']['value'],
                "duration": route['legs'][0]['duration']['value']
            }
        }
        features.append(feature)
    
    # Return in GeoJSON format
    directions = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return directions

def get_regular_bikes_count(bikes_available, ebikes_available):
    """Calculate number of regular (non-electric) bikes"""
    return bikes_available - ebikes_available

def get_color_for_availability(bikes_available, ebikes_available):
    """Get color based on availability (matching driver.py logic)"""
    # Green for ebikes available
    if ebikes_available > 0:
        brightness = max(bikes_available, 25.5) * 10
        return '#00FF00'  # Green
    # Blue for regular bikes available
    elif bikes_available > 0:
        brightness = max(bikes_available, 25.5) * 10
        return '#0000FF'  # Blue
    # Red for no bikes
    else:
        return '#FF0000'  # Red

def add_general_view_stations(m, station_list, gbfs_status):
    """Add all stations to map with color coding based on availability"""
    if not station_list or not gbfs_status:
        return 0
    
    stations_added = 0
    for station in station_list:
        station_id = str(station['station_id'])
        if station_id in gbfs_status:
            status = gbfs_status[station_id]
            bikes = status['bikes_available']
            ebikes = status['ebikes_available']
            docks = status['docks_available']
            regular_bikes = get_regular_bikes_count(bikes, ebikes)
            
            # Get color based on availability
            color = get_color_for_availability(bikes, ebikes, docks)
            
            # Create popup with station info
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 150px;">
                <b>{station.get('name', ' ')}</b><br>
                <hr style="margin: 5px 0;">
                🚲 Regular Bikes: {regular_bikes}<br>
                ⚡ E-Bikes: {ebikes}<br>
                🅿️ Docks: {docks}
            </div>
            """
            
            folium.CircleMarker(
                location=[station['latitude'], station['longitude']],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                opacity=0.9,
                weight=2,
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(m)
            stations_added += 1
    
    return stations_added

def add_historic_view_stations(m, station_list, historic_data):
    """Add stations to map with historic data"""
    if not station_list or not historic_data:
        return 0
    
    # Create a lookup dict for historic data by station_id
    historic_dict = {item['station_id']: item for item in historic_data}
    
    stations_added = 0
    for station in station_list:
        station_id = str(station['station_id'])
        if station_id in historic_dict:
            historic = historic_dict[station_id]
            bikes = historic['bikes_available']
            ebikes = historic['ebikes_available']
            docks = historic['docks_available']
            regular_bikes = get_regular_bikes_count(bikes, ebikes)
            
            # Get color based on availability
            color = get_color_for_availability(bikes, ebikes, docks)
            
            # Create popup with station info
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 150px;">
                <b>{station.get('name', 'Station ' + station_id)}</b><br>
                <hr style="margin: 5px 0;">
                🚲 Regular Bikes: {regular_bikes}<br>
                ⚡ E-Bikes: {ebikes}<br>
                🅿️ Docks: {docks}
            </div>
            """
            
            folium.CircleMarker(
                location=[station['latitude'], station['longitude']],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                opacity=0.9,
                weight=2,
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(m)
            stations_added += 1
    
    return stations_added

def main():
    st.set_page_config(page_title="CitiBike Station Finder", layout="wide")

    st.title("🚲 CitiBike Station Finder")

    # Load CitiBike station data
    stations_df = load_citibike_stations()

    # Initialize session state
    if 'click_origin' not in st.session_state:
        st.session_state['click_origin'] = None
    if 'click_destination' not in st.session_state:
        st.session_state['click_destination'] = None
    if 'click_explore' not in st.session_state:
        st.session_state['click_explore'] = None
    if 'setting_origin' not in st.session_state:
        st.session_state['setting_origin'] = True
    if 'run' not in st.session_state:
        st.session_state['run'] = False
    if 'zoom_to_station' not in st.session_state:
        st.session_state['zoom_to_station'] = None
    if 'selected_route' not in st.session_state:
        st.session_state['selected_route'] = None
    if 'app_mode' not in st.session_state:
        st.session_state['app_mode'] = 'Route Finder'

    # Mode selection
    st.sidebar.title("Mode Selection")
    app_mode = st.sidebar.radio(
        "Choose Mode:",
        ['Route Finder', 'General View', 'Historic View'],
        key='app_mode'
    )
    
    st.sidebar.divider()

    # Show different sidebars based on mode
    if app_mode == 'Route Finder':
        st.sidebar.subheader("Route Settings")

        touch = st.sidebar.checkbox('Click map to set points', on_change=reset_run)

        if not touch:
            origin = st.sidebar.text_input("Origin", "East Village, New York, NY", on_change=reset_run)
            destination = st.sidebar.text_input("Destination", "Times Square, New York, NY", on_change=reset_run)
            o_c = geocode(origin)
            d_c = geocode(destination)
        else:
            o_c = st.session_state['click_origin']
            d_c = st.session_state['click_destination']

        num_routes = st.sidebar.slider("Number of Routes", 1, 3, 1, on_change=reset_run_keep_points, key="num_routes_slider")
        
        # Bike type preference
        bike_type = st.sidebar.selectbox(
            "Bike Type Preference",
            options=["All Bikes", "E-Bike Only", "Regular Bike Only"],
            index=0,
            on_change=reset_run_keep_points,
            key="bike_type_selector"
        )
        # Map display text to internal values
        bike_type_map = {
            "All Bikes": "all",
            "E-Bike Only": "ebike",
            "Regular Bike Only": "regular"
        }
        bike_type_value = bike_type_map[bike_type]
        
        station_threshold = st.sidebar.slider('Station Search Distance (meters)', 10, 200, 50, key='station_threshold_slider')

        # Find Routes button in sidebar
        st.sidebar.button("🗺️ Find Routes", on_click=set_run_true, disabled=(not o_c or not d_c), use_container_width=True, type="primary")
        
        # Clear Route button - only show if routes exist
        if st.session_state.get('run'):
            st.sidebar.button("🗑️ Clear Route", on_click=clear_route, use_container_width=True, type="secondary")
    
    elif app_mode == 'General View':
        st.sidebar.subheader("Live Station View")
        st.sidebar.info("Viewing all stations with current availability. Stations are color-coded:\n\n🟢 Green = E-bikes available\n\n🔵 Blue = Regular bikes available\n\n🔴 Red = No bikes available")
        db_manager = get_db_manager()
        db_manager.update_metadata(type=LIVE)
        # Add refresh button
        if st.sidebar.button("🔄 Refresh Data", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    elif app_mode == 'Historic View':
        st.sidebar.subheader("Historic Station View")
        db_manager = get_db_manager()
        if db_manager:
            timestamp_range = db_manager.get_timestamp_range()
            if timestamp_range:
                min_time = timestamp_range['min']
                max_time = timestamp_range['max']
                st.sidebar.write(f"📅 Data Range:")
                st.sidebar.write(f"From: {min_time.strftime('%Y-%m-%d %H:%M')}")
                st.sidebar.write(f"To: {max_time.strftime('%Y-%m-%d %H:%M')}")
                if 'historic_timestamp' not in st.session_state:
                    st.session_state['historic_timestamp'] = min_time
                selected_datetime = st.sidebar.slider(
                    "Select Time:",
                    min_value=min_time,
                    max_value=max_time,
                    value=st.session_state.get('historic_timestamp', min_time),
                    format="MM/DD/YY HH:mm",
                    key='historic_time_slider'
                )
                speed = st.sidebar.slider("Playback Speed (Seconds/Step)", 1, 60, 10, key='historic_speed_slider')
                st.session_state['historic_timestamp'] = selected_datetime
                st.sidebar.info("Stations are color-coded:\n\n🟢 Green = E-bikes available\n\n🔵 Blue = Regular bikes available\n\n🔴 Red = No bikes available")
                # Update metadata table for historic view
                db_manager.update_metadata(type=HISTORIC, viewing_timestamp=selected_datetime, speed=speed)
        else:
            st.sidebar.error("Could not connect to database")

    # Main map area - this will be different for each mode
    if app_mode == 'Route Finder':
        # Original route finder logic
        # Main map area
        col1, col2 = st.columns([3, 1])

        with col1:
            # Determine map center and zoom
            if st.session_state.get('zoom_to_station'):
                # Zoom to specific station
                st_loc = st.session_state['zoom_to_station']
                midpoint = (st_loc['lat'], st_loc['lon'])
                zoom_start = 19  # Close zoom to see the station
            elif o_c and d_c:
                midpoint = ((o_c[0] + d_c[0]) / 2, (o_c[1] + d_c[1]) / 2)
                zoom_start = 13
            else:
                midpoint = (40.7812, -73.9665)
                zoom_start = 13

            map_obj = folium.Map(location=midpoint, zoom_start=zoom_start, tiles='CartoDB positron')

            if o_c:
                folium.Marker(o_c, popup="Origin", icon=folium.Icon(color="green")).add_to(map_obj)
            if d_c:
                folium.Marker(d_c, popup="Destination", icon=folium.Icon(color="red")).add_to(map_obj)

            # Add marker for zoomed station if present
            if st.session_state.get('zoom_to_station'):
                st_loc = st.session_state['zoom_to_station']
                folium.Marker(
                    [st_loc['lat'], st_loc['lon']],
                    popup=f"🚲 Station: {st_loc.get('station_id', '')}",
                    icon=folium.Icon(color="orange", icon="star")
                ).add_to(map_obj)

            if st.session_state['run'] and o_c and d_c:
                o = tuple(o_c)
                d = tuple(d_c)
                
                # Find recommended stations with bikes/docks BEFORE generating routes
                # Use the bike_type_value to determine filter type
                filter_type = 'ebikes' if bike_type_value == 'ebike' else 'bikes'
                start_station = find_closest_station(o, type=filter_type)
                end_station = find_closest_station(d, type='docks')
                
                # Use station locations for routing if found, otherwise use original points
                route_origin = (start_station['latitude'], start_station['longitude']) if start_station else o
                route_destination = (end_station['latitude'], end_station['longitude']) if end_station else d
                
                # Generate directions between the actual start and end stations
                directions = get_directions(route_origin, route_destination, num_routes)

                # Check if directions were successfully retrieved
                if directions is None:
                    st.error("❌ Could not retrieve routes. Please try again in a moment.")
                else:
                    # Add markers for recommended stations
                    if start_station:
                        folium.Marker(
                            [start_station['latitude'], start_station['longitude']],
                            popup=f"🚲 Start Here<br>Station: {start_station['station_id']}<br>Bikes: {start_station['bikes_available']} (E-bikes: {start_station['ebikes_available']})<br>Distance: {start_station['distance']:.0f}m",
                            icon=folium.Icon(color="blue", icon="bicycle", prefix='fa')
                        ).add_to(map_obj)
                    
                    if end_station:
                        folium.Marker(
                            [end_station['latitude'], end_station['longitude']],
                            popup=f"🅿️ End Here<br>Station: {end_station['station_id']}<br>Docks: {end_station['docks_available']}<br>Distance: {end_station['distance']:.0f}m",
                            icon=folium.Icon(color="purple", icon="stop", prefix='fa')
                        ).add_to(map_obj)
                    
                    total_stations, route_stats = add_routes_to_map(
                        map_obj, directions, stations_df,
                        selected_route=st.session_state['selected_route'],
                        station_threshold=station_threshold
                    )
                    
                    # Write route stations to PostgreSQL
                    write_route_stations_to_db(route_stats, st.session_state['selected_route'])

                    with col2:
                        # Show "Back to Route" button if zoomed to a station
                        if st.session_state.get('zoom_to_station'):
                            if st.button("↩️ Back to Route View", key="back_to_route", use_container_width=True, type="primary"):
                                st.session_state['zoom_to_station'] = None
                                st.rerun()
                            st.divider()

                        # Recommended Stations Section
                        st.subheader("🎯 Recommended Stations")
                        
                        if start_station:
                            st.success("**🚲 Start Station**")
                            if start_station.get('name'):
                                st.write(f"**{start_station['name']}**")
                            st.write(f"**Distance:** {start_station['distance']:.0f}m from origin")
                            if bike_type_value == 'ebike':
                                st.write(f"**E-bikes:** {start_station['ebikes_available']}")
                            elif bike_type_value == 'regular':
                                st.write(f"**Regular Bikes:** {start_station['regular_bikes']}")
                            else:
                                st.write(f"**Total Bikes:** {start_station['bikes_available']} (E-bikes: {start_station['ebikes_available']})")
                        else:
                            bike_type_text = bike_type.lower()
                            st.warning(f"⚠️ No stations with {bike_type_text} found near origin")
                        
                        st.write("")
                        
                        if end_station:
                            st.info("**🅿️ End Station**")
                            if end_station.get('name'):
                                st.write(f"**{end_station['name']}**")
                                st.write(f"**Distance:** {end_station['distance']:.0f}m from destination")
                            st.write(f"**Docks Available:** {end_station['docks_available']}")
                        else:
                            st.warning("⚠️ No stations with docks found near destination")
                    
                    st.divider()

                    # Stats summary
                    # Summary: Stations Found (rounded)
                    st.metric("Stations Found", f"{total_stations:,}")
                    # Show total Citibike stations found along the shown routes
                    if route_stats:
                        st.metric("CitiBike Stations Found", f"{total_stations:,}")

                    # No tree filters in station-only app

                    st.divider()

                    # Route selection
                    if route_stats and len(route_stats) > 0:
                        st.subheader("Routes")

                        # All Routes button
                        all_selected = st.session_state['selected_route'] is None
                        if st.button("All Routes" if not all_selected else "All Routes ✓",
                                    key="all_routes_btn", use_container_width=True,
                                    type="primary" if all_selected else "secondary"):
                            st.session_state['selected_route'] = None
                            st.rerun()

                        # Individual route buttons
                        for idx, stat in enumerate(route_stats):
                            is_selected = st.session_state['selected_route'] == idx
                            button_label = f"{stat['route_name']} - {stat['station_count']} stations"
                            if is_selected:
                                button_label += " ✓"

                            if st.button(button_label, key=f"route_btn_{idx}", use_container_width=True,
                                        type="primary" if is_selected else "secondary"):
                                if st.session_state['selected_route'] == idx:
                                    st.session_state['selected_route'] = None
                                else:
                                    st.session_state['selected_route'] = idx
                                st.rerun()

                        st.divider()

                        # Show route details if one is selected
                        if st.session_state['selected_route'] is not None:
                            idx = st.session_state['selected_route']
                            stat = route_stats[idx]

                            # Quick stats
                            st.caption(f"**Route {idx + 1} Details**")
                            st.write(f"🚲 Stations on route: **{stat['station_count']}**")

                            st.divider()

            return_vals = ["last_clicked"] if touch else []
            map_data = st_folium(map_obj, returned_objects= return_vals, use_container_width=True)

            if touch and map_data and map_data.get("last_clicked"):
                clicked_lat = map_data["last_clicked"]["lat"]
                clicked_lng = map_data["last_clicked"]["lng"]
                st.session_state['run'] = False
                if st.session_state['setting_origin']:
                    st.session_state['click_origin'] = [clicked_lat, clicked_lng]
                    st.session_state['setting_origin'] = False
                    st.info("✓ Origin set. Click again to set destination.")
                else:
                    st.session_state['click_destination'] = [clicked_lat, clicked_lng]
                    st.session_state['setting_origin'] = True
                    st.info("✓ Destination set. Click 'Find Routes' to continue.")
                st.rerun()
    
    elif app_mode == 'General View':
        # General view mode - show all stations with current availability
        # Fetch live station status
        gbfs_status = fetch_gbfs_station_status()
        
        # Create map centered on Manhattan
        midpoint = (40.7812, -73.9665)
        zoom_start = 12
        map_obj = folium.Map(location=midpoint, zoom_start=zoom_start, tiles='CartoDB positron')
        
        # Add all stations with color coding
        stations_added = add_general_view_stations(map_obj, stations_df, gbfs_status)
        
        # Display map
        st_folium(map_obj, use_container_width=True, height=700)
        
        # Show stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Stations", f"{stations_added:,}")
        with col2:
            if gbfs_status:
                total_bikes = sum(s['bikes_available'] for s in gbfs_status.values())
                st.metric("Total Bikes Available", f"{total_bikes:,}")
    
    elif app_mode == 'Historic View':
        # Historic view mode - show stations at a specific point in time
        db_manager = get_db_manager()
        
        if db_manager and 'historic_timestamp' in st.session_state:
            selected_datetime = st.session_state['historic_timestamp']
            
            # Fetch historic data for the selected timestamp
            historic_data = db_manager.get_closest_artifact(selected_datetime)
            
            if historic_data:
                # Create map centered on Manhattan
                midpoint = (40.7812, -73.9665)
                zoom_start = 12
                map_obj = folium.Map(location=midpoint, zoom_start=zoom_start, tiles='CartoDB positron')
                
                # Add stations with historic data
                stations_added = add_historic_view_stations(map_obj, stations_df, historic_data)
                
                # Display map
                st_folium(map_obj, use_container_width=True, height=700)
                
                # Show stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Stations Displayed", f"{stations_added:,}")
                with col2:
                    if historic_data:
                        total_bikes = sum(item['bikes_available'] for item in historic_data)
                        st.metric("Total Bikes", f"{total_bikes:,}")
                with col3:
                    # Show the actual timestamp that was used
                    if historic_data and len(historic_data) > 0:
                        actual_time = historic_data[0]['timestamp']
                        st.info(f"Showing data from: {actual_time.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.warning("No historic data available for the selected time")
        else:
            st.error("Could not load historic data")


if __name__ == "__main__":
    main()
