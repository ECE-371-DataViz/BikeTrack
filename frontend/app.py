import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import googlemaps
import polyline
from redis_manager import RedisStationManager
from API_Config import GOOGLE_MAPS

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
def get_redis_manager():
    """Get Redis manager instance"""
    manager = RedisStationManager()
    if manager.ping():
        return manager
    else:
        st.error("Could not connect to Redis")
        return None

@st.cache_data(ttl=30, show_spinner='Loading CitiBike Stations from Redis...')
def load_citibike_stations():
    """Load CitiBike stations from Redis
    Returns a DataFrame with columns ['station_id', 'latitude', 'longitude', 'name', 'index']
    """
    redis_manager = get_redis_manager()
    if not redis_manager:
        return pd.DataFrame(columns=['station_id', 'latitude', 'longitude'])
    
    stations = redis_manager.get_all_stations()
    
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
    
    st.success(f"✓ Loaded {len(df):,} CitiBike stations from Redis")
    return df

@st.cache_data(ttl=30, show_spinner='Fetching Live Station Status from Redis...')
def fetch_gbfs_station_status():
    """Fetch live station status from Redis
    Returns a dict mapping station_id to availability data
    """
    redis_manager = get_redis_manager()
    if not redis_manager:
        return {}
    
    return redis_manager.get_all_station_status()

def find_closest_station_with_bikes(origin_coords, stations_df, gbfs_status, min_bikes=1, search_radius_meters=500, bike_type='all'):
    """Find the closest station to origin that has bikes available
    
    Args:
        origin_coords: tuple (lat, lon) of the starting point
        stations_df: DataFrame with station locations
        gbfs_status: dict with live station status
        min_bikes: minimum number of bikes required
        search_radius_meters: maximum search radius
        bike_type: 'ebike', 'regular', or 'all'
    
    Returns:
        dict with station info or None if not found
    """
    if stations_df.empty or not gbfs_status:
        return None
    
    origin_lat, origin_lon = origin_coords
    
    # Filter stations with bikes available and are renting
    available_stations = []
    for _, station in stations_df.iterrows():
        station_id = str(station['station_id'])
        if station_id in gbfs_status:
            status = gbfs_status[station_id]
            
            # Check bike availability based on type
            has_required_bikes = False
            if bike_type == 'ebike':
                has_required_bikes = status['ebikes_available'] >= min_bikes
            elif bike_type == 'regular':
                regular_bikes = status['bikes_available'] - status['ebikes_available']
                has_required_bikes = regular_bikes >= min_bikes
            else:  # 'all'
                has_required_bikes = status['bikes_available'] >= min_bikes
            
            if has_required_bikes and status['is_renting']:
                distance = haversine_distance(origin_lat, origin_lon, 
                                            station['latitude'], station['longitude'])
                if distance <= search_radius_meters:
                    available_stations.append({
                        'station_id': station_id,
                        'latitude': station['latitude'],
                        'longitude': station['longitude'],
                        'distance': distance,
                        'bikes_available': status['bikes_available'],
                        'ebikes_available': status['ebikes_available'],
                        'regular_bikes': status['bikes_available'] - status['ebikes_available']
                    })
    
    if not available_stations:
        return None
    
    # Return the closest station
    return min(available_stations, key=lambda x: x['distance'])

def find_closest_station_with_docks(destination_coords, stations_df, gbfs_status, min_docks=1, search_radius_meters=500):
    """Find the closest station to destination that has docks available
    
    Args:
        destination_coords: tuple (lat, lon) of the ending point
        stations_df: DataFrame with station locations
        gbfs_status: dict with live station status
        min_docks: minimum number of docks required
        search_radius_meters: maximum search radius
    
    Returns:
        dict with station info or None if not found
    """
    if stations_df.empty or not gbfs_status:
        return None
    
    dest_lat, dest_lon = destination_coords
    
    # Filter stations with docks available and are accepting returns
    available_stations = []
    for _, station in stations_df.iterrows():
        station_id = str(station['station_id'])
        if station_id in gbfs_status:
            status = gbfs_status[station_id]
            if (status['docks_available'] >= min_docks and 
                status['is_returning']):
                distance = haversine_distance(dest_lat, dest_lon, 
                                            station['latitude'], station['longitude'])
                if distance <= search_radius_meters:
                    available_stations.append({
                        'station_id': station_id,
                        'latitude': station['latitude'],
                        'longitude': station['longitude'],
                        'distance': distance,
                        'docks_available': status['docks_available']
                    })
    
    if not available_stations:
        return None
    
    # Return the closest station
    return min(available_stations, key=lambda x: x['distance'])


def point_to_segment_distance(point_lat, point_lon, seg_lat1, seg_lon1, seg_lat2, seg_lon2):
    dx = seg_lon2 - seg_lon1
    dy = seg_lat2 - seg_lat1

    length_sq = dx * dx + dy * dy

    if np.isscalar(length_sq):
        if length_sq == 0:
            return haversine_distance(point_lat, point_lon, seg_lat1, seg_lon1)
        t = np.clip(((point_lon - seg_lon1) * dx + (point_lat - seg_lat1) * dy) / length_sq, 0, 1)
    else:
        t = np.where(length_sq == 0, 0,
                     np.clip(((point_lon - seg_lon1) * dx + (point_lat - seg_lat1) * dy) / length_sq, 0, 1))

    closest_lat = seg_lat1 + t * dy
    closest_lon = seg_lon1 + t * dx

    return haversine_distance(point_lat, point_lon, closest_lat, closest_lon)

@st.cache_data(show_spinner='Filtering Docks Along Path...')
def filter_points_along_path(all_points_df, path, threshold=10, start_end_padding=200):
    """Filter points along a path with extra padding around start and end points
    
    Args:
        all_points_df: DataFrame with station locations
        path: List of (lat, lon) coordinates defining the route
        threshold: Distance threshold in meters for regular path segments
        start_end_padding: Extra padding in meters around start and end points
    """
    path_array = np.array(path)

    # Add significant padding for start and end points
    start_point = path_array[0]
    end_point = path_array[-1]
    
    # Calculate average latitude for degree conversion
    avg_lat = path_array[:, 0].mean()
    
    # Convert padding to degrees
    padding_lat_deg = start_end_padding / 111000
    padding_lon_deg = start_end_padding / (111000 * np.cos(np.radians(avg_lat)))
    
    # Expand bounding box to include padded start and end areas
    min_lat = min(path_array[:, 0].min(), start_point[0] - padding_lat_deg, end_point[0] - padding_lat_deg)
    max_lat = max(path_array[:, 0].max(), start_point[0] + padding_lat_deg, end_point[0] + padding_lat_deg)
    min_lon = min(path_array[:, 1].min(), start_point[1] - padding_lon_deg, end_point[1] - padding_lon_deg)
    max_lon = max(path_array[:, 1].max(), start_point[1] + padding_lon_deg, end_point[1] + padding_lon_deg)

    margin_lat_deg = threshold / 111000
    margin_lon_deg = threshold / (111000 * np.cos(np.radians(avg_lat)))

    bbox_mask = (
        (all_points_df['latitude'] >= min_lat - margin_lat_deg) &
        (all_points_df['latitude'] <= max_lat + margin_lat_deg) &
        (all_points_df['longitude'] >= min_lon - margin_lon_deg) &
        (all_points_df['longitude'] <= max_lon + margin_lon_deg)
    )
    filtered_df = all_points_df[bbox_mask]
    if len(filtered_df) == 0:
        return pd.DataFrame(columns=all_points_df.columns)

    points_lat = filtered_df['latitude'].values
    points_lon = filtered_df['longitude'].values

    min_distances = np.full(len(filtered_df), np.inf)
    for i in range(len(path_array) - 1):
        seg_lat1, seg_lon1 = path_array[i]
        seg_lat2, seg_lon2 = path_array[i + 1]

        seg_min_lat = min(seg_lat1, seg_lat2) - margin_lat_deg
        seg_max_lat = max(seg_lat1, seg_lat2) + margin_lat_deg
        seg_min_lon = min(seg_lon1, seg_lon2) - margin_lon_deg
        seg_max_lon = max(seg_lon1, seg_lon2) + margin_lon_deg

        seg_bbox_mask = (
            (points_lat >= seg_min_lat) & (points_lat <= seg_max_lat) &
            (points_lon >= seg_min_lon) & (points_lon <= seg_max_lon)
        )

        if not seg_bbox_mask.any():
            continue

        seg_points_lat = points_lat[seg_bbox_mask]
        seg_points_lon = points_lon[seg_bbox_mask]

        distances = point_to_segment_distance(seg_points_lat, seg_points_lon, seg_lat1, seg_lon1, seg_lat2, seg_lon2)
        min_distances[seg_bbox_mask] = np.minimum(min_distances[seg_bbox_mask], distances)

    mask = min_distances <= threshold
    return filtered_df[mask]

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

def add_routes_to_map(m, directions, stations_df, selected_route=None, station_threshold=50):
    """Add routes and CitiBike stations along routes to map with per-route statistics and highlighting"""

    features = directions["features"]
    route_colors = ["#0077be", "#e74c3c", "#2ecc71"]
    route_names = ["Route 1", "Route 2", "Route 3"]

    all_paths = []
    for feature in features:
        geom = feature.get("geometry", {})
        coords_list = geom.get("coordinates", [])
        path = [(lat, lon) for lon, lat in coords_list]
        all_paths.append(path)

    # Track unique stations to avoid double-counting across routes
    unique_stations = set()

    # Per-route statistics
    route_stats = []

    # Draw routes with opacity based on selection
    for idx, path in enumerate(all_paths):
        if selected_route is None:
            # Show all routes normally
            opacity = 0.9
            weight = 4
        elif selected_route == idx:
            # Highlight selected route
            opacity = 1.0
            weight = 6
        else:
            # Dim unselected routes
            opacity = 0.15
            weight = 2

        folium.PolyLine(
            path,
            color=route_colors[idx],
            weight=weight,
            opacity=opacity
        ).add_to(m)

    for idx, path in enumerate(all_paths):
        # Find stations along the route
        route_stations = pd.DataFrame(columns=['station_id', 'latitude', 'longitude'])
        if stations_df is not None and not stations_df.empty:
            route_stations = filter_points_along_path(stations_df, path, threshold=station_threshold)

        route_station_count = len(route_stations)

        # Store route stats
        route_stats.append({
            'route_name': route_names[idx],
            'color': route_colors[idx],
            'station_count': route_station_count,
            'route_stations': route_stations,
        })


        # Show stations for selected route, or all routes if none selected
        show_route = (selected_route is None) or (selected_route == idx)
        if show_route and not route_stations.empty:
            # Track unique stations across all shown routes
            for _, srow in route_stations.iterrows():
                station_id = str(srow.get('station_id', ''))
                unique_stations.add(station_id)
                
                # Draw stations in white
                folium.CircleMarker(
                    location=[srow['latitude'], srow['longitude']],
                    radius=5,
                    color='#ffffff',
                    fill=True,
                    fill_color='#ffffff',
                    fill_opacity=0.9,
                    opacity=0.9,
                    weight=1,
                    popup=f"Station: {station_id}"
                ).add_to(m)

    # Return count of unique stations (avoids double-counting)
    total_stations = len(unique_stations)
    return total_stations, route_stats

def write_route_stations_to_redis(route_stats, selected_route=None):
    """Write route stations and their colors to Redis"""
    redis_manager = get_redis_manager()
    if not redis_manager:
        return
    
    redis_manager.write_route_stations(route_stats, selected_route)

def add_all_stations_to_map(m, stations_df, max_stations=10000):
    """Add all stations to map with optional limiting for performance"""
    if stations_df is None or stations_df.empty:
        return 0, pd.DataFrame(columns=['station_id', 'latitude', 'longitude'])

    display_stations = stations_df.copy()
    if len(display_stations) > max_stations:
        display_stations = display_stations.sample(n=max_stations, random_state=42)

    for _, row in display_stations.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color="#ffffff",
            fill=True,
            fill_color="#ffffff",
            fill_opacity=0.8,
            weight=1,
            popup=f"Station: {row.get('station_id', '')}"
        ).add_to(m)

    return len(display_stations), display_stations

def set_run_true():
    st.session_state['run'] = True

def clear_route():
    """Clear the route from Redis and reset UI state"""
    st.session_state['run'] = False
    st.session_state['selected_route'] = None
    st.session_state['zoom_to_station'] = None
    
    # Clear route data from Redis
    redis_manager = get_redis_manager()
    if redis_manager:
        redis_manager.clear_route_stations()
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

    st.sidebar.subheader("CitiBike Station Settings")
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
            
            # Fetch live GBFS data first
            gbfs_status = fetch_gbfs_station_status()
            
            # Find recommended stations with bikes/docks BEFORE generating routes
            start_station = find_closest_station_with_bikes(o, stations_df, gbfs_status, min_bikes=1, search_radius_meters=500, bike_type=bike_type_value)
            end_station = find_closest_station_with_docks(d, stations_df, gbfs_status, min_docks=1, search_radius_meters=500)
            
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
                
                # Write route stations to Redis
                write_route_stations_to_redis(route_stats, st.session_state['selected_route'])

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

if __name__ == "__main__":
    main()
