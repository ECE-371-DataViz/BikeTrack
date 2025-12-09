import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import googlemaps
import polyline
import os
from postgres_manager import DBManager
from globals import *
from api_keys import *


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


@st.cache_resource
def get_db_manager():
    """Get PostgreSQL manager instance"""
    manager = DBManager(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)
    return manager


@st.cache_resource
def get_gmaps_client():
    return googlemaps.Client(key=GOOGLE_MAPS)


@st.cache_data(ttl=30, show_spinner="Fetching Live Station Status from PostgreSQL...")
def get_station_status():
    db_manager = get_db_manager()
    return db_manager.get_all_station_status()


def closest_avail_station(coords, type="bikes"):
    db_manager = get_db_manager()
    lat, lon = coords
    stations = db_manager.get_stations_by_distance(
        lat, lon, limit=1, filter_type=type)
    station = stations[0]
    distance_meters = haversine_distance(
        lat, lon, station["latitude"], station["longitude"]
    )
    station["distance"] = distance_meters
    return station


def get_points_on_path(path, threshold=10):
    db_manager = get_db_manager()
    return db_manager.get_stations_on_path(path, threshold)


@st.cache_data(show_spinner="Getting Coordinates...")
def geocode(address: str):
    geolocator = Nominatim(user_agent="streamlit-route-app", timeout=10)
    loc = geolocator.geocode(address)
    if loc is None:
        return None
    return (loc.latitude, loc.longitude)


def render_routes(map, paths, start, end, stations, selected_route=None, route_type="bikes"):
    start_color = "green" if route_type == "ebikes" else "blue"
    folium.Marker(
        start, popup="Origin", icon=folium.Icon(color=start_color, icon="play")
    ).add_to(map)
    folium.Marker(
        end, popup="Destination", icon=folium.Icon(color="red", icon="stop")
    ).add_to(map)
    path_idxs = range(len(paths))
    if selected_route:
        path_idxs = [selected_route - 1]
    for i in path_idxs:
        for segment in paths[i]:
            folium.PolyLine(
                segment,
                color="#3388ff",
                weight=5,
                opacity=0.8,
            ).add_to(map)
        for station in stations[i]:
            folium.CircleMarker(
                location=[station["latitude"], station["longitude"]],
                radius=5,
                color="white",
                fill=True,
                fill_color="white",
                fill_opacity=0.9,
                opacity=0.9,
                weight=1,
                popup=f"Station: {station.get('station_name', station.get('name', station.get('station_id', '')))}",
            ).add_to(map)

def write_route_stations_to_db(start, end, station_list, selected_route=None):
    db_manager = get_db_manager()
    pass
    

def add_all_stations_to_map(m, station_list):
    if not station_list:
        return 0, []
    display_stations = station_list
    for row in display_stations:
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color="#3388ff",
            fill=True,
            fill_color="#3388ff",
            fill_opacity=0.7,
            opacity=0.9,
            weight=1,
            popup=f"Station: {row['station_name']}",
        ).add_to(m)
    return len(display_stations), display_stations


def set_run_true():
    st.session_state["run"] = True


def clear_route():
    """Clear the route from PostgreSQL and reset UI state"""
    st.session_state["run"] = False
    st.session_state["selected_route"] = None
    st.session_state["zoom_to_station"] = None
    st.session_state["route_written"] = False
    # Clear route data from PostgreSQL
    db_manager = get_db_manager()
    db_manager.clear_route()
    st.success("✓ Route cleared - LEDs will return to normal status mode")


def reset_run():
    st.session_state["click_origin"] = None
    st.session_state["click_destination"] = None
    st.session_state["click_explore"] = None
    st.session_state["run"] = False
    st.session_state["zoom_to_station"] = None
    st.session_state["selected_route"] = None  # Clear route selection on reset
    st.session_state["route_written"] = False


def reset_run_keep_points():
    """Reset run state but keep clicked origin/destination points"""
    st.session_state["run"] = False
    st.session_state["zoom_to_station"] = None
    st.session_state["selected_route"] = None  # Clear route selection on reset
    st.session_state["route_written"] = False


@st.cache_data(show_spinner="Getting Directions...")
def get_directions(o, d, num_routes):

    travel_mode = "bicycling"
    client = get_gmaps_client()

    # Google Maps expects (lat, lng) format
    origin = f"{o[0]},{o[1]}"
    destination = f"{d[0]},{d[1]}"

    # Request directions with alternatives
    result = client.directions(
        origin=origin,
        destination=destination,
        mode=travel_mode,
        alternatives=(num_routes > 1),
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
        encoded_polyline = route["overview_polyline"]["points"]
        decoded_coords = polyline.decode(encoded_polyline)

        # Convert to GeoJSON format: [lon, lat] pairs
        coordinates = [[lon, lat] for lat, lon in decoded_coords]

        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coordinates},
            "properties": {
                "summary": route.get("summary", ""),
                "distance": route["legs"][0]["distance"]["value"],
                "duration": route["legs"][0]["duration"]["value"],
            },
        }
        features.append(feature)

    # Return in GeoJSON format
    directions = {"type": "FeatureCollection", "features": features}

    return directions


def get_regular_bikes_count(bikes_available, ebikes_available):
    """Calculate number of regular (non-electric) bikes"""
    return bikes_available - ebikes_available


def get_color_for_availability(bikes_available, ebikes_available, docks_available=0):
    """Get color based on availability (matching driver.py logic)"""
    # Grey for stations with no bikes and no docks (out of service)
    if bikes_available == 0 and docks_available == 0:
        return "#808080"  # Grey
    # Green if more than 10% of bikes are ebikes
    elif bikes_available > 0 and (ebikes_available / bikes_available) > 0.1:
        return "#00FF00"  # Green
    # Blue for regular bikes available
    elif bikes_available > 0:
        return "#0000FF"  # Blue
    # Red for no bikes
    else:
        return "#FF0000"  # Red


def general_view_render(map, station_list, gbfs_status):
    """Add all stations to map with color coding based on availability"""
    if not station_list or not gbfs_status:
        return 0

    stations_added = 0
    for station in station_list:
        station_id = str(station["station_id"])
        if station_id in gbfs_status:
            status = gbfs_status[station_id]
            bikes = status["bikes_available"]
            ebikes = status["ebikes_available"]
            docks = status["docks_available"]
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
                location=[station["latitude"], station["longitude"]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                opacity=0.9,
                weight=2,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(map)
            stations_added += 1

    return stations_added


def add_historic_view_stations(m, station_list, historic_data):
    """Add stations to map with historic data"""
    if not station_list or not historic_data:
        return 0

    # Create a lookup dict for historic data by station_id
    historic_dict = {item["station_id"]: item for item in historic_data}

    stations_added = 0
    for station in station_list:
        station_id = str(station["station_id"])
        if station_id in historic_dict:
            historic = historic_dict[station_id]
            bikes = historic["bikes_available"]
            ebikes = historic["ebikes_available"]
            docks = historic["docks_available"]
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
                location=[station["latitude"], station["longitude"]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                opacity=0.9,
                weight=2,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(m)
            stations_added += 1

    return stations_added

def init_session_states():
    # Initialize session state
    if "click_origin" not in st.session_state:
        st.session_state["click_origin"] = None
    if "click_destination" not in st.session_state:
        st.session_state["click_destination"] = None
    if "click_explore" not in st.session_state:
        st.session_state["click_explore"] = None
    if "setting_origin" not in st.session_state:
        st.session_state["setting_origin"] = True
    if "run" not in st.session_state:
        st.session_state["run"] = False
    if "zoom_to_station" not in st.session_state:
        st.session_state["zoom_to_station"] = None
    if "selected_route" not in st.session_state:
        st.session_state["selected_route"] = None
    if "app_mode" not in st.session_state:
        st.session_state["app_mode"] = "Route Finder"
    if "route_written" not in st.session_state:
        st.session_state["route_written"] = False

def main():
    # Mode selection
    st.sidebar.title("Mode Selection")
    app_mode = st.sidebar.radio(
        "Choose Mode:",
        ["Route Finder", "General View", "Historic View"],
    )
    st.sidebar.divider()

    if app_mode == "Route Finder":
        st.sidebar.subheader("Route Settings")
        st.session_state["app_mode"] = "Route Finder"
        touch = st.sidebar.checkbox(
            "Click map to set points", on_change=reset_run)

        if not touch:
            origin = st.sidebar.text_input(
                "Origin", "East Village, New York, NY", on_change=reset_run
            )
            destination = st.sidebar.text_input(
                "Destination", "Times Square, New York, NY", on_change=reset_run
            )
            o_c = geocode(origin)
            d_c = geocode(destination)
        else:
            o_c = st.session_state["click_origin"]
            d_c = st.session_state["click_destination"]
        num_routes = st.sidebar.slider(
            "Number of Routes",
            1,
            3,
            1,
            on_change=reset_run_keep_points,
            key="num_routes_slider",
        )

        bike_type = st.sidebar.selectbox(
            "Bike Type Preference",
            options=["All Bikes", "E-Bike Only", "Regular Bike Only"],
            index=0,
            on_change=reset_run_keep_points,
            key="bike_type_selector",
        )
        # Map display text to internal values
        bike_type_map = {
            "All Bikes": "all",
            "E-Bike Only": "ebike",
            "Regular Bike Only": "bike",
        }
        bike_type_value = bike_type_map[bike_type]

        station_threshold = st.sidebar.slider(
            "Station Search Distance (meters)",
            10,
            200,
            50,
            key="station_threshold_slider",
        )

        # Find Routes button in sidebar
        st.sidebar.button(
            "🗺️ Find Routes",
            on_click=set_run_true,
            disabled=(not o_c or not d_c),
            use_container_width=True,
            type="primary",
        )
        if st.session_state["run"]:
            st.sidebar.button(
                "🗑️ Clear Route",
                on_click=clear_route,
                use_container_width=True,
                type="secondary",
            )
            ## Run the route rendering logic, and show that data in the db
            
    elif app_mode == "General View":
        st.sidebar.subheader("Live Station View")
        st.sidebar.info(
            "Viewing all stations with current availability. Stations are color-coded:" \
            "\n\n🟢 Green = >10% of bikes are e-bikes" \
            "\n\n🔵 Blue = Regular bikes available" \
            "\n\n🔴 Red = No bikes available" \
            "\n\n⚫ Grey = Out of service"
        )
        db_manager = get_db_manager()
        db_manager.update_metadata(type=LIVE)
        if st.sidebar.button(
            "🔄 Refresh Data", use_container_width=True, type="primary"
        ):
            st.cache_data.clear()
            st.rerun()
        

    elif app_mode == "Historic View":
        st.sidebar.subheader("Historic Station View")
        db_manager = get_db_manager()
        if db_manager:
            timestamp_range = db_manager.get_timestamp_range()
            if timestamp_range:
                min_time = timestamp_range["min"]
                max_time = timestamp_range["max"]
                if not min_time or not max_time:
                    st.error("No historic data available")
                    st.stop()
                st.sidebar.write(f"📅 Data Range:")
                st.sidebar.write(
                    f"From: {min_time.strftime('%Y-%m-%d %H:%M')}")
                st.sidebar.write(f"To: {max_time.strftime('%Y-%m-%d %H:%M')}")
                if "historic_timestamp" not in st.session_state:
                    st.session_state["historic_timestamp"] = min_time
                selected_datetime = st.sidebar.slider(
                    "Select Time:",
                    min_value=min_time,
                    max_value=max_time,
                    value=st.session_state.get("historic_timestamp", min_time),
                    format="MM/DD/YY HH:mm",
                    key="historic_time_slider",
                )
                speed = st.sidebar.slider(
                    "Playback Speed (Seconds/Step)",
                    1,
                    60,
                    10,
                    key="historic_speed_slider",
                )
                st.session_state["historic_timestamp"] = selected_datetime
                st.sidebar.info(
                    "Stations are color-coded:\n\n🟢 Green = >10% of bikes are e-bikes\n\n🔵 Blue = Regular bikes available\n\n🔴 Red = No bikes available\n\n⚫ Grey = Out of service"
                )
                # Update metadata table for historic view
                db_manager.update_metadata(
                    type=HISTORIC, viewing_timestamp=selected_datetime, speed=speed
                )
        else:
            st.sidebar.error("Could not connect to database")

    # Map rendering section - common for all modes
    db_manager = get_db_manager()
    
    # Create base map centered on Manhattan
    m = folium.Map(
        location=[40.7589, -73.9851],  # Manhattan center
        zoom_start=13,
        tiles="OpenStreetMap"
    )
    
    # Render map based on selected mode
    if app_mode == "Route Finder":
        if st.session_state["run"] and o_c and d_c:
            # Get directions and render routes
            directions = get_directions(o_c, d_c, num_routes)
            if directions:
                paths = []
                stations = []
                route_stations_list = []
                
                for feature in directions["features"]:
                    coords = feature["geometry"]["coordinates"]
                    path = [[lat, lon] for lon, lat in coords]
                    paths.append([path])
                    
                    # Get stations along this route
                    route_stations = get_points_on_path(path, station_threshold)
                    stations.append(route_stations)
                    route_stations_list.append(route_stations)
                
                # Find closest start and end stations
                start_station = closest_avail_station(o_c, type=bike_type_value if bike_type_value != "all" else "bikes")

                end_station = closest_avail_station(d_c, type="docks")
                
                # Write route stations to database with colors
                all_route_stations = []
                
                start_color = "#00FF00" if bike_type_value == "ebike" else "#0000FF"  # Green for ebike, Blue for regular
                all_route_stations.append({
                    "station_id": str(start_station["station_id"]),
                    "color": start_color
                })                
                all_route_stations.append({
                    "station_id": str(end_station["station_id"]),
                    "color": "#FF0000"
                })
                # Add route stations (white)
                for idx, route_stn_list in enumerate(route_stations_list):
                    color = "#FFFFFF"
                    for station in route_stn_list:
                        all_route_stations.append({
                            "station_id": str(station["station_id"]),
                            "color": color
                        })
                # Write to database only if not already written
                if all_route_stations and not st.session_state.get("route_written", False):
                    db_manager.set_route_stations(all_route_stations)
                    st.session_state["route_written"] = True
                # Render the routes on map
                render_routes(
                    m, paths, o_c, d_c, stations, 
                    st.session_state.get("selected_route"), 
                    bike_type_value
                )
                # Show route info
                st.subheader("Route Information")
                for i, feature in enumerate(directions["features"], 1):
                    with st.expander(f"Route {i}", expanded=(i == 1)):
                        distance_km = feature["properties"]["distance"] / 1000
                        duration_min = feature["properties"]["duration"] / 60
                        st.write(f"Distance: {distance_km:.2f} km")
                        st.write(f"Duration: {duration_min:.0f} minutes")
    elif app_mode == "General View":
        # General View mode - show all stations with live data
        station_list = db_manager.get_all_stations()
        gbfs_status = get_station_status()
        if station_list and gbfs_status:
            stations_added = general_view_render(m, station_list, gbfs_status)
            st.success(f"Displaying {stations_added} stations with live availability data")
        else:
            st.error("Could not load station data")
    
    elif app_mode == "Historic View":
        # Historic View mode - show stations with historic data
        station_list = db_manager.get_all_stations()
        if station_list and "historic_timestamp" in st.session_state:
            selected_timestamp = st.session_state["historic_timestamp"]
            historic_data = db_manager.get_closest_artifact(selected_timestamp)
            if historic_data:
                stations_added = add_historic_view_stations(m, station_list, historic_data)
                st.success(f"Displaying {stations_added} stations at {selected_timestamp.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.warning("No historic data available for selected time")
        else:
            st.error("Could not load station data")
    
    # Display the map
    st_folium(m, use_container_width=True, returned_objects=[])

if __name__ == "__main__":
    init_session_states()
    st.set_page_config(page_title="CitiBike Station Finder", layout="wide")
    st.title("🚲 CitiBike Station Finder")
    main()
