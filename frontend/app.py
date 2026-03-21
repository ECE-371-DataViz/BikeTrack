import streamlit as st
import numpy as np
import folium
from datetime import datetime
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import googlemaps
import polyline
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


@st.cache_data(show_spinner="Fetching historic timestamps from PostgreSQL...")
def fetch_timestamps():
    db_manager = get_db_manager()
    return db_manager.get_timestamps()


@st.cache_data(show_spinner="Fetching historic snapshot from PostgreSQL...")
def fetch_artifact(timestamp):
    db_manager = get_db_manager()
    return db_manager.get_artifact(timestamp)


@st.cache_resource
def get_gmaps_client():
    return googlemaps.Client(key=GOOGLE_MAPS)


@st.cache_data(ttl=30, show_spinner="Fetching Live Station Status from PostgreSQL...")
def get_station_status():
    db_manager = get_db_manager()
    return db_manager.get_all_station_status()


def find_route_stations(
    origin_coords, dest_coords, bike_type, station_threshold, num_routes=1
):
    """
    Optimized multi-route station finder.
    Returns combined route_dict for all routes plus per-route data for rendering.

    Returns:
        route_dict: Combined {station_id: color} for all routes (for DB write)
        paths: List of paths for each route
        stations_per_route: List of station lists for each route
        start_station: Start station dict
        end_station: End station dict
        features: List of Google Maps feature objects for each route
    """
    db_manager = get_db_manager()

    # Find start and end stations (2 DB calls total, shared across all routes)
    start_station = db_manager.get_stations_by_distance(
        origin_coords[0],
        origin_coords[1],
        limit=1,
        filter_type=bike_type if bike_type != "all" else "bikes",
    )
    end_station = db_manager.get_stations_by_distance(
        dest_coords[0], dest_coords[1], limit=1, filter_type="docks"
    )

    if not start_station or not end_station:
        return None, None, None, None, None, None

    start_station = start_station[0]
    end_station = end_station[0]

    start_coords = (start_station["latitude"], start_station["longitude"])
    end_coords   = (end_station["latitude"],   end_station["longitude"])
    # Get Google Maps directions for multiple routes
    directions = get_directions(start_coords, end_coords, num_routes)
    if not directions or not directions["features"]:
        return None, None, None, None, None, None

    # Process all routes - accumulate stations in combined dict
    # Route color palette (matches map): violet, pink, orange
    route_colors = ["#8A2BE2", "#FF69B4", "#FFA500"]
    route_dict = {}
    paths = []
    stations_per_route = []
    features = []

    for idx, feature in enumerate(directions["features"]):
        # Extract path coordinates
        coords = feature["geometry"]["coordinates"]
        path = [[lat, lon] for lon, lat in coords]
        paths.append([path])  # Wrapped in list for backward compatibility

        # Get stations along this specific path (1 DB call per route)
        path_stations = db_manager.get_stations_on_path(
            path, station_threshold)
        stations_per_route.append(path_stations)

        # Determine color for this route
        route_color = route_colors[idx % len(route_colors)]

        # Add to combined dict using route color. Preserve first-seen color for stations
        for stn in path_stations:
            sid = str(stn["station_id"])
            if sid not in route_dict:
                route_dict[sid] = route_color

        features.append(feature)

    # Start/end keep their previous semantics: start is green (ebike) or blue, end is red
    start_color = "#00FF00" if bike_type == "ebike" else "#0000FF"
    end_color = "#FF0000"
    # Ensure start/end override any path color
    route_dict[str(start_station["station_id"])] = start_color
    route_dict[str(end_station["station_id"])] = end_color

    return route_dict, paths, stations_per_route, start_station, end_station, features


def closest_avail_station(coords, type="bikes"):
    db_manager = get_db_manager()
    lat, lon = coords
    stations = db_manager.get_stations_by_distance(
        lat, lon, limit=1, filter_type=type)
    if not stations:
        return None
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


def render_routes(
    map,
    paths,
    start,
    end,
    selected_route=None,
    route_type="bikes",
    start_station=None,
    end_station=None,
):
    """Render route lines and origin/destination markers.

    start_station and end_station are optional dicts that include station metadata
    (name, bikes_available, ebikes_available, docks_available). If provided,
    use station coordinates & build a detailed popup. Otherwise fall back to
    the supplied start/end coordinates.
    """
    start_color = "green" if route_type == "ebikes" else "blue"
    start_location = start
    end_location = end

    # Build origin popup
    if start_station:
        start_location = (
            (start_station["latitude"], start_station["longitude"])
            if isinstance(start_station.get("latitude"), (int, float))
            else start
        )
        regular = get_regular_bikes_count(
            start_station.get("bikes_available", 0),
            start_station.get("ebikes_available", 0),
        )
        start_popup = f"""
        <div style=\"font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; min-width: 220px; padding: 6px;\">
            <b style=\"font-size: 15px;\">{start_station.get('name', start_station.get('station_id', 'Origin'))}</b><br>
            <hr style=\"margin: 6px 0;\">
            🚲 Bikes (total): {start_station.get('bikes_available', 0)}<br>
            ⚡ E-bikes: {start_station.get('ebikes_available', 0)}<br>
            🚴 Regular Bikes: {regular}
        </div>
        """
        folium.Marker(
            start_location,
            popup=folium.Popup(start_popup, max_width=350),
            icon=folium.Icon(color=start_color, icon="play"),
        ).add_to(map)
    else:
        folium.Marker(
            start, popup="Origin", icon=folium.Icon(color=start_color, icon="play")
        ).add_to(map)

    # Build destination popup
    if end_station:
        end_location = (
            (end_station["latitude"], end_station["longitude"])
            if isinstance(end_station.get("latitude"), (int, float))
            else end
        )
        regular_end = get_regular_bikes_count(
            end_station.get("bikes_available", 0),
            end_station.get("ebikes_available", 0),
        )
        end_popup = f"""
        <div style=\"font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; min-width: 220px; padding: 6px;\">
            <b style=\"font-size: 15px;\">{end_station.get('name', end_station.get('station_id', 'Destination'))}</b><br>
            <hr style=\"margin: 6px 0;\">
            🚲 Bikes (total): {end_station.get('bikes_available', 0)}<br>
            ⚡ E-bikes: {end_station.get('ebikes_available', 0)}<br>
            🚴 Regular Bikes: {regular_end}
        </div>
        """
        folium.Marker(
            end_location,
            popup=folium.Popup(end_popup, max_width=350),
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(map)
    else:
        folium.Marker(
            end, popup="Destination", icon=folium.Icon(color="red", icon="stop")
        ).add_to(map)
    path_idxs = range(len(paths))
    if selected_route:
        path_idxs = [selected_route - 1]
    # Route color palette: violet, pink, orange
    route_colors = ["#8A2BE2", "#FF69B4", "#FFA500"]
    for idx in path_idxs:
        route_color = route_colors[idx % len(route_colors)]
        for segment in paths[idx]:
            folium.PolyLine(
                segment,
                color=route_color,
                weight=5,
                opacity=0.85,
            ).add_to(map)


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
    # Set metadata mode back to LIVE
    db_manager.update_metadata(in_type=LIVE)
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


@st.cache_data(ttl=3600, show_spinner=False)
def get_historic_trip_route_data(start_lat, start_lon, end_lat, end_lon):
    """Fetch one Google Maps bicycling route and duration for a historic trip."""
    directions = get_directions((start_lat, start_lon), (end_lat, end_lon), 1)
    if not directions or not directions.get("features"):
        return [], None

    feature = directions["features"][0]
    coordinates = feature["geometry"]["coordinates"]
    # Convert GeoJSON [lon, lat] pairs into Folium [lat, lon] pairs.
    route_geometry = [[lat, lon] for lon, lat in coordinates]
    gmaps_duration_seconds = feature.get("properties", {}).get("duration")
    return route_geometry, gmaps_duration_seconds


def format_duration_minutes(duration_seconds):
    """Format a duration in seconds to one decimal minute string."""
    if duration_seconds is None:
        return "Unavailable"
    return f"{duration_seconds / 60:.1f} min"


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
            <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; min-width: 220px; padding: 6px;">
                <b style="font-size: 15px;">{station.get('name', ' ')}</b><br>
                <hr style="margin: 6px 0;">
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
                popup=folium.Popup(popup_html, max_width=350),
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
            <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; min-width: 220px; padding: 6px;">
                <b style="font-size: 15px;">{station.get('name', 'Station ' + station_id)}</b><br>
                <hr style="margin: 6px 0;">
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
                popup=folium.Popup(popup_html, max_width=350),
            ).add_to(m)
            stations_added += 1

    return stations_added


def build_route_station_color_maps(db_manager):
    """Build station/trip color maps from active historic route rows."""
    station_list = db_manager.get_all_stations()
    station_color_map = db_manager.get_route_station_map()
    trip_color_map = {}

    for group in db_manager.get_route_groups():
        trip_id = group.get("trip_id")
        if trip_id is not None and trip_id not in trip_color_map:
            trip_color_map[trip_id] = group["color"]

    return station_list, station_color_map, trip_color_map


def render_route_station_colors(m, station_list, station_color_map, station_trip_map=None):
    """Render colored station dots from a station_id->color map."""
    if not station_list or not station_color_map:
        return 0

    stations_added = 0
    for station in station_list:
        sid = str(station["station_id"])
        color = station_color_map.get(sid)
        if not color:
            continue

        popup_lines = []
        if station_trip_map and sid in station_trip_map:
            for trip in station_trip_map[sid][:3]:
                popup_lines.append(
                    f"🧭 Route: {trip['start_station_name']} → {trip['end_station_name']}<br>"
                    f"⏱️ Google Maps Time: {trip['gmaps_duration_text']}"
                )
        else:
            popup_lines.append("🧭 Active route station")

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.5; min-width: 230px; padding: 6px;">
            <b style="font-size: 15px;">{station.get('name', sid)}</b><br>
            <hr style="margin: 6px 0;">
            {'<br><br>'.join(popup_lines)}
        </div>
        """
        folium.CircleMarker(
            location=[station["latitude"], station["longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            opacity=0.9,
            weight=2,
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(m)
        stations_added += 1
    return stations_added


# ---- Historic trip playback ----


def start_historic_playback():
    """Callback: start historic playback with trips sampled around the selected time."""
    load_historic_trip_sample(keep_playing=True)


def load_historic_trip_sample(keep_playing=True):
    """Load a fresh random set of historic trips for the selected date/time."""
    db = get_db_manager()
    start_dt = datetime.combine(
        st.session_state["hist_date"],
        st.session_state["hist_time"],
    )
    speed = st.session_state["hist_speed"]
    num = min(st.session_state["hist_num_trips"], 20)

    # Set playback metadata and seed a fresh historic queue.
    db.update_metadata(
        in_type=HISTORIC,
        speed=speed,
        viewing_timestamp=start_dt,
        num_trips=num,
    )
    db.clear_route(set_live=False)
    
    # Get candidate trips and enrich with Google Maps paths
    candidate_trips = db.get_random_trips_in_window(start_dt)
    if not candidate_trips:
        st.session_state["historic_playing"] = False
        st.session_state["historic_no_trips"] = True
        return
    
    # Enrich each trip with Google Maps route path
    trips_with_paths = []
    gmaps_client = get_gmaps_client()
    for trip in candidate_trips[:num]:
        try:
            directions = get_directions(
                (trip["start_lat"], trip["start_lon"]),
                (trip["end_lat"], trip["end_lon"]),
                1
            )
            if directions and directions.get("features"):
                feature = directions["features"][0]
                # Convert GeoJSON [lon, lat] to [lat, lon] for path
                path = [[lat, lon] for lon, lat in feature["geometry"]["coordinates"]]
                gmaps_duration = feature.get("properties", {}).get("duration")
                trips_with_paths.append({
                    **trip,
                    "path": path,
                    "gmaps_duration": gmaps_duration
                })
        except Exception as e:
            print(f"Error fetching Google Maps route for trip {trip.get('trip_id')}: {e}")
            continue
    
    # Load trips with their Google Maps paths
    loaded = db.load_trips_with_gmaps_paths(trips_with_paths)
    if loaded == 0:
        st.session_state["historic_playing"] = False if keep_playing else False
        st.session_state["historic_no_trips"] = True
        return

    station_list, station_color_map, trip_color_map = build_route_station_color_maps(db)

    # Fetch exact active trips from route rows so map lines match playback rows.
    active_trip_ids = [
        group["trip_id"]
        for group in db.get_route_groups()
        if group.get("trip_id") is not None
    ]
    active_trip_ids = active_trip_ids[:num]
    trips = db.get_trips_by_ids(active_trip_ids)
    order_map = {trip_id: idx for idx, trip_id in enumerate(active_trip_ids)}
    trips.sort(key=lambda t: order_map.get(t["trip_id"], 10**9))

    for i, t in enumerate(trips):
        t["position"] = i
        t["color"] = trip_color_map.get(t.get("trip_id"), "#FFFFFF")
        route_geometry, gmaps_duration_seconds = get_historic_trip_route_data(
            t["start_lat"],
            t["start_lon"],
            t["end_lat"],
            t["end_lon"],
        )
        t["route_geometry"] = route_geometry
        t["gmaps_duration_seconds"] = gmaps_duration_seconds
        t["gmaps_duration_text"] = format_duration_minutes(gmaps_duration_seconds)

    st.session_state.update(
        {
            "historic_playing": keep_playing,
            "historic_no_trips": False,
            "historic_trips": trips,
            "historic_station_color_map": station_color_map,
            "historic_station_count": len(station_list),
        }
    )


def shuffle_historic_trips():
    """Callback: reshuffle to a different random trip sample at the selected time."""
    load_historic_trip_sample(keep_playing=True)


def stop_historic_playback():
    """Callback: stop playback by setting mode back to LIVE and clearing historic queue."""
    st.session_state["historic_playing"] = False
    st.session_state["historic_trips"] = []
    st.session_state["historic_station_color_map"] = {}
    st.session_state["historic_station_count"] = 0
    db = get_db_manager()
    db.clear_route(set_live=False)
    db.update_metadata(in_type=LIVE)


def init_session_states():
    needed_keys = [
        "click_origin",
        "click_destination",
        "click_explore",
        "run",
        "zoom_to_station",
        "selected_route",
        "app_mode",
        "route_written",
        "historic_playing",
        "historic_no_trips",
        "historic_trips",
        "historic_station_color_map",
        "historic_station_count",
    ]
    for key in needed_keys:
        if key not in st.session_state:
            st.session_state[key] = None
    if "app_mode" not in st.session_state:    
        st.session_state["app_mode"] = "General View"  # Default mode
    if "historic_trips" not in st.session_state:
        st.session_state["historic_trips"] = []


def main():
    # Mode selection
    st.sidebar.title("Mode Selection")
    app_mode = st.sidebar.radio(
        "Choose Mode:",
        ["Route Finder", "General View", "Historic View"],
    )
    st.sidebar.divider()

    # Stop historic playback when switching away from Historic View
    if app_mode != "Historic View" and st.session_state.get("historic_playing"):
        stop_historic_playback()

    if app_mode == "Route Finder":
        st.sidebar.subheader("Route Settings")
        st.session_state["app_mode"] = "Route Finder"
        # touch = st.sidebar.checkbox(
        #     "Click map to set points", on_change=reset_run)

        # if not touch:
        origin = st.sidebar.text_input(
            "Origin", "East Village, New York, NY", on_change=reset_run
        )
        destination = st.sidebar.text_input(
            "Destination", "Times Square, New York, NY", on_change=reset_run
        )
        o_c = geocode(origin)
        d_c = geocode(destination)
        # else:
        #     o_c = st.session_state["click_origin"]
        #     d_c = st.session_state["click_destination"]
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
            100,
            on_change=reset_run_keep_points,
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
            # Run the route rendering logic, and show that data in the db

    elif app_mode == "General View":
        st.sidebar.subheader("Live Station View")
        st.sidebar.info(
            "Viewing all stations with current availability. Stations are color-coded:"
            "\n\n🟢 Green = >10% of bikes are e-bikes"
            "\n\n🔵 Blue = Regular bikes available"
            "\n\n🔴 Red = No bikes available"
            "\n\n⚫ Grey = Out of service"
        )
        db_manager = get_db_manager()
        db_manager.update_metadata(in_type=LIVE)
        if st.sidebar.button(
            "🔄 Refresh Data", use_container_width=True, type="primary"
        ):
            st.cache_data.clear()
            st.rerun()

    elif app_mode == "Historic View":
        st.sidebar.subheader("Historic Trip Playback")
        st.sidebar.date_input(
            "Start Date",
            value=datetime(2025, 9, 15).date(),
            key="hist_date",
        )
        st.sidebar.time_input(
            "Start Time",
            value=datetime(2025, 9, 15, 8, 0).time(),
            key="hist_time",
        )
        st.sidebar.slider(
            "Playback Speed (x real-time)",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            key="hist_speed",
        )
        st.sidebar.number_input(
            "Number of Trips",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            key="hist_num_trips",
        )
        if not st.session_state.get("historic_playing"):
            st.sidebar.button(
                "▶️ Start Playback",
                on_click=start_historic_playback,
                type="primary",
                use_container_width=True,
            )
        else:
            st.sidebar.button(
                "🔀 Shuffle Trips",
                on_click=shuffle_historic_trips,
                use_container_width=True,
                type="primary",
            )
            st.sidebar.button(
                "⏹️ Stop Playback",
                on_click=stop_historic_playback,
                use_container_width=True,
                type="secondary",
            )
        st.sidebar.info(
            "Plays back random trips from the selected time window on the LED board. "
            "The driver animates trips simultaneously along station paths; "
            "completed trips flash and are replaced with new ones."
        )

    # Map rendering section - common for all modes
    db_manager = get_db_manager()

    # Create base map centered on Manhattan
    m = folium.Map(
        location=[40.7589, -73.9851],  # Manhattan center
        zoom_start=13,
        tiles="CartoDB dark_matter",
    )

    # Render map based on selected mode
    if app_mode == "Route Finder":
        if st.session_state["run"] and o_c and d_c:
            # Optimized call to find routes and stations (2 + num_routes DB calls)
            result = find_route_stations(
                o_c, d_c, bike_type_value, station_threshold, num_routes)

            if result[0] is not None:
                (route_dict,
                    paths,
                    stations_per_route,
                    start_station,
                    end_station,
                    features) = result

                # Write to database only if not already written
                if route_dict and not st.session_state.get("route_written", False):
                    # Convert dict to list and write atomically
                    route_list = [
                        {"station_id": sid, "color": color}
                        for sid, color in route_dict.items()
                    ]
                    db_manager.clear_route()
                    db_manager.set_route_stations(route_list)
                    st.session_state["route_written"] = True

                # Render routes on map
                render_routes(m, paths, o_c, d_c,
                    st.session_state.get("selected_route"),
                    bike_type_value,
                    start_station,
                    end_station)

                route_station_map = db_manager.get_route_station_map()
                render_route_station_colors(
                    m,
                    db_manager.get_all_stations(),
                    route_station_map,
                )

                # Show route info
                st.subheader("Route Information")
                for i, feature in enumerate(features, 1):
                    with st.expander(f"Route {i}", expanded=(i == 1)):
                        distance_km = feature["properties"]["distance"] / 1000
                        duration_min = feature["properties"]["duration"] / 60
                        st.write(f"Distance: {distance_km:.2f} km")
                        st.write(f"Duration: {duration_min:.0f} minutes")
            else:
                st.error(
                    "Could not find a valid route. Please try different locations."
                )
    elif app_mode == "General View":
        # General View mode - show all stations with live data
        station_list = db_manager.get_all_stations()
        gbfs_status = get_station_status()
        if station_list and gbfs_status:
            stations_added = general_view_render(m, station_list, gbfs_status)
            st.success(
                f"Displaying {stations_added} stations with live availability data"
            )
        else:
            st.error("Could not load station data")

    elif app_mode == "Historic View":
        if st.session_state.get("historic_no_trips"):
            st.error(
                "No trips found in the selected time window. Try a different start time."
            )
            st.session_state["historic_no_trips"] = False

        if st.session_state.get("historic_playing"):
            trips = st.session_state["historic_trips"]

            station_trip_map = {}
            for t in trips:
                for sid in [str(t.get("start_station_id")), str(t.get("end_station_id"))]:
                    station_trip_map.setdefault(sid, []).append(t)

            station_list, station_color_map, _ = build_route_station_color_maps(
                db_manager
            )
            rendered = render_route_station_colors(
                m,
                station_list,
                station_color_map,
                station_trip_map=station_trip_map,
            )

            # Show sampled trip start/end pairs using colors sourced from route rows.
            for t in trips:
                color = t["color"]
                bike_label = (
                    (t.get("rideable_type") or "unknown").replace("_", " ").title()
                )
                start_popup_html = f"""
                <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; min-width: 240px; padding: 6px;">
                    <b style="font-size: 15px;">🟢 {t['start_station_name']}</b><br>
                    <hr style="margin: 6px 0;">
                    ➡️ To: {t['end_station_name']}<br>
                    ⏱️ Google Maps Time: {t.get('gmaps_duration_text', format_duration_minutes(None))}<br>
                    🚲 Bike Type: {bike_label}
                </div>
                """
                folium.CircleMarker(
                    location=[t["start_lat"], t["start_lon"]],
                    radius=7,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    weight=2,
                    popup=folium.Popup(start_popup_html, max_width=350),
                ).add_to(m)
                
                # Render the full Google Maps bicycling route geometry
                route_geometry = t.get("route_geometry") or []
                if len(route_geometry) >= 2:
                    folium.PolyLine(
                        route_geometry,
                        color=color,
                        weight=5,
                        opacity=0.55,
                        dash_array="8 6",
                    ).add_to(m)
                else:
                    # Fallback to direct connector if Google route is unavailable
                    folium.PolyLine(
                        [[t["start_lat"], t["start_lon"]],
                            [t["end_lat"], t["end_lon"]]],
                        color=color,
                        weight=4,
                        opacity=0.75,
                        dash_array="6 4",
                    ).add_to(m)
                
                # Faded end-station marker
                end_popup_html = f"""
                <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; min-width: 240px; padding: 6px;">
                    <b style="font-size: 15px;">🔴 {t['end_station_name']}</b><br>
                    <hr style="margin: 6px 0;">
                    ⬅️ From: {t['start_station_name']}<br>
                    ⏱️ Google Maps Time: {t.get('gmaps_duration_text', format_duration_minutes(None))}<br>
                    🚲 Bike Type: {bike_label}
                </div>
                """
                folium.CircleMarker(
                    location=[t["end_lat"], t["end_lon"]],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.3,
                    weight=1,
                    popup=folium.Popup(end_popup_html, max_width=350),
                ).add_to(m)
            st.caption(
                f"Rendering {rendered} active route stations from the shared route backend."
            )
        else:
            st.info(
                "Configure playback settings in the sidebar and click **Start Playback**."
            )

    # Display the map
    st_folium(m, use_container_width=True, height=1200, returned_objects=[])

    # Historic playback: show trip summary below the map (static, no rerun)
    if app_mode == "Historic View" and st.session_state.get("historic_playing"):
        trips = st.session_state["historic_trips"]
        st.success(
            f"Playback started — {len(trips)} trips queued. "
            f"The LED driver will animate them."
        )
        st.subheader("Sample Trips (driver will animate its own selection)")
        table_data = []
        for t in trips:
            table_data.append(
                {
                    "#": t["position"] + 1,
                    "From": t["start_station_name"][:30],
                    "To": t["end_station_name"][:30],
                    "Google Maps Time": t.get("gmaps_duration_text", "Unavailable"),
                    "Type": t.get("rideable_type", ""),
                }
            )
        st.table(table_data)


if __name__ == "__main__":
    init_session_states()
    st.set_page_config(page_title="CitiBike Station Finder", layout="wide")
    st.title("🚲 CitiBike Station Finder")
    main()
