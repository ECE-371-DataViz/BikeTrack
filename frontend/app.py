import json
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import googlemaps
import polyline
from postgres_manager import DBManager
from globals import *
from api_keys import *
from trip_data import load_trips


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


def find_route_stations(origin_coords, dest_coords, bike_type, station_threshold, num_routes=1):
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
        origin_coords[0], origin_coords[1], limit=1, 
        filter_type=bike_type if bike_type != "all" else "bikes"
    )
    end_station = db_manager.get_stations_by_distance(
        dest_coords[0], dest_coords[1], limit=1, filter_type="docks"
    )
    
    if not start_station or not end_station:
        return None, None, None, None, None, None
    
    start_station = start_station[0]
    end_station = end_station[0]
    
    # Get Google Maps directions for multiple routes
    directions = get_directions(origin_coords, dest_coords, num_routes)
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
        path_stations = db_manager.get_stations_on_path(path, station_threshold)
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


def render_routes(map, paths, start, end, stations, selected_route=None, route_type="bikes", start_station=None, end_station=None):
    """Render routes and label origin/destination stations.

    start_station and end_station are optional dicts that include station metadata
    (name, bikes_available, ebikes_available, docks_available). If provided,
    use station coordinates & build a detailed popup. Otherwise fall back to
    the supplied start/end coordinates.
    """
    start_color = "green" if route_type == "ebikes" else "blue"
    # Build origin popup
    if start_station:
        start_location = (start_station["latitude"], start_station["longitude"]) if isinstance(start_station.get("latitude"), (int, float)) else start
        regular = get_regular_bikes_count(start_station.get("bikes_available", 0), start_station.get("ebikes_available", 0))
        start_popup = f"""
        <div style=\"font-family: Arial, sans-serif; min-width: 150px;\">
            <b>{start_station.get('name', start_station.get('station_id', 'Origin'))}</b><br>
            <hr style=\"margin: 5px 0;\">
            🚲 Bikes (total): {start_station.get('bikes_available', 0)}<br>
            ⚡ E-bikes: {start_station.get('ebikes_available', 0)}<br>
            🚴 Regular Bikes: {regular}
        </div>
        """
        folium.Marker(
            start_location,
            popup=folium.Popup(start_popup, max_width=300),
            icon=folium.Icon(color=start_color, icon="play"),
        ).add_to(map)
    else:
        folium.Marker(
            start, popup="Origin", icon=folium.Icon(color=start_color, icon="play")
        ).add_to(map)

    # Build destination popup
    if end_station:
        end_location = (end_station["latitude"], end_station["longitude"]) if isinstance(end_station.get("latitude"), (int, float)) else end
        regular_end = get_regular_bikes_count(end_station.get("bikes_available", 0), end_station.get("ebikes_available", 0))
        end_popup = f"""
        <div style=\"font-family: Arial, sans-serif; min-width: 150px;\">
            <b>{end_station.get('name', end_station.get('station_id', 'Destination'))}</b><br>
            <hr style=\"margin: 5px 0;\">
            🚲 Bikes (total): {end_station.get('bikes_available', 0)}<br>
            ⚡ E-bikes: {end_station.get('ebikes_available', 0)}<br>
            🚴 Regular Bikes: {regular_end}
        </div>
        """
        folium.Marker(
            end_location,
            popup=folium.Popup(end_popup, max_width=300),
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
        # Draw stations for this route using the same route color
        for station in stations[idx]:
            folium.CircleMarker(
                location=[station["latitude"], station["longitude"]],
                radius=5,
                color=route_color,
                fill=True,
                fill_color=route_color,
                fill_opacity=0.9,
                opacity=0.9,
                weight=1,
                popup=f"Station: {station.get('station_name', station.get('name', station.get('station_id', '')))}",
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


# ---------------------------------------------------------------------------
# Trip Playback helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading CitiBike trip data (this may take a moment)…")
def load_playback_trips():
    """Load and return a sample of CitiBike trip data from 2024 to present."""
    return load_trips(sample_per_month=5000)


@st.cache_data(show_spinner=False)
def get_trip_route(start_lat: float, start_lng: float, end_lat: float, end_lng: float):
    """Return a list of {lat, lng} dicts for the Google Maps bicycling route."""
    client = get_gmaps_client()
    try:
        result = client.directions(
            f"{start_lat},{start_lng}",
            f"{end_lat},{end_lng}",
            mode="bicycling",
            alternatives=False,
        )
        if result:
            encoded = result[0]["overview_polyline"]["points"]
            decoded = polyline.decode(encoded)
            return [{"lat": lat, "lng": lng} for lat, lng in decoded]
    except Exception as exc:
        logger.warning("Failed to get Google Maps route (%s, %s) -> (%s, %s): %s",
                       start_lat, start_lng, end_lat, end_lng, exc)
    return [
        {"lat": start_lat, "lng": start_lng},
        {"lat": end_lat, "lng": end_lng},
    ]


@st.cache_data(show_spinner="Computing trip routes via Google Maps…")
def compute_trip_routes(trips_df):
    """
    Compute Google Maps bicycling routes for every row in trips_df.
    Returns a list of dicts: {path: [{lat, lng}…], duration: float (seconds)}.
    Routes are cached so repeated calls for the same (start, end) are free.
    """
    result = []
    for _, row in trips_df.iterrows():
        try:
            path = get_trip_route(
                # Round to 4 d.p. (~11 m) so nearby trips share cached routes
                round(float(row["start_lat"]), 4),
                round(float(row["start_lng"]), 4),
                round(float(row["end_lat"]), 4),
                round(float(row["end_lng"]), 4),
            )
            result.append({"path": path, "duration": float(row["duration_seconds"])})
        except Exception:
            continue
    return result


def render_trip_playback(trip_routes: list, playback_speed: int, gmaps_key: str, height: int = 700) -> None:
    """
    Render animated CitiBike trip playback using the Google Maps JavaScript API.

    Always maintains 10 active trips; when a trip ends a replacement begins
    within a random 0-10 simulated-second window.  The route is visualised with
    a spotlight effect: a bright window ±15% around the current position, with
    the far-past fading out and the far-future shown faintly.
    """
    trips_json = json.dumps(trip_routes)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%}}
#map{{width:100%;height:{height}px;background:#1a1a1a}}
#hud{{position:absolute;top:10px;left:50%;transform:translateX(-50%);
      background:rgba(0,0,0,.75);color:#ccc;padding:6px 14px;
      border-radius:20px;font:12px/1.5 monospace;white-space:nowrap;
      pointer-events:none}}
</style></head><body>
<div id="map"></div><div id="hud">Initialising…</div>
<script>(function(){{
  const trips={trips_json};
  const SPEED={playback_speed};
  const COLORS=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFD93D',
                '#DDA0DD','#98D8C8','#FF9F43','#A29BFE','#FD79A8'];
  const DARK=[
    {{elementType:'geometry',stylers:[{{color:'#212121'}}]}},
    {{elementType:'labels.text.fill',stylers:[{{color:'#757575'}}]}},
    {{elementType:'labels.text.stroke',stylers:[{{color:'#212121'}}]}},
    {{featureType:'water',elementType:'geometry',stylers:[{{color:'#000000'}}]}},
    {{featureType:'road',elementType:'geometry.fill',stylers:[{{color:'#2c2c2c'}}]}},
    {{featureType:'road.arterial',elementType:'geometry',stylers:[{{color:'#373737'}}]}},
    {{featureType:'road.highway',elementType:'geometry',stylers:[{{color:'#3c3c3c'}}]}},
    {{featureType:'poi',stylers:[{{visibility:'off'}}]}},
    {{featureType:'transit',stylers:[{{visibility:'off'}}]}}
  ];

  let map,active={{}},nextIdx=0,simTime=0,prevTs=null;

  window.initMap=function(){{
    map=new google.maps.Map(document.getElementById('map'),{{
      center:{{lat:40.7589,lng:-73.9851}},zoom:13,styles:DARK,
      disableDefaultUI:true,zoomControl:true
    }});
    const n=Math.min(10,trips.length);
    for(let i=0;i<n;i++) addTrip(i,0);
    nextIdx=n;
    requestAnimationFrame(frame);
  }};

  function addTrip(idx,t0){{
    if(idx>=trips.length) return;
    const trip=trips[idx],col=COLORS[idx%COLORS.length];
    const dimPast=new google.maps.Polyline({{
      path:[],strokeColor:col,strokeOpacity:.08,strokeWeight:3,map}});
    const bright=new google.maps.Polyline({{
      path:[trip.path[0]],strokeColor:col,strokeOpacity:.9,strokeWeight:5,map}});
    const dimFuture=new google.maps.Polyline({{
      path:trip.path,strokeColor:col,strokeOpacity:.2,strokeWeight:3,map}});
    const dot=new google.maps.Marker({{
      position:trip.path[0],map,
      icon:{{path:google.maps.SymbolPath.CIRCLE,scale:7,
             fillColor:col,fillOpacity:1,strokeColor:'#fff',strokeWeight:2}}
    }});
    active[idx]={{dot,dimPast,bright,dimFuture,t0}};
  }}

  function removeTrip(idx){{
    const a=active[idx]; if(!a) return;
    a.dot.setMap(null);a.dimPast.setMap(null);
    a.bright.setMap(null);a.dimFuture.setMap(null);
    delete active[idx];
    // Start the next trip within 0-10 simulated seconds of the current time
    // so we always keep ~10 trips active without jarring simultaneous starts.
    if(nextIdx<trips.length){{
      addTrip(nextIdx,simTime+Math.random()*10);
      nextIdx++;
    }}
  }}

  function lerp(p,q,t){{
    return {{lat:p.lat+t*(q.lat-p.lat),lng:p.lng+t*(q.lng-p.lng)}};
  }}
  function atProg(path,prog){{
    prog=Math.max(0,Math.min(1,prog));
    const f=prog*(path.length-1),i=Math.min(Math.floor(f),path.length-2);
    return lerp(path[i],path[i+1],f-i);
  }}

  function frame(ts){{
    if(prevTs!==null) simTime+=(ts-prevTs)/1000*SPEED;
    prevTs=ts;
    const ended=[];
    for(const [k,a] of Object.entries(active)){{
      const trip=trips[+k],prog=(simTime-a.t0)/trip.duration;
      if(prog>=1){{ended.push(+k);continue;}}
      if(prog<0) continue;
      const pos=atProg(trip.path,prog);
      a.dot.setPosition(pos);
      // SPOTLIGHT_WIN: fraction of total route shown bright around current pos (±15%)
      const WIN=0.15,n=trip.path.length;
      const lo=Math.max(0,prog-WIN),hi=Math.min(1,prog+WIN);
      const iLo=Math.floor(lo*(n-1)),iHi=Math.ceil(hi*(n-1));
      // far past (fades out)
      a.dimPast.setPath(iLo>1?trip.path.slice(0,iLo+1):[]);
      // spotlight window (bright)
      const bSeg=trip.path.slice(iLo,iHi+1).concat([pos]);
      a.bright.setPath(bSeg);
      // future (faint, shows upcoming route)
      a.dimFuture.setPath(iHi<n-1?trip.path.slice(iHi):[]);
    }}
    ended.forEach(removeTrip);
    document.getElementById('hud').textContent=
      'Active: '+Object.keys(active).length+'/10  ·  '+
      'Trips: '+nextIdx+'/'+trips.length+'  ·  '+
      SPEED+'× speed';
    requestAnimationFrame(frame);
  }}
}})();
</script>
<script async
  src="https://maps.googleapis.com/maps/api/js?key={gmaps_key}&callback=initMap&loading=async">
</script>
</body></html>"""

    components.html(html, height=height + 10, scrolling=False)


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
    if "playback_active" not in st.session_state:
        st.session_state["playback_active"] = False

def main():
    # Mode selection
    st.sidebar.title("Mode Selection")
    app_mode = st.sidebar.radio(
        "Choose Mode:",
        ["Route Finder", "General View", "Trip Playback"],
    )
    st.sidebar.divider()

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
            50,
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
        db_manager.update_metadata(in_type=LIVE)
        if st.sidebar.button(
            "🔄 Refresh Data", use_container_width=True, type="primary"
        ):
            st.cache_data.clear()
            st.rerun()
        

    elif app_mode == "Trip Playback":
        st.sidebar.subheader("Trip Playback")
        playback_speed = st.sidebar.slider(
            "Playback Speed (× real time)",
            min_value=1,
            max_value=200,
            value=30,
            step=1,
            key="playback_speed_slider",
        )
        n_trips = st.sidebar.slider(
            "Trips to preload",
            min_value=20,
            max_value=200,
            value=100,
            step=10,
            key="n_trips_slider",
        )
        st.sidebar.info(
            "Animates real CitiBike trips from 2024 using Google Maps routing.\n\n"
            "10 trips run simultaneously; when one ends a new one begins."
        )
        if st.sidebar.button(
            "▶ Start Playback", type="primary", use_container_width=True, key="start_playback"
        ):
            st.session_state["playback_active"] = True
        if st.session_state.get("playback_active"):
            if st.sidebar.button(
                "⏹ Stop Playback", use_container_width=True, key="stop_playback"
            ):
                st.session_state["playback_active"] = False

    # Map rendering section
    if app_mode == "Trip Playback":
        # Render Google Maps JS animation (no Folium map)
        if st.session_state.get("playback_active"):
            trips_df = load_playback_trips()
            if trips_df is not None and len(trips_df) > 0:
                routes = compute_trip_routes(trips_df.head(n_trips))
                if routes:
                    render_trip_playback(routes, playback_speed, GOOGLE_MAPS)
                else:
                    st.error("Could not compute routes. Check your Google Maps API key.")
            else:
                st.error(
                    "No trip data available. Check internet connectivity — "
                    "CitiBike trip files are downloaded from s3.amazonaws.com."
                )
        else:
            st.info(
                "Click **▶ Start Playback** in the sidebar to begin the animated "
                "CitiBike trip overlay."
            )
    else:
        db_manager = get_db_manager()

        # Create base map centered on Manhattan
        m = folium.Map(
            location=[40.7589, -73.9851],
            zoom_start=13,
            tiles="CartoDB dark_matter",
        )

        if app_mode == "Route Finder":
            if st.session_state["run"] and o_c and d_c:
                result = find_route_stations(o_c, d_c, bike_type_value, station_threshold, num_routes)

                if result[0] is not None:
                    route_dict, paths, stations_per_route, start_station, end_station, features = result

                    if route_dict and not st.session_state.get("route_written", False):
                        route_list = [
                            {"station_id": sid, "color": color}
                            for sid, color in route_dict.items()
                        ]
                        db_manager.clear_route()
                        db_manager.set_route_stations(route_list)
                        st.session_state["route_written"] = True

                    render_routes(
                        m, paths, o_c, d_c, stations_per_route,
                        st.session_state.get("selected_route"), bike_type_value,
                        start_station, end_station
                    )

                    st.subheader("Route Information")
                    for i, feature in enumerate(features, 1):
                        with st.expander(f"Route {i}", expanded=(i == 1)):
                            distance_km = feature["properties"]["distance"] / 1000
                            duration_min = feature["properties"]["duration"] / 60
                            st.write(f"Distance: {distance_km:.2f} km")
                            st.write(f"Duration: {duration_min:.0f} minutes")
                else:
                    st.error("Could not find a valid route. Please try different locations.")

        elif app_mode == "General View":
            station_list = db_manager.get_all_stations()
            gbfs_status = get_station_status()
            if station_list and gbfs_status:
                stations_added = general_view_render(m, station_list, gbfs_status)
                st.success(f"Displaying {stations_added} stations with live availability data")
            else:
                st.error("Could not load station data")

        st_folium(m, use_container_width=True, height=1200, returned_objects=[])

if __name__ == "__main__":
    init_session_states()
    st.set_page_config(page_title="CitiBike Station Finder", layout="wide")
    st.title("🚲 CitiBike Station Finder")
    main()
