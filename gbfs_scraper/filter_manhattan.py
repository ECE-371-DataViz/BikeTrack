import os
from typing import List, Tuple

from shapely.geometry import Point, Polygon
import folium


def load_station_data(file_path: str) -> List[Tuple[str, float, float]]:
    """Load station data from a file.

    File format for each line must be: station_id \t lat \t lon
    (Note: function assumes the file is clean; it does not catch exceptions.)
    """
    stations = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            station_id = parts[0]
            lat = float(parts[1])
            lon = float(parts[2])
            stations.append((station_id, lat, lon))
    return stations

def manhattan_polygon():
    """Return an approximate polygon that covers Manhattan island.

    This polygon is conservative and not perfect; it is meant for a quick
    filter and visualization. Coordinates are in (lon, lat) order.
    """

    ##Coordinates generated using streamlit_polygon_editor.py
    coords = [
        [40.70029233758598, -74.01369094848634],
        [40.70705936809059, -74.02210235595705],
        [40.755319574776024, -74.01369094848634],
        [40.76182096906601, -74.00733947753908],
        [40.81393915021812, -73.96905899047853],
        [40.866146238852586, -73.93970489501955],
        [40.871857950788446, -73.93627166748048],
        [40.878477727669704, -73.93026351928712],
        [40.87545997030689, -73.91710996627809],
        [40.874015940971326, -73.90966415405275],
        [40.85783741473627, -73.91897678375246],
        [40.834284674577425, -73.93421173095705],
        [40.819281875389116, -73.93245220184328],
        [40.799581489929515, -73.92571449279787],
        [40.75154847474455, -73.96236419677736],
        [40.72254287000301, -73.96802902221681],
        [40.709336579498036, -73.97618293762208]]
    # We expect coords as lat, lon so convert to lon,lat for shapely geometry
    poly_coords = [(lon, lat) for lat, lon in coords]
    poly = Polygon(poly_coords)
    return coords, poly


def is_point_in_manhattan(lat: float, lon: float, poly: Polygon) -> bool:
    """Check if a lat/lon point is in Manhattan via shapely polygon.

    Returns True if inside polygon.
    """
    point = Point(lon, lat)
    inside = poly.contains(point)
    return inside


def filter_manhattan(stations: List[Tuple[str, float, float]], poly: Polygon) -> List[Tuple[str, float, float]]:
    """Return subset of stations that fall inside Manhattan polygon."""
    results = []
    for station in stations:
        station_id, lat, lon = station
        if is_point_in_manhattan(lat, lon, poly):
            results.append(station)
    return results


def filter_non_manhattan(stations: List[Tuple[str, float, float]], poly: Polygon) -> List[Tuple[str, float, float]]:
    """Return subset of stations that fall outside the Manhattan polygon."""
    results = []
    for station in stations:
        station_id, lat, lon = station
        if not is_point_in_manhattan(lat, lon, poly):
            results.append(station)
    return results


def build_map(points: List[Tuple[str, float, float]], out_html: str, color: str = 'red', poly_map_coords: List[Tuple[float, float]] = None) -> None:
    """Create a folium map with the given points and save to HTML.

    This map uses a small tiled basemap and clustering for readability.
    """
    if not points:
        m = folium.Map(location=[40.78, -73.96], zoom_start=12)
        m.save(out_html)
        return

    # Center the map on average of points
    avg_lat = sum([p[1] for p in points]) / len(points)
    avg_lon = sum([p[2] for p in points]) / len(points)

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)

    for station_id, lat, lon in points:
        folium.CircleMarker(
            location=[lat, lon], radius=3, popup=station_id, color=color).add_to(m)

    # If polygon coordinates for folium are provided, draw the polygon.
    if poly_map_coords:
        folium.Polygon(locations=poly_map_coords, color='orange',
                       fill=True, fill_opacity=0.15).add_to(m)

    m.save(out_html)


def main() -> None:
    """Load station_data, filter Manhattan points, save results and map.

    Output files:
     - manhattan_stations.txt: station_id, lat, lon for Manhattan
     - manhattan_map.html: a small visualization you can open in a browser
    """
    base_path = os.path.dirname(__file__)
    input_file = os.path.join(base_path, '..', 'station_data.txt')
    # If a local path is desired, user can provide absolute path by editing above
    stations = load_station_data(input_file)
    coords, poly = manhattan_polygon()
    manhattan = filter_manhattan(stations, poly)
    outside = filter_non_manhattan(stations, poly)

    # Save plain text
    out_station_file = os.path.join(base_path, 'manhattan_stations.txt')
    with open(out_station_file, 'w') as f:
        for sid, lat, lon in manhattan:
            f.write(f"{sid}\t{lat}\t{lon}\n")

    out_outside_file = os.path.join(
        base_path, 'outside_manhattan_stations.txt')
    with open(out_outside_file, 'w') as f:
        for sid, lat, lon in outside:
            f.write(f"{sid}\t{lat}\t{lon}\n")

    # Build a small HTML map
    out_html = os.path.join(base_path, 'manhattan_map.html')
    out_html_outside = os.path.join(base_path, 'outside_manhattan_map.html')
    # draw the polygon on both the Manhattan points map and the outside map
    build_map(manhattan, out_html, color='red',
              poly_map_coords=coords)
    build_map(outside, out_html_outside, color='blue',
              poly_map_coords=coords)
    # Console summary
    print(f"Total stations read: {len(stations)}")
    print(f"Stations on Manhattan (approx): {len(manhattan)}")
    print(f"Stations outside Manhattan (approx): {len(outside)}")
    print(
        f"Saved {out_station_file}, {out_outside_file}, {out_html} and {out_html_outside}")


if __name__ == '__main__':
    main()
