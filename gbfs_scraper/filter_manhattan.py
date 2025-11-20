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
    with open(file_path, "r") as f:
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

    # Coordinates generated using streamlit_polygon_editor.py
    coords = [[40.70094304347228,-74.0153217315674],[40.70434287835176,-74.01860475540163],[40.70473734556466,-74.0189427137375],[40.706058997033665,-74.01936113834383],[40.707006503329524,-74.01912510395051],[40.71369152943539,-74.0176284313202],[40.71865606887142,-74.01669502258302],[40.748963847316034,-74.01248931884767],[40.78749525996358,-73.98922920227052],[40.827644243840076,-73.95738601684572],[40.85030664256397,-73.94691467285158],[40.87802345041109,-73.9277744293213],[40.87778008631153,-73.92251729965211],[40.874648721748194,-73.91871929168703],[40.87458382143405,-73.91535043716432],[40.87369143566273,-73.9118528366089],[40.872003982418164,-73.91007184982301],[40.869440269224285,-73.91080141067506],[40.867639120210725,-73.91148805618288],[40.85873001417227,-73.91938447952272],[40.856214475856085,-73.92165899276735],[40.83545358131836,-73.93474817276002],[40.80718299114835,-73.93266677856447],[40.80101089850059,-73.92786026000978],[40.79691751000055,-73.92837524414064],[40.794838231871346,-73.92906188964845],[40.791589229420794,-73.93378257751466],[40.78671542763972,-73.93773078918458],[40.78372598556362,-73.94150733947755],[40.78242618616746,-73.94348144531251],[40.779826511063824,-73.94262313842775],[40.77658896091275,-73.94191503524782],[40.77535808175501,-73.94219398498537],[40.771360613935286,-73.94506931304933],[40.751418432997454,-73.96455287933351],[40.74340411957345,-73.97086143493654],[40.73586035408156,-73.97232055664064],[40.72813686316017,-73.97051811218263],[40.71011731976856,-73.97601127624513],[40.70831184400125,-73.9982843399048],[40.70447301212353,-74.000301361084],[40.69925119494625,-74.00939941406251],[40.69899090674369,-74.0142059326172]]
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


def filter_manhattan(
    stations: List[Tuple[str, float, float]], poly: Polygon
) -> List[Tuple[str, float, float]]:
    """Return subset of stations that fall inside Manhattan polygon."""
    results = []
    for station in stations:
        station_id, lat, lon = station
        if is_point_in_manhattan(lat, lon, poly):
            results.append(station)
    return results


def filter_non_manhattan(
    stations: List[Tuple[str, float, float]], poly: Polygon
) -> List[Tuple[str, float, float]]:
    """Return subset of stations that fall outside the Manhattan polygon."""
    results = []
    for station in stations:
        station_id, lat, lon = station
        if not is_point_in_manhattan(lat, lon, poly):
            results.append(station)
    return results


def build_map(
    points: List[Tuple[str, float, float]],
    out_html: str,
    color: str = "red",
    poly_map_coords: List[Tuple[float, float]] = None,
) -> None:
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
            location=[lat, lon], radius=3, popup=station_id, color=color
        ).add_to(m)

    # If polygon coordinates for folium are provided, draw the polygon.
    if poly_map_coords:
        folium.Polygon(
            locations=poly_map_coords, color="orange", fill=True, fill_opacity=0.15
        ).add_to(m)

    m.save(out_html)


def main() -> None:
    """Load station_data, filter Manhattan points, save results and map.

    Output files:
     - manhattan_stations.txt: station_id, lat, lon for Manhattan
     - manhattan_map.html: a small visualization you can open in a browser
    """
    base_path = os.path.dirname(__file__)
    output_path = os.path.join(base_path, "..", "processed_data")
    input_file = os.path.join(base_path, "..", "raw_data", "station_data.txt")
    # If a local path is desired, user can provide absolute path by editing above
    stations = load_station_data(input_file)
    coords, poly = manhattan_polygon()
    manhattan = filter_manhattan(stations, poly)
    outside = filter_non_manhattan(stations, poly)

    # Save plain text
    out_station_file = os.path.join(output_path, "manhattan_stations.txt")
    with open(out_station_file, "w") as f:
        for sid, lat, lon in manhattan:
            f.write(f"{sid}\t{lat}\t{lon}\n")

    out_outside_file = os.path.join(output_path, "outside_manhattan_stations.txt")
    with open(out_outside_file, "w") as f:
        for sid, lat, lon in outside:
            f.write(f"{sid}\t{lat}\t{lon}\n")

    # Build a small HTML map
    out_html = os.path.join(output_path, "manhattan_map.html")
    out_html_outside = os.path.join(output_path, "outside_manhattan_map.html")
    # draw the polygon on both the Manhattan points map and the outside map
    build_map(manhattan, out_html, color="red", poly_map_coords=coords)
    build_map(outside, out_html_outside, color="blue", poly_map_coords=coords)
    # Console summary
    print(f"Total stations read: {len(stations)}")
    print(f"Stations on Manhattan (approx): {len(manhattan)}")
    print(f"Stations outside Manhattan (approx): {len(outside)}")
    print(
        f"Saved {out_station_file}, {out_outside_file}, {out_html} and {out_html_outside}"
    )


if __name__ == "__main__":
    main()
