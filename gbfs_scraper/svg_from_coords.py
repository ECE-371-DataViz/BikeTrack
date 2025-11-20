"""Generate an SVG of a Manhattan polygon and the points contained within.

This script reads station data (default: raw_data/station_data.txt) and the
Manhattan polygon from `filter_manhattan.py` then generates a single SVG
visualization showing the polygon and the points that are inside.

Usage:
    python gbfs_scraper/svg_from_coords.py --out-file processed_data/manhattan.svg

Options:
    --stations FILE  Input station data (tab-separated station_id lat lon)
    --out-file FILE  Output SVG file path
    --width INT      SVG width in px (default 1200)
    --height INT     SVG height in px (default 900)
    --no-labels      Don't label the points
    --dedupe         Remove duplicate coordinates (default: True)
    --rotate-180     Rotate final image 180 degrees (default: enabled)

This script avoids duplicate consecutive polygon points to avoid streamlit/shapely
issues and deduplicates station points to make rendering cleaner.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple
import math
import xml.etree.ElementTree as ET

from gbfs_scraper.filter_manhattan import (
    load_station_data,
    manhattan_polygon,
    filter_manhattan,
)


def unique_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return points


def project_points(
    lat_lon_list: List[Tuple[float, float]],
    bounds: Tuple[float, float, float, float],
    width: int,
    height: int,
    padding: int = 20,
) -> List[Tuple[float, float]]:
    """Project lat/lon into SVG X,Y coordinates using a linear mapping.

    bounds: (min_lat, max_lat, min_lon, max_lon)
    Returns list of (x, y) pairs with (0,0) at top-left as expected by SVG.
    """
    min_lat, max_lat, min_lon, max_lon = bounds
    lon_range = max_lon - min_lon if max_lon != min_lon else 1.0
    lat_range = max_lat - min_lat if max_lat != min_lat else 1.0

    # compute scales to maintain aspect ratio
    drawable_w = width - 2 * padding
    drawable_h = height - 2 * padding
    x_scale = drawable_w / lon_range
    y_scale = drawable_h / lat_range
    scale = min(x_scale, y_scale)

    # compute margins if the scale left extra space
    x_span_pix = scale * lon_range
    y_span_pix = scale * lat_range
    x_offset = padding + (drawable_w - x_span_pix) / 2
    y_offset = padding + (drawable_h - y_span_pix) / 2

    projected = []
    for lat, lon in lat_lon_list:
        # X increases eastward (lon), Y increases downward (SVG coords), so invert lat
        x = (lon - min_lon) * scale + x_offset
        y = (max_lat - lat) * scale + y_offset  # so higher lat is closer to top
        projected.append((x, y))
    return projected


def bounds_from_points(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return min(lats), max(lats), min(lons), max(lons)


def generate_svg(
    poly_coords_latlon: List[Tuple[float, float]],
    inside_points: List[Tuple[str, float, float]],
    out_file: str,
    width: int = 1200,
    height: int = 900,
    dedupe: bool = True,
    labels: bool = False,
    rotate: bool = True,
    rotate_180: bool = True,
):
    # Deduplicate polygon coords (lat, lon tuples)
    if dedupe:
        poly_coords_clean = unique_points(poly_coords_latlon)
    else:
        poly_coords_clean = list(poly_coords_latlon)

    if len(poly_coords_clean) < 3:
        raise ValueError("Polygon needs at least 3 unique coordinates")

    # Build a list of all points (for bounds and projection)
    all_coords_for_bounds = poly_coords_clean.copy()
    point_coords = []  # (lat, lon)
    for sid, lat, lon in inside_points:
        point_coords.append((lat, lon))
        all_coords_for_bounds.append((lat, lon))

    bounds = bounds_from_points(all_coords_for_bounds)
    # First pass: project into XY (initial fit)
    poly_xy = project_points(poly_coords_clean, bounds, width, height, padding=20)
    points_xy = project_points(point_coords, bounds, width, height, padding=20)

    # Optionally rotate the geometry to align Manhattan major axis vertically
    def pca_angle(points_xy_list: List[Tuple[float, float]]) -> float:
        # Compute centroid
        xs = [p[0] for p in points_xy_list]
        ys = [p[1] for p in points_xy_list]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        # centralize
        dx = [x - cx for x in xs]
        dy = [y - cy for y in ys]
        cov_xx = sum(d * d for d in dx) / len(dx)
        cov_yy = sum(d * d for d in dy) / len(dy)
        cov_xy = sum(d1 * d2 for d1, d2 in zip(dx, dy)) / len(dx)
        # angle of principal axis: tan(2θ) = 2*cov_xy/(cov_xx-cov_yy)
        theta = 0.5 * math.atan2(2 * cov_xy, (cov_xx - cov_yy))
        return theta, (cx, cy)

    if rotate and len(poly_xy) >= 3:
        theta, centroid = pca_angle(poly_xy)
        # We want the principal axis to be vertical: rotate by (pi/2 - theta)
        rot_angle = math.pi / 2 - theta
    else:
        rot_angle = 0.0
        centroid = (sum([p[0] for p in poly_xy]) / len(poly_xy), sum([p[1] for p in poly_xy]) / len(poly_xy))

    def rotate_about(center: Tuple[float, float], angle: float, pts: List[Tuple[float, float]]):
        cx, cy = center
        c = math.cos(angle)
        s = math.sin(angle)
        out = []
        for x, y in pts:
            dx = x - cx
            dy = y - cy
            rx = c * dx - s * dy + cx
            ry = s * dx + c * dy + cy
            out.append((rx, ry))
        return out

    if rot_angle != 0.0:
        # rotate polygon and points
        poly_xy = rotate_about(centroid, rot_angle, poly_xy)
        points_xy = rotate_about(centroid, rot_angle, points_xy)

    # After rotation, re-fit the geometry into the SVG canvas (uniform scale)
    # to ensure everything is centered and visible
    def fit_to_canvas(pts: List[Tuple[float, float]], w: int, h: int, padding: int = 20):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max_x - min_x if max_x != min_x else 1.0
        span_y = max_y - min_y if max_y != min_y else 1.0
        drawable_w = w - 2 * padding
        drawable_h = h - 2 * padding
        scale_x = drawable_w / span_x
        scale_y = drawable_h / span_y
        scale = min(scale_x, scale_y)
        # center offsets
        x_offset = padding + (drawable_w - span_x * scale) / 2
        y_offset = padding + (drawable_h - span_y * scale) / 2
        out = [((x - min_x) * scale + x_offset, (y - min_y) * scale + y_offset) for x, y in pts]
        return out

    # Fit both polygon and points together to maintain consistent mapping
    all_rotated = poly_xy + points_xy
    all_fitted = fit_to_canvas(all_rotated, width, height, padding=20)
    poly_c = all_fitted[: len(poly_xy)]
    points_c = all_fitted[len(poly_xy) :]

    # Optionally rotate the final canvas by 180 degrees (flip about center)
    if rotate_180:
        cx = width / 2.0
        cy = height / 2.0
        def rotate_180_about_center(cx: float, cy: float, pts: List[Tuple[float, float]]):
            return [(2 * cx - x, 2 * cy - y) for x, y in pts]

        poly_c = rotate_180_about_center(cx, cy, poly_c)
        points_c = rotate_180_about_center(cx, cy, points_c)

    # Build SVG document (basic namespacing)
    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", width=str(width), height=str(height))

    # Background
    # No background fill to keep SVG transparent by default

    # Draw polygon
    poly_points_str = " ".join([f"{x:.2f},{y:.2f}" for x, y in poly_c])
    # Draw an unfilled polygon with a clear outline
    ET.SubElement(
        svg,
        "polygon",
        points=poly_points_str,
        fill="none",
        stroke="#ff7f00",
        **{"stroke-width": "2", "stroke-linejoin": "round", "stroke-linecap": "round"},
    )

    # Draw points
    # Shrink point marker radius for better legibility in dense areas
    for (sid, lat, lon), (x, y) in zip(inside_points, points_c):
        ET.SubElement(svg, "circle", cx=f"{x:.2f}", cy=f"{y:.2f}", r="2", fill="#d62728", stroke="#8b0000", **{"stroke-width": "0.5"})
        # labels are optional; default is disabled to avoid cluttering the output
        if labels:
            # Optional label slightly offset from the point
            label = sid
            ET.SubElement(svg, "text", x=f"{x+6:.2f}", y=f"{y-6:.2f}", fill="#333333").text = label

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    tree = ET.ElementTree(svg)
    tree.write(out_file, encoding="utf-8", xml_declaration=True)
    print(f"SVG saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate SVG for Manhattan polygon and points inside")
    parser.add_argument("--stations", default=None, help="Station input file (tab-separated station_id lat lon)")
    parser.add_argument("--out-file", default="processed_data/manhattan_polygon.svg")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--labels", action="store_true", help="Enable point labels (default: disabled)")
    parser.add_argument("--dedupe", dest="dedupe", action="store_true")
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    parser.add_argument("--rotate-180", dest="rotate_180", action="store_true", help="Rotate final image 180 degrees (default: enabled)")
    parser.add_argument("--no-rotate-180", dest="rotate_180", action="store_false")
    parser.set_defaults(dedupe=True)
    parser.set_defaults(rotate_180=True)

    args = parser.parse_args()

    base_path = os.path.dirname(__file__)
    if args.stations is None:
        raw_path = os.path.join(base_path, "..", "raw_data", "station_data.txt")
        station_file = os.path.normpath(raw_path)
    else:
        station_file = args.stations

    # load data and polygon
    stations = load_station_data(station_file)
    poly_coords, poly = manhattan_polygon()

    # filter points inside
    inside = filter_manhattan(stations, poly)
    # remove duplicates for points for rendering
    if args.dedupe:
        seen = set()
        unique_inside = []
        for sid, lat, lon in inside:
            key = (round(lat, 6), round(lon, 6))
            if key in seen:
                continue
            seen.add(key)
            unique_inside.append((sid, lat, lon))
        inside = unique_inside

    # write SVG
    generate_svg(
        poly_coords_latlon=poly_coords,
        inside_points=inside,
        out_file=args.out_file,
        width=args.width,
        height=args.height,
        dedupe=args.dedupe,
        labels=args.labels,
        rotate_180=args.rotate_180,
    )


if __name__ == "__main__":
    main()
