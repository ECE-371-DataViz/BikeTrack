#!/usr/bin/env python3
"""
Image to LED Grid Mapper

Maps an input color image onto the LED grid using latitude/longitude coordinates.
Samples the image at each LED position to determine the RGB color value.

Usage:
    python image_to_led_grid.py --image path/to/image.png --center-lat 40.7589 --center-lon -73.9851 --size 0.05
    python image_to_led_grid.py --image logo.jpg --center-lat 40.75 --center-lon -73.99 --size 0.08 --output led_values.csv

Arguments:
    --image: Path to input image file (PNG, JPG, etc.)
    --center-lat: Latitude of the center point where image should be rendered
    --center-lon: Longitude of the center point where image should be rendered
    --size: Size of the image in degrees (controls how large the image appears on the map)
    --output: Output CSV file path (default: led_output.csv)
    --data-src: Path to data_src.csv with LED coordinates (default: data_src.csv)
    --preview: Generate a preview HTML map (default: False)
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: PIL/Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


def load_led_positions(data_src_path):
    """Load LED positions from data_src.csv
    
    Returns:
        List of dicts with keys: index, station_id, latitude, longitude
    """
    leds = []
    with open(data_src_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            leds.append({
                'index': int(row['index']),
                'station_id': row['station_id'],
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude'])
            })
    return leds


def lat_lon_to_image_coords(lat, lon, center_lat, center_lon, size, img_width, img_height):
    """Convert lat/lon to image pixel coordinates
    
    Args:
        lat, lon: LED position
        center_lat, center_lon: Center of the image in lat/lon
        size: Size of the image region in degrees
        img_width, img_height: Image dimensions in pixels
    
    Returns:
        (x, y) pixel coordinates in the image, or None if outside bounds
    """
    # Calculate relative position from center
    rel_lon = lon - center_lon
    rel_lat = lat - center_lat
    
    # Map to image coordinates
    # Image spans from -size/2 to +size/2 in both directions
    half_size = size / 2.0
    
    # Normalize to [0, 1] range
    norm_x = (rel_lon + half_size) / size
    norm_y = (half_size - rel_lat) / size  # Invert Y (image Y increases downward)
    
    # Check bounds
    if norm_x < 0 or norm_x > 1 or norm_y < 0 or norm_y > 1:
        return None
    
    # Convert to pixel coordinates
    x = int(norm_x * img_width)
    y = int(norm_y * img_height)
    
    # Clamp to image bounds
    x = max(0, min(img_width - 1, x))
    y = max(0, min(img_height - 1, y))
    
    return (x, y)


def sample_image_at_leds(image_path, leds, center_lat, center_lon, size):
    """Sample the image at each LED position
    
    Returns:
        List of dicts with keys: index, station_id, lat, lon, r, g, b, in_bounds
    """
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size
    
    results = []
    in_bounds_count = 0
    
    for led in leds:
        coords = lat_lon_to_image_coords(
            led['latitude'], 
            led['longitude'],
            center_lat, 
            center_lon, 
            size,
            img_width, 
            img_height
        )
        
        if coords is None:
            # LED is outside the image bounds
            results.append({
                'index': led['index'],
                'station_id': led['station_id'],
                'latitude': led['latitude'],
                'longitude': led['longitude'],
                'r': 0,
                'g': 0,
                'b': 0,
                'in_bounds': False
            })
        else:
            # Sample the pixel color
            x, y = coords
            r, g, b = img.getpixel((x, y))
            results.append({
                'index': led['index'],
                'station_id': led['station_id'],
                'latitude': led['latitude'],
                'longitude': led['longitude'],
                'r': r,
                'g': g,
                'b': b,
                'in_bounds': True
            })
            in_bounds_count += 1
    
    return results, in_bounds_count


def write_led_output(results, output_path):
    """Write LED color values to CSV"""
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['index', 'station_id', 'latitude', 'longitude', 'r', 'g', 'b', 'in_bounds']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def generate_preview_map(results, center_lat, center_lon, size, output_path):
    """Generate an HTML preview map showing the LED colors"""
    try:
        import folium
    except ImportError:
        print("Warning: folium not installed, skipping preview map generation")
        print("Install with: pip install folium")
        return
    
    # Create map centered on the specified location
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB dark_matter"
    )
    
    # Draw bounding box of the image region
    half_size = size / 2.0
    bounds = [
        [center_lat - half_size, center_lon - half_size],  # SW
        [center_lat - half_size, center_lon + half_size],  # SE
        [center_lat + half_size, center_lon + half_size],  # NE
        [center_lat + half_size, center_lon - half_size],  # NW
        [center_lat - half_size, center_lon - half_size],  # SW (close)
    ]
    folium.PolyLine(
        bounds,
        color='yellow',
        weight=2,
        opacity=0.8,
        popup='Image Bounds'
    ).add_to(m)
    
    # Add LEDs as colored markers
    for led in results:
        if led['in_bounds']:
            color_hex = f"#{led['r']:02x}{led['g']:02x}{led['b']:02x}"
            folium.CircleMarker(
                location=[led['latitude'], led['longitude']],
                radius=5,
                color=color_hex,
                fill=True,
                fill_color=color_hex,
                fill_opacity=0.9,
                popup=f"LED {led['index']}<br>RGB: ({led['r']}, {led['g']}, {led['b']})"
            ).add_to(m)
        else:
            # Out of bounds LEDs shown in gray
            folium.CircleMarker(
                location=[led['latitude'], led['longitude']],
                radius=3,
                color='#555555',
                fill=True,
                fill_color='#555555',
                fill_opacity=0.3,
                popup=f"LED {led['index']}<br>(out of bounds)"
            ).add_to(m)
    
    m.save(output_path)
    print(f"✓ Preview map saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Map an input image onto LED grid using lat/lon coordinates'
    )
    parser.add_argument('--image', required=True, help='Path to input image file')
    parser.add_argument('--center-lat', type=float, required=True, 
                        help='Latitude of center point')
    parser.add_argument('--center-lon', type=float, required=True,
                        help='Longitude of center point')
    parser.add_argument('--size', type=float, required=True,
                        help='Size of image region in degrees (e.g., 0.05 for ~5.5km)')
    parser.add_argument('--output', default='led_output.csv',
                        help='Output CSV file path (default: led_output.csv)')
    parser.add_argument('--data-src', default='data_src.csv',
                        help='Path to data_src.csv (default: data_src.csv)')
    parser.add_argument('--preview', action='store_true',
                        help='Generate preview HTML map')
    
    args = parser.parse_args()
    
    # Validate image file
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Validate data source
    if not Path(args.data_src).exists():
        print(f"Error: Data source file not found: {args.data_src}")
        sys.exit(1)
    
    print(f"Loading LED positions from {args.data_src}...")
    leds = load_led_positions(args.data_src)
    print(f"✓ Loaded {len(leds)} LED positions")
    
    print(f"Loading image from {args.image}...")
    print(f"Mapping to region centered at ({args.center_lat}, {args.center_lon})")
    print(f"Image size: {args.size}° (~{args.size * 111:.1f} km)")
    
    results, in_bounds = sample_image_at_leds(
        args.image, 
        leds, 
        args.center_lat, 
        args.center_lon, 
        args.size
    )
    
    print(f"✓ Sampled colors for {len(results)} LEDs")
    print(f"  - {in_bounds} LEDs within image bounds")
    print(f"  - {len(results) - in_bounds} LEDs outside bounds (set to black)")
    
    print(f"Writing output to {args.output}...")
    write_led_output(results, args.output)
    print(f"✓ Output saved to: {args.output}")
    
    if args.preview:
        preview_path = args.output.replace('.csv', '_preview.html')
        print(f"Generating preview map...")
        generate_preview_map(results, args.center_lat, args.center_lon, args.size, preview_path)
    
    print("\nDone! Use the output CSV to set LED colors in your driver.")


if __name__ == '__main__':
    main()
