#!/usr/bin/env python3
"""
Quick helper to load and print LED color data from cu_logo_leds_lower.csv

Usage:
    python print_led_csv.py [--csv cu_logo_leds_lower.csv]
"""
import csv
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Print LED color data from CSV')
    parser.add_argument('--csv', default='cu_logo_leds_lower.csv', help='CSV file to load (default: cu_logo_leds_lower.csv)')
    parser.add_argument('--show-all', action='store_true', help='Show all rows (default: only in-bounds)')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return 1

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    in_bounds = [row for row in rows if row.get('in_bounds', 'True') == 'True']
    print(f"Loaded {len(rows)} rows from {csv_path}")
    print(f"  - {len(in_bounds)} LEDs in bounds (with color)")
    print(f"  - {len(rows) - len(in_bounds)} LEDs out of bounds (black)")

    if args.show_all:
        to_show = rows
    else:
        to_show = in_bounds

    print("\nindex,station_id,latitude,longitude,r,g,b")
    for row in to_show:
        print(f"{row['index']},{row['station_id']},{row['latitude']},{row['longitude']},{row['r']},{row['g']},{row['b']}")

    return 0

if __name__ == '__main__':
    exit(main())
