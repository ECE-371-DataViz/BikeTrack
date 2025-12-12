#!/usr/bin/env python3
"""
LED Grid Image Display Driver

Loads LED color values from CSV and displays them on the LED strip.
Can be used standalone or integrated with the main driver.py

Usage:
    python display_led_image.py --input led_output.csv
    python display_led_image.py --input logo_colors.csv --duration 10
"""

import argparse
import csv
import time
from pathlib import Path

# Try to import LED libraries (only available on Raspberry Pi)
try:
    import board
    import neopixel
    PI_AVAILABLE = True
except (ImportError, NotImplementedError):
    PI_AVAILABLE = False
    print("Warning: Running in simulation mode (not on Raspberry Pi)")


def load_led_colors(csv_path):
    """Load LED colors from CSV file
    
    Returns:
        List of tuples: [(index, r, g, b), ...]
    """
    colors = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = int(row['index'])
            r = int(row['r'])
            g = int(row['g'])
            b = int(row['b'])
            colors.append((index, r, g, b))
    return colors
    


def display_on_leds(colors, duration=None, brightness=0.1):
    """Display colors on LED strip
    
    Args:
        colors: List of (index, r, g, b) tuples
        duration: How long to display (seconds), None for indefinite
        brightness: LED brightness (0.0 to 1.0)
    """
    if not PI_AVAILABLE:
        print("Simulation mode - would display:")
        for idx, r, g, b in sorted(colors)[:10]:
            print(f"  LED {idx}: RGB({r}, {g}, {b})")
        if len(colors) > 10:
            print(f"  ... and {len(colors) - 10} more LEDs")
        return
    
    # Initialize LED strip
    NUM_LEDS = 665
    leds = neopixel.NeoPixel(board.D18, NUM_LEDS, brightness=brightness, auto_write=False)
    
    # Clear all LEDs first
    leds.fill((0, 0, 0))
    leds.show()
    
    print(f"Setting {len(colors)} LED colors...")
    
    # Set each LED color
    for index, r, g, b in colors:
        if 0 <= index < NUM_LEDS:
            leds[index] = (r, g, b)
    
    # Update the strip
    leds.show()
    print("✓ LEDs updated!")
    
    if duration:
        print(f"Displaying for {duration} seconds...")
        time.sleep(duration)
        # Clear LEDs after duration
        leds.fill((0, 0, 0))
        leds.show()
        print("✓ LEDs cleared")
    else:
        print("Press Ctrl+C to clear LEDs and exit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nClearing LEDs...")
            leds.fill((0, 0, 0))
            leds.show()
            print("✓ LEDs cleared")


def main():
    parser = argparse.ArgumentParser(
        description='Display image colors on LED strip'
    )
    parser.add_argument('--input', required=True,
                        help='Input CSV file with LED colors')
    parser.add_argument('--duration', type=float, default=None,
                        help='Display duration in seconds (default: indefinite)')
    parser.add_argument('--brightness', type=float, default=0.1,
                        help='LED brightness 0.0-1.0 (default: 0.1)')
    parser.add_argument('--filter-bounds', action='store_true',
                        help='Only display LEDs that are in_bounds')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    print(f"Loading LED colors from {args.input}...")
    
    # Load and optionally filter colors
    all_colors = []
    with open(args.input, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.filter_bounds and row.get('in_bounds') == 'False':
                continue
            index = int(row['index'])
            r = int(row['r'])
            g = int(row['g'])
            b = int(row['b'])
            all_colors.append((index, r, g, b))
    
    print(f"✓ Loaded {len(all_colors)} LED colors")
    
    display_on_leds(all_colors, duration=args.duration, brightness=args.brightness)
    
    return 0


if __name__ == '__main__':
    exit(main())
