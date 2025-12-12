#!/usr/bin/env python3
"""
Batch Image to LED Mapper

Process multiple images at once with the same center and size settings.

Usage:
    python batch_image_to_led.py --images *.png --center-lat 40.7589 --center-lon -73.9851 --size 0.05
"""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Batch process images for LED grid')
    parser.add_argument('--images', nargs='+', required=True, help='Image files to process')
    parser.add_argument('--center-lat', type=float, required=True, help='Center latitude')
    parser.add_argument('--center-lon', type=float, required=True, help='Center longitude')
    parser.add_argument('--size', type=float, required=True, help='Size in degrees')
    parser.add_argument('--output-dir', default='led_frames', help='Output directory')
    parser.add_argument('--preview', action='store_true', help='Generate preview maps')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Processing {len(args.images)} images...")
    print(f"Center: ({args.center_lat}, {args.center_lon})")
    print(f"Size: {args.size}°")
    print(f"Output: {output_dir}/")
    print()
    
    for i, img_path in enumerate(args.images, 1):
        img = Path(img_path)
        if not img.exists():
            print(f"⚠️  Skipping {img} (not found)")
            continue
        
        output_csv = output_dir / f"{img.stem}.csv"
        
        cmd = [
            'python', 'image_to_led_grid.py',
            '--image', str(img),
            '--center-lat', str(args.center_lat),
            '--center-lon', str(args.center_lon),
            '--size', str(args.size),
            '--output', str(output_csv)
        ]
        
        if args.preview:
            cmd.append('--preview')
        
        print(f"[{i}/{len(args.images)}] Processing {img.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✓ {output_csv.name}")
        else:
            print(f"  ✗ Failed: {result.stderr}")
    
    print(f"\n✓ Done! Output files in {output_dir}/")


if __name__ == '__main__':
    main()
