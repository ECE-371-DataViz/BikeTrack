#!/usr/bin/env python3
"""
Generate simple test images for LED grid mapping

Creates basic geometric patterns and test images for verifying the LED mapper.
"""

import argparse
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Error: PIL/Pillow is required. Install with: pip install Pillow")
    exit(1)


def create_gradient(width, height, direction='horizontal'):
    """Create a color gradient image"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    for i in range(width if direction == 'horizontal' else height):
        # Rainbow gradient
        hue = int(255 * i / (width if direction == 'horizontal' else height))
        r = int(255 * abs((hue / 255 * 6) % 2 - 1))
        g = int(255 * abs(((hue / 255 * 6 + 4) % 6) / 2 - 1))
        b = int(255 * abs(((hue / 255 * 6 + 2) % 6) / 2 - 1))
        
        if direction == 'horizontal':
            draw.line([(i, 0), (i, height)], fill=(r, g, b))
        else:
            draw.line([(0, i), (width, i)], fill=(r, g, b))
    
    return img


def create_bullseye(width, height, num_rings=5):
    """Create a bullseye/target pattern"""
    img = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(img)
    
    center_x, center_y = width // 2, height // 2
    max_radius = min(width, height) // 2
    
    colors = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)]
    
    for i in range(num_rings, 0, -1):
        radius = int(max_radius * i / num_rings)
        color = colors[(num_rings - i) % len(colors)]
        draw.ellipse(
            [center_x - radius, center_y - radius, 
             center_x + radius, center_y + radius],
            fill=color
        )
    
    return img


def create_checkerboard(width, height, squares=8):
    """Create a checkerboard pattern"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    square_width = width // squares
    square_height = height // squares
    
    for row in range(squares):
        for col in range(squares):
            if (row + col) % 2 == 0:
                color = (255, 255, 255)  # White
            else:
                color = (255, 0, 0)  # Red
            
            x1 = col * square_width
            y1 = row * square_height
            x2 = x1 + square_width
            y2 = y1 + square_height
            draw.rectangle([x1, y1, x2, y2], fill=color)
    
    return img


def create_cross(width, height, thickness=0.2):
    """Create a cross/plus sign"""
    img = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(img)
    
    bar_width = int(width * thickness)
    bar_height = int(height * thickness)
    
    # Horizontal bar
    draw.rectangle(
        [0, (height - bar_height) // 2, width, (height + bar_height) // 2],
        fill=(0, 255, 0)
    )
    
    # Vertical bar
    draw.rectangle(
        [(width - bar_width) // 2, 0, (width + bar_width) // 2, height],
        fill=(0, 255, 0)
    )
    
    return img


def create_nyc_test(width, height):
    """Create NYC-themed test pattern"""
    img = Image.new('RGB', (width, height), color=(0, 0, 100))  # Dark blue background
    draw = ImageDraw.Draw(img)
    
    # Yellow taxi cab colors
    # Vertical stripes
    stripe_width = width // 3
    draw.rectangle([stripe_width, 0, 2*stripe_width, height], fill=(255, 204, 0))
    
    # Add some stars (white dots)
    import random
    random.seed(42)
    for _ in range(50):
        x = random.randint(0, width)
        y = random.randint(0, height)
        draw.ellipse([x-2, y-2, x+2, y+2], fill=(255, 255, 255))
    
    return img


def main():
    parser = argparse.ArgumentParser(description='Generate test images for LED mapper')
    parser.add_argument('--pattern', 
                        choices=['gradient', 'gradient-v', 'bullseye', 'checkerboard', 'cross', 'nyc', 'all'],
                        default='all',
                        help='Pattern to generate')
    parser.add_argument('--width', type=int, default=512, help='Image width (default: 512)')
    parser.add_argument('--height', type=int, default=512, help='Image height (default: 512)')
    parser.add_argument('--output-dir', default='test_images', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    patterns = {
        'gradient': lambda: create_gradient(args.width, args.height, 'horizontal'),
        'gradient-v': lambda: create_gradient(args.width, args.height, 'vertical'),
        'bullseye': lambda: create_bullseye(args.width, args.height),
        'checkerboard': lambda: create_checkerboard(args.width, args.height),
        'cross': lambda: create_cross(args.width, args.height),
        'nyc': lambda: create_nyc_test(args.width, args.height),
    }
    
    if args.pattern == 'all':
        for name, func in patterns.items():
            img = func()
            output_path = output_dir / f"{name}.png"
            img.save(output_path)
            print(f"✓ Created: {output_path}")
    else:
        img = patterns[args.pattern]()
        output_path = output_dir / f"{args.pattern}.png"
        img.save(output_path)
        print(f"✓ Created: {output_path}")
    
    print(f"\nTest images saved to: {output_dir}/")
    print("\nExample usage:")
    print(f"  python image_to_led_grid.py --image {output_dir}/bullseye.png \\")
    print("    --center-lat 40.7589 --center-lon -73.9851 --size 0.05 --preview")


if __name__ == '__main__':
    main()
