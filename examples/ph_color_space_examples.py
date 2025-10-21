#!/usr/bin/env python3
"""
Example usage of multi-color-space pH detection with ImageReqClient.

This script demonstrates how to use the RGB, LAB, and HSV color space analysis.
The client always returns all three color spaces in a dictionary.
"""

import sys
from pathlib import Path

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_req_client import ImageReqClient


def example_1_basic_usage():
    """Example 1: Basic usage - always returns all color spaces"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    client = ImageReqClient()
    client.connect("172.31.34.163")
    
    # Request and analyze photo - returns dict with RGB, LAB, and HSV
    image_path, ph_result = client.request_and_analyze_photo()
    
    if ph_result != "NULL":
        print(f"\n>>> RGB: {ph_result['rgb']:.1f}")
        print(f">>> LAB: {ph_result['lab']:.1f}")
        print(f">>> HSV: {ph_result['hsv']:.1f}")
        print(f"\n>>> Distances: {ph_result['distances']}")
    
    client.disconnect()


def example_2_analyze_existing_image():
    """Example 2: Analyze an existing image file"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Analyze Existing Image")
    print("="*70)
    
    client = ImageReqClient()
    # No need to connect to server for analyzing existing images
    
    # Analyze an existing photo
    image_path = "/path/to/your/photo.jpg"
    
    if Path(image_path).exists():
        ph_result = client.analyze_photo(image_path, output_dir="same")
        
        if ph_result != "NULL":
            print(f"\n>>> RGB: {ph_result['rgb']:.1f}")
            print(f">>> LAB: {ph_result['lab']:.1f}")
            print(f">>> HSV: {ph_result['hsv']:.1f}")
    else:
        print(f"Image not found: {image_path}")


def example_3_use_specific_color_space():
    """Example 3: Use a specific color space value"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Using a Specific Color Space")
    print("="*70)
    
    client = ImageReqClient()
    client.connect("172.31.34.163")
    
    image_path, ph_result = client.request_and_analyze_photo()
    
    if ph_result != "NULL":
        # You can choose which color space to use
        lab_ph = ph_result['lab']
        print(f"\n>>> Using LAB pH: {lab_ph:.1f}")
        
        # Or compare them
        if abs(ph_result['rgb'] - ph_result['lab']) > 0.5:
            print(">>> Warning: RGB and LAB differ by more than 0.5 pH units")
    
    client.disconnect()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Multi-Color-Space pH Detection Examples")
    print("="*70)
    print("\nThe client always returns pH values from all three color spaces:")
    print("  - RGB: Standard red-green-blue color space")
    print("  - LAB: Perceptually uniform color space (best for color matching)")
    print("  - HSV: Hue-saturation-value color space")
    print("\nNote: Update the IP address '172.31.34.163' to your server's IP")
    print("="*70)
    
    # Run examples
    # NOTE: Uncomment the examples you want to run
    
    # example_1_basic_usage()
    # example_2_analyze_existing_image()
    # example_3_use_specific_color_space()
    
    print("\n" + "="*70)
    print("Examples Complete!")
    print("="*70)
