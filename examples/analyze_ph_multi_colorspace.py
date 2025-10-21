#!/usr/bin/env python3
"""
Example: Analyze pH using multiple color spaces with ImageReqClient

This example demonstrates how to:
1. Connect to the image server
2. Request a photo
3. Analyze the pH using RGB, LAB, and HSV color spaces
4. Print and compare results from all three color spaces
"""

from image_req_client import ImageReqClient

def main():
    # Initialize the client
    client = ImageReqClient(port=2222)
    
    # Connect to the server (replace with your server IP)
    server_ip = "172.31.34.163"  # Change this to your server's IP
    
    if not client.connect(server_ip):
        print("Failed to connect to server")
        return
    
    try:
        # Request and analyze a photo with all color spaces
        print("\n" + "="*60)
        print("Requesting photo from server...")
        print("="*60)
        
        image_path, ph_results = client.request_and_analyze_photo(
            return_all_color_spaces=True,
            output_dir="same",  # Save annotated images in same folder
            interpolate=True    # Enable pH interpolation
        )
        
        if image_path and ph_results != "NULL":
            print(f"\n[OK] Photo saved to: {image_path}")
            print("\n" + "="*60)
            print("pH ESTIMATES FROM DIFFERENT COLOR SPACES (with interpolation)")
            print("="*60)
            print(f"  RGB Color Space: pH {ph_results['rgb']:>5.1f}  (min distance: {ph_results['distances']['rgb']:>7.2f})")
            print(f"  LAB Color Space: pH {ph_results['lab']:>5.1f}  (min distance: {ph_results['distances']['lab']:>7.2f})")
            print(f"  HSV Color Space: pH {ph_results['hsv']:>5.1f}  (min distance: {ph_results['distances']['hsv']:>7.2f})")
            print("="*60)
            
            # Determine which color space gave the most confident result (lowest distance)
            min_distance_space = min(ph_results['distances'], key=ph_results['distances'].get)
            print(f"\n[OK] Most confident result: {min_distance_space.upper()} with pH {ph_results[min_distance_space]:.1f}")
            print(f"     (smallest distance: {ph_results['distances'][min_distance_space]:.2f})")
            
        else:
            print("\n[ERROR] Failed to get photo or analyze pH")
            
    finally:
        # Always disconnect when done
        client.disconnect()


def analyze_existing_image(image_path):
    """
    Analyze an existing image file with all three color spaces.
    
    Args:
        image_path: Path to the image file
    """
    import os
    from pathlib import Path
    from image_req_client import ph_from_image
    
    # Convert to absolute path if relative
    image_path = Path(image_path).expanduser()
    
    # Check if file exists
    if not image_path.exists():
        # Try looking in common locations
        possible_paths = [
            Path.home() / "Pictures" / "pH_photos" / image_path.name,
            Path(__file__).parent.parent / "photos" / image_path.name,
            Path.cwd() / image_path.name,
        ]
        
        for path in possible_paths:
            if path.exists():
                image_path = path
                break
        else:
            print(f"\n[ERROR] Image file not found: {image_path}")
            print(f"\nSearched in:")
            for path in possible_paths:
                print(f"  - {path}")
            print(f"\nPlease provide the full path to the image file.")
            return
    
    print(f"\nAnalyzing image: {image_path}")
    print("="*60)
    
    # Analyze with interpolation enabled and save to same folder
    results = ph_from_image(str(image_path), 
                           return_all_color_spaces=True, 
                           output_dir="same",  # Save in same folder as input
                           interpolate=True)   # Enable pH interpolation to 1 decimal
    
    if results != "NULL":
        print("\nPH ESTIMATES FROM DIFFERENT COLOR SPACES (with interpolation)")
        print("="*60)
        print(f"  RGB Color Space: pH {results['rgb']:>5.1f}  (min distance: {results['distances']['rgb']:>7.2f})")
        print(f"  LAB Color Space: pH {results['lab']:>5.1f}  (min distance: {results['distances']['lab']:>7.2f})")
        print(f"  HSV Color Space: pH {results['hsv']:>5.1f}  (min distance: {results['distances']['hsv']:>7.2f})")
        print("="*60)
        
        # Check if all color spaces agree (within tolerance)
        tolerance = 0.2  # Allow 0.2 pH difference
        max_diff = max(abs(results['rgb'] - results['lab']), 
                      abs(results['rgb'] - results['hsv']),
                      abs(results['lab'] - results['hsv']))
        
        if max_diff <= tolerance:
            avg_ph = (results['rgb'] + results['lab'] + results['hsv']) / 3
            print(f"\n[OK] All color spaces agree (within {tolerance}): pH {avg_ph:.1f}")
        else:
            print(f"\n[WARNING] Color spaces disagree:")
            print(f"  RGB: {results['rgb']:.1f}, LAB: {results['lab']:.1f}, HSV: {results['hsv']:.1f}")
            min_distance_space = min(results['distances'], key=results['distances'].get)
            print(f"  Recommendation: Use {min_distance_space.upper()} result (pH {results[min_distance_space]:.1f})")
        
        # Show where annotated images with ROI highlighted are saved
        output_folder = Path(image_path).parent
        input_filename = Path(image_path).stem
        
        print(f"\n[INFO] Annotated images with ROI highlighted saved to:")
        print(f"  {output_folder}/")
        print(f"\nGenerated files (in same folder as input):")
        print(f"  1. {input_filename}_step1_text_boxes.png")
        print(f"     (shows detected pH numbers from color chart)")
        print(f"  2. {input_filename}_step2_color_boxes.png")
        print(f"     (shows color sampling regions below each number)")
        print(f"  3. {input_filename}_step3_highlighted_box.png")
        print(f"     (shows pH strip ROI highlighted with detected pH value)")
        print(f"\n[INFO] All files saved in: {output_folder}")
        
    else:
        print("[ERROR] Failed to analyze image")


if __name__ == "__main__":
    # Example 1: Request photo from server and analyze
    # Uncomment to use:
    # main()
    
    # Example 2: Analyze an existing image
    # Uncomment and replace with your image path:
    # analyze_existing_image("/Users/macbook_m2/PycharmProjects/pizerocam/examples/capture_20250930-175507_010010010.jpg")
    analyze_existing_image("/Users/macbook_m2/PycharmProjects/pizerocam/examples/capture_20251001-172245_010010010.jpg")
    
    print("Please uncomment the example you want to run in the __main__ section")

