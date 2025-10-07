#!/usr/bin/env python3
"""
Command-line interface for the ImageReqClient.
"""

import argparse
import sys
from .image_req_client import ImageReqClient


def main_client():
    """Main entry point for the client CLI."""
    parser = argparse.ArgumentParser(
        description="PiZeroCam Image Request Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive session
  pizerocam-client --ip 192.168.1.100
  
  # Request photo and analyze pH
  pizerocam-client --ip 192.168.1.100 --photo --analyze
  
  # Change LED color
  pizerocam-client --ip 192.168.1.100 --led 255,0,0
  
  # Run motor
  pizerocam-client --ip 192.168.1.100 --motor
        """
    )
    
    parser.add_argument(
        "--ip", 
        required=True,
        help="IP address of the image server"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=2222,
        help="Port of the image server (default: 2222)"
    )
    
    parser.add_argument(
        "--photo", 
        action="store_true",
        help="Request a photo from the server"
    )
    
    parser.add_argument(
        "--analyze", 
        action="store_true",
        help="Analyze the photo for pH (requires --photo)"
    )
    
    parser.add_argument(
        "--led", 
        metavar="R,G,B",
        help="Change LED color (e.g., 255,0,0 for red)"
    )
    
    parser.add_argument(
        "--motor", 
        action="store_true",
        help="Run the motor"
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Start interactive session (default if no other actions specified)"
    )
    
    args = parser.parse_args()
    
    # If no specific actions are specified, default to interactive mode
    if not any([args.photo, args.led, args.motor]):
        args.interactive = True
    
    # Create client
    client = ImageReqClient(port=args.port)
    
    try:
        # Connect to server
        print(f"Connecting to server at {args.ip}:{args.port}...")
        if not client.connect(args.ip):
            print("Failed to connect to server")
            sys.exit(1)
        
        print("Connected successfully!")
        
        # Handle specific actions
        if args.photo:
            print("Requesting photo...")
            image_path = client.request_photo()
            
            if image_path:
                print(f"Photo saved to: {image_path}")
                
                if args.analyze:
                    print("Analyzing photo for pH...")
                    ph_value = client.analyze_photo(image_path)
                    print(f"pH value: {ph_value}")
            else:
                print("Failed to receive photo")
                
        elif args.led:
            try:
                r, g, b = map(int, args.led.split(','))
                print(f"Changing LED color to RGB({r},{g},{b})...")
                if client.change_led_color(r, g, b):
                    print("LED color changed successfully!")
                else:
                    print("Failed to change LED color")
            except ValueError:
                print("Invalid LED color format. Use R,G,B (e.g., 255,0,0)")
                
        elif args.motor:
            print("Running motor...")
            if client.run_motor():
                print("Motor run complete!")
            else:
                print("Failed to run motor")
                
        elif args.interactive:
            # Start interactive session
            client.interactive_session()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main_client() 