#!/usr/bin/env python3
"""
Command-line interface for the ImageServer.
"""

import argparse
import sys
import signal
from .image_server import ImageServer


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\nShutting down server...")
    sys.exit(0)


def main_server():
    """Main entry point for the server CLI."""
    parser = argparse.ArgumentParser(
        description="PiZeroCam Image Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with automatic Tailscale detection
  pizerocam-server
  
  # Start server with motor disabled (if PCA9685 not available)
  pizerocam-server --no-motor
  
  # Start server with no hardware (testing mode)
  pizerocam-server --no-camera --no-motor
  
  # Start server on specific port
  pizerocam-server --port 3333
  
  # Start server on specific interface
  pizerocam-server --host 192.168.1.100
  
  # Force specific Tailscale IP
  PIZEROCAM_SERVER_IP=100.x.x.x pizerocam-server
        """
    )
    
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host interface to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=2222,
        help="Port to listen on (default: 2222)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-camera",
        action="store_true", 
        help="Disable camera initialization"
    )
    
    parser.add_argument(
        "--no-motor",
        action="store_true",
        help="Disable motor initialization"
    )
    
    args = parser.parse_args()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Determine hardware initialization based on arguments
    init_camera = not args.no_camera
    init_motor = not args.no_motor
    
    # Create and start server with fallback for hardware issues
    if init_camera and init_motor:
        try:
            server = ImageServer(host=args.host, port=args.port, init_camera=True, init_motor=True)
            print("✓ Server created with full hardware support")
        except Exception as e:
            print(f"⚠ Full hardware initialization failed: {e}")
            print("Trying with motor disabled...")
            try:
                server = ImageServer(host=args.host, port=args.port, init_camera=True, init_motor=False)
                print("✓ Server created with camera only (motor disabled)")
            except Exception as e2:
                print(f"⚠ Camera initialization failed: {e2}")
                print("Trying with minimal configuration...")
                server = ImageServer(host=args.host, port=args.port, init_camera=False, init_motor=False)
                print("✓ Server created with minimal configuration (no hardware)")
    else:
        # User explicitly disabled some hardware
        try:
            server = ImageServer(host=args.host, port=args.port, init_camera=init_camera, init_motor=init_motor)
            hw_status = []
            if init_camera: hw_status.append("camera")
            if init_motor: hw_status.append("motor") 
            if not hw_status: hw_status.append("no hardware")
            print(f"✓ Server created with: {', '.join(hw_status)}")
        except Exception as e:
            print(f"⚠ Server initialization failed: {e}")
            print("Falling back to minimal configuration...")
            server = ImageServer(host=args.host, port=args.port, init_camera=False, init_motor=False)
            print("✓ Server created with minimal configuration (no hardware)")
    
    try:
        print(f"Starting PiZeroCam Image Server...")
        print(f"Host: {args.host}")
        print(f"Port: {args.port}")
        print(f"Server IP: {server.get_ip_address()}")
        print("Press Ctrl+C to stop the server")
        print("-" * 40)
        
        # Start server in blocking mode
        if server.start(background=False):
            print("Server started successfully")
        else:
            print("Failed to start server")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)
    finally:
        print("Stopping server...")
        server.stop()
        print("Server stopped")


if __name__ == "__main__":
    main_server() 