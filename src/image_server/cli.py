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
  # Start server on default port (2222)
  pizerocam-server
  
  # Start server on specific port
  pizerocam-server --port 3333
  
  # Start server on specific interface
  pizerocam-server --host 192.168.1.100
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
    
    args = parser.parse_args()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start server
    server = ImageServer(host=args.host, port=args.port)
    
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