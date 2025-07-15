#!/usr/bin/env python3
"""
Test script to verify server initialization with different configurations
"""

from src.image_server import ImageServer

def test_server_without_camera():
    """Test server initialization without camera hardware"""
    print("Testing server initialization without camera...")
    try:
        server = ImageServer(init_camera=False)
        print(f"✓ Server created successfully without camera")
        print(f"✓ Server IP: {server.get_ip_address()}")
        print(f"✓ Server running status: {server.is_running()}")
        return True
    except Exception as e:
        print(f"✗ Failed to create server without camera: {e}")
        return False

def test_server_with_camera():
    """Test server initialization with camera hardware"""
    print("\nTesting server initialization with camera...")
    try:
        server = ImageServer(init_camera=True)
        print(f"✓ Server created successfully with camera")
        print(f"✓ Server IP: {server.get_ip_address()}")
        print(f"✓ Server running status: {server.is_running()}")
        return True
    except Exception as e:
        print(f"✗ Failed to create server with camera: {e}")
        print(f"  This is expected if camera hardware is not available or has memory issues")
        return False

if __name__ == "__main__":
    print("PiZeroCam Server Test")
    print("=" * 50)
    
    # Test without camera first
    success_no_camera = test_server_without_camera()
    
    # Test with camera 
    success_with_camera = test_server_with_camera()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Server without camera: {'PASS' if success_no_camera else 'FAIL'}")
    print(f"Server with camera: {'PASS' if success_with_camera else 'FAIL'}")
    
    if success_no_camera:
        print("\n✓ Server can be initialized without camera hardware")
        print("  You can now use: server = ImageServer(init_camera=False)")
    else:
        print("\n✗ Server initialization failed completely") 