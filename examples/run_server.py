from image_server import server

"""
This script runs the camera server on Pi Zero 2 W/WH
"""

if __name__ == "__main__":
    camera = server.CameraServer()
    camera.test_led(camera.led)
    camera.start_server()
