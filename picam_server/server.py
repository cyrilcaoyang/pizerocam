import os
import sys
import json
import socket
import threading
from datetime import datetime
from time import sleep
from picamera2 import Picamera2
from libcamera import controls
from neopixel import NeoPixel
from neopixel import board
from sdl_utils import get_logger

"""
This is a module for the Raspberry Pi Camera Server
Please install the dependencies ONLY on Pi Zero 2 W/WH
Code will NOT work on Pi 5
"""

with open('server_settings.json', 'r') as file:
    data = json.load(file)
    buffer_size = data["BufferSize"]
    chunk_size = data["ChunkSize"]
    server_port = data["ServerPort"]


class CameraServer:
    """
    This is a class of a server with ability to take photos on demand
    """
    def __init__(self, host="0,0,0,0", port=server_port):
        self.host = host
        self.port = port
        self.logger = self.setup_logger()
        self.server_ip = self.get_server_ip()
        self.led = self.init_led()
        self.color = (200, 200, 200)
        self.camera = None
        self.camera_lock = threading.Lock()     # Add thread lock

    @staticmethod
    def setup_logger():
        # Create the logger and file handler
        logger = get_logger("WirelessCameraLogger")
        return logger

    def get_server_ip(self):
        # First, let us find out the IP address of this server
        s_test = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s_test.connect(("8.8.8.8", 80))
        server_ip = s_test.getsockname()[0]
        self.logger.info(f"My IP address is : {server_ip}")
        return server_ip

    def init_led(self):
        # NeoPixel LED RING with 12 pixels needs to use board.D10
        led = NeoPixel(board.D10, 12, auto_write=True)
        
        # Blink to show initialization
        led.fill((100, 100, 100))
        sleep(1)
        led.fill((0, 0, 0))
        self.logger.info("LED initialized!")
        return led

    def test_led(self, led):
        self.logger.info("Start testing LED")
        for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]:
            for i in range(0, 12):
                led.fill((0, 0, 0))
                led[i] = color
                sleep(0.1)
        led.fill((0, 0, 0))
        self.logger.info("LED test done")

    def init_camera(self):
        # Initialize the camera with connectivity checks
        try:
            # Check for available cameras
            cameras = Picamera2.global_camera_info()
            if not cameras:
                self.logger.error("No cameras detected! Check connection.")
                raise RuntimeError("Camera not found")

            self.logger.info(f"Found {len(cameras)} camera(s)")

            # Initialize first available camera
            self.camera = Picamera2(0)  # Explicitly use first camera
            config = self.camera.create_still_configuration()
            self.camera.configure(config)

            # Verify camera can start
            self.camera.start()
            self.logger.info("Camera initialized successfully")

            # Set autofocus if available
            if 'AfMode' in self.camera.camera_controls:
                self.camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                self.logger.info("Autofocus enabled")
            else:
                self.logger.warning("Autofocus not supported")

            return self.camera

        except Exception as e:
            self.logger.error(f"Camera initialization failed: {str(e)}")
            self.logger.error("Possible causes:")
            self.logger.error("1. Camera not properly connected")
            self.logger.error("2. Camera not enabled in raspi-config")
            self.logger.error("3. Hardware incompatibile")
            raise  # Re-raise exception for upstream handling

    def take_photo(self):
        # Early return :)
        if not self.init_camera():
            return None

        # Create a directory to store photos
        photo_dir = 'photos'
        os.makedirs(photo_dir, exist_ok=True)

        # Generate a timestamped image path
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        img_path = os.path.join(photo_dir, f"{timestamp}.jpg")
        self.camera = self.init_camera()

        # Take photo with camera connectivity check
        try:
            # Capture photo using existing camera instance
            with self.camera_lock:
                # Turn on the LED, take a photo, and turn off LED
                self.logger.info(f"The LED color will be {self.color}")
                self.led.fill(self.color)
                sleep(3)
                self.camera.capture_file(img_path)
                self.logger.info(f"Photo captured and saved as {img_path}")
                self.led.fill((0, 0, 0))
                self.camera.close()
                return img_path
        except Exception as e:
            self.logger.error(f"Failed to take photo: {str(e)}")
            return None

    @staticmethod
    def _recv_until_newline(self, conn):
        """
        Helper: Read bytes from 'conn' until we encounter a newline (b'\\n').
        Returns the line as a string (minus the newline).
        """
        data_chunks = []
        while True:
            chunk = conn.recv(1)
            if not chunk:
                # Connection closed or error
                return ""
            if chunk == b'\n':
                break
            data_chunks.append(chunk)
        return b''.join(data_chunks).decode('utf-8')

    @staticmethod
    def _send_ascii_length(self, conn, value):
        """
        Helper: Sends the integer 'value' as ASCII digits followed by a newline.
        """
        message = f"{value}\n"
        conn.sendall(message.encode("utf-8"))

    def send_photo(self, conn, img_path):
        """

        :param conn:
        :param img_path:
        :return: Boolean
        """

        # Read the entire file into memory:
        with open(img_path, 'rb') as f:
            image_data = f.read()
        file_size = len(image_data)

        # 1) Send size as ASCII plus newline
        self._send_ascii_length(conn, file_size)
        self.logger.info(f"Sent file size {file_size} (ASCII + delimiter) to client")

        # 2) Receive the echoed size back (as ASCII + newline)
        echoed_size_str = self._recv_until_newline(conn)
        if not echoed_size_str:
            self.logger.error("Failed to receive echoed size from client (connection closed).")
            return False

        # Try to parse it into an integer
        try:
            echoed_size = int(echoed_size_str)
        except ValueError:
            self.logger.error(f"Invalid size echoed: '{echoed_size_str}'")
            return False

        self.logger.info(f"Client echoed size: {echoed_size}")

        # 3) Confirm they match
        if echoed_size != file_size:
            self.logger.error("File size mismatch! Aborting transfer.")
            return False
        else:
            self.logger.info("File size confirmed. Proceeding with file transfer.")

        # 4) Send the file data in chunks
        offset = 0
        while offset < file_size:
            end = offset + chunk_size
            chunk = image_data[offset:end]
            conn.sendall(chunk)
            offset = end

        self.logger.info("File transfer complete.")
        self.logger.info("Waiting for new command...")

    def handle_client(self, conn):
        """
        Handles connection after connected with client
        :param conn:
        :return: None
        """
        try:
            while True:
                try:
                    msg = conn.recv(buffer_size).decode('utf-8').strip()
                    if not msg:
                        break
                    self.logger.info(f"Received message: {msg}")

                    if msg == "TAKE_PHOTO":
                        image_path = self.take_photo()
                        self.send_photo(conn, image_path)

                    elif msg == "CHANGE_COLOR":
                        # Request color coordinates from client
                        conn.sendall("PLEASE SEND RGB".encode('utf-8'))
                        self.logger.info("Sent color request to client")

                        # Receive and process RGB values
                        rgb_data = conn.recv(buffer_size).decode('utf-8').strip()
                        try:
                            r, g, b = map(int, rgb_data.split(','))
                            if all(0 <= val <= 255 for val in (r, g, b)):
                                self.color = (r, g, b)
                                self.led.fill(self.color)
                                sleep(1)
                                self.led.fill((0, 0, 0))
                                conn.sendall("COLOR_CHANGED".encode('utf-8'))
                                self.logger.info(f"LED color changed to ({r},{g},{b})")
                            else:
                                raise ValueError("Values out of range (0-255)")
                        except Exception as e:
                            conn.sendall(f"INVALID_RGB: {e}".encode('utf-8'))
                            self.logger.error(f"Invalid RGB values: {rgb_data}")

                    elif msg == "RESET_CAMERA":
                        self.shutdown()
                        success = self.initialize_camera()
                        conn.sendall(f"Camera reset {'successful' if success else 'failed'}".encode())
                    
                    else:
                        conn.sendall(f"Unknown command: {msg}".encode('utf-8'))

                except (ConnectionResetError, BrokenPipeError):
                    self.logger.error("Client disconnected unexpectedly")
                    break

        except Exception as e:
            self.logger.error(f"Handle client error: {e}")
        finally:
            conn.close()
            self.logger.info("Client connection closed")
            self.logger.info("Waiting for new connection")

    def start_server(self):
        # Make sure that camera can start
        if not self.init_camera():
            self.logger.error("Failed to initialize camera.")
            return

        # Initialize a socket object for IPv4 (AF_INET) using TCP (SOCK_STREAM)
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            server_socket.bind((self.server_ip, self.port))
            server_socket.listen(5)
            self.logger.info(f"Server started on {self.server_ip}:{self.port}")
            self.logger.info("Waiting for connection...")

            while True:
                try:
                    # Accept the connection from client
                    conn, addr = server_socket.accept()
                    self.logger.info(f"Connected with address: {addr}")
                    self.handle_client(conn)
                except KeyboardInterrupt:
                    self.logger.info("Server shutdown requested")
                    break
                except Exception as e:
                    self.logger.error(f"Connection error: {e}")
        finally:
            server_socket.close()
            self.logger.info("Server socket closed")

    def shutdown(self):
        with self.camera_lock:
            if self.camera is not None:
                self.camera.close()
                self.camera = None
        self.led.fill((0, 0, 0))


if __name__ == "__main__":
    camera = CameraServer()
    camera.test_led(camera.led)
    try:
        camera.start_server()
    except Exception as e:
        camera.logger.error(f"Critical failure: {e}")
    finally:
        camera.shutdown()
