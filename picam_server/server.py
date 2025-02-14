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
        self.logger = self._setup_logger()
        self.server_ip = self._get_server_ip()
        self.led = self._init_led()
        self.color = (200, 200, 200)
        self._camera_initialized = False
        self.camera_lock = threading.Lock()     # Add thread lock

    @staticmethod
    def _setup_logger():
        # Create the logger and file handler
        return get_logger("WirelessCameraLogger")

    def _get_server_ip(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_test:
            s_test.connect(("8.8.8.8", 80))
            server_ip = s_test.getsockname()[0]
            self.logger.info(f"My IP address is : {server_ip}")
            return server_ip

    def _init_led(self):
        # NeoPixel LED RING with 12 pixels needs to use board.D10
        led = NeoPixel(board.D10, 12, auto_write=True)
        # Blink to show initialization
        for i in range(0, 3):
            led.fill((100, 100, 100))
            sleep(0.5)
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
        self.logger.info("LED test complete")

    def _single_camera_session(self):
        """Context manager for camera operations"""
        class CameraContext:
            def __init__(self, outer):
                self.outer = outer
                self.picam = None

            def __enter__(self):
                self.outer.logger.debug("Initializing camera session")
                self.picam = Picamera2(0)
                config = self.picam.create_still_configuration()
                self.picam.configure(config)
                self.picam.start()

                # Set autofocus if available
                if 'AfMode' in self.picam.camera_controls:
                    self.picam.set_controls({"AfMode": controls.AfModeEnum.Continuous})

                return self.picam

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.outer.logger.debug("Closing camera session")
                try:
                    self.picam.stop()
                    self.picam.close()
                except Exception as e:
                    self.outer.logger.error(f"Error closing camera: {e}")
                return False

        return CameraContext(self)

    def take_photo(self):
        # New camera instance every time
        try:
            with self.camera_lock:
                # Create output directory
                photo_dir = os.path.join(os.getcwd(), "photos")
                os.makedirs(photo_dir, exist_ok=True)

                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                img_path = os.path.join(photo_dir, filename)

                # Control LED
                self.led.fill(self.color)

                # Camera operations
                with self._single_camera_session() as cam:
                    # Wait for auto-exposure to settle
                    sleep(3)
                    cam.capture_file(img_path)

                self.led.fill((0, 0, 0))
                self.logger.info(f"Captured {filename}")
                return img_path

        except Exception as e:
            self.logger.error(f"Capture failed: {e}")
            self.led.fill((0, 0, 0))
            return None

    @staticmethod
    def _recv_until_newline(conn):
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
    def _send_ascii_length(conn, value):
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
        """Handle client connection in a thread-safe manner"""
        try:
            while True:
                msg = conn.recv(buffer_size).decode('utf-8').strip()
                if not msg:
                    break
                self.logger.info(f"Received message: {msg}")

                if msg == "TAKE_PHOTO":
                    image_path = self.take_photo()
                    if image_path:
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

        except Exception as e:
            self.logger.error(f"Handle client error: {e}")
        finally:
            conn.close()
            self.logger.info("Client connection closed")
            self.logger.info("Waiting for new connection")

    def start_server(self):
        """Start the server with clean error handling"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            server.bind((self.server_ip, self.port))
            server.listen(5)
            self.logger.info(f"Server started on {self.server_ip}:{self.port}")
            self.logger.info("Waiting for connection...")

            while True:
                # Accept the connection from client
                conn, addr = server.accept()
                self.logger.info(f"Connected with address: {addr}")
                threading.Thread(
                    target=self.handle_client,
                    args=(conn,),
                    daemon=True
                ).start()

        except KeyboardInterrupt:
            self.logger.info("Server shutdown requested")
        finally:
            server.close()
            self.led.fill((0, 0, 0))
            self.logger.info("Server socket closed")


if __name__ == "__main__":
    camera = CameraServer()
    camera.test_led(camera.led)
    camera.start_server()
