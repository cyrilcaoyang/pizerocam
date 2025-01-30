"""
Please read the instruction.
Please install the dependencies on Pi Zero 2 W/WH
Code will not work on Pi 5
"""
import socket
import time
import json
import logging
from picamera2 import Picamera2
from libcamera import controls
from neopixel import NeoPixel
from neopixel import board

with open('server_settings.json', 'r') as file:
    data = json.load(file)
    bufferSize = data["bufferSize"]
    chunkSize = data["chunkSize"]
    ServerPort = data["ServerPort"]


class WirelessCamera:
    def __init__(self, host="0, 0, 0, 0", port=2222):
        self.host = host
        self.port = port
        self.logger = self.setup_logger()
        self.server_ip = self.get_server_ip()
        self.server_socket = self.setup_socket()
        self.led = self.init_led()
        self.picam2 = self.init_camera()

    @staticmethod
    def setup_logger():
        logger = logging.getLogger('WirelessCameraLogger')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('server.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def get_server_ip(self):
        # First, let us find out the IP address of this server
        s_test = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s_test.connect(("8.8.8.8", 80))
        server_ip = s_test.getsockname()[0]
        self.logger.info(f"My IP address is : {server_ip}")
        return server_ip

    def setup_socket(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.server_ip, self.port))
        self.logger.info(f"Server started on {self.server_ip}:{self.port}")
        server_socket.listen(5)
        return server_socket

    def init_led(self):
        # NeoPixel LED RING with 12 pixels needs to use board.D10
        led = NeoPixel(board.D10, 12, auto_write=True)

        # Blink to show initialization
        led.fill((100, 100, 100))
        time.sleep(3)
        led.fill((0, 0, 0))
        self.logger.info("LED initialized!")
        return led

    def init_camera(self):
        # Initialize the camera
        picam2 = Picamera2()
        picam2.configure(picam2.create_still_configuration())
        picam2.start()
        # Important to set autofocus
        picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        self.logger.info("Camera initialized!")
        return picam2

    def run(self):
        self.logger.info("Running the server...")
        try:
            while True:
                # Accept the connection from client
                client_socket, addr = self.server_socket.accept()
                self.logger.debug(f"Accepted connection from {addr}")
                msg = "Hi, I am Wireless PiCamera. Happy to be your Server!"
                client_socket.send(msg.encode('utf-8'))

                # Get the message from the client
                message, address = self.server_socket.recvfrom(bufferSize)
                message = message.decode('utf-8')
                self.logger.info(f"Client Address: {address[0]}")
                self.logger.info(f"Message Received: {message}")

                if message == "TAKE_PHOTO":
                    # Define the image file path
                    image_path = 'image.jpg'
                    self.led.fill((255, 255, 255))
                    time.sleep(5)
                    self.picam2.capture_file(image_path)
                    self.logger.info(f"Photo captured and saved as {image_path}")
                    self.led.fill((0, 0, 0))

                    # Read the image file as binary data
                    with open(image_path, 'rb') as file:
                        image_data = file.read()
                        image_size = len(image_data)

                    # Send the size of the image
                    self.server_socket.sendto(str(image_size).encode(), address)
                    self.logger.info('Sending captured image...')
        except Exception as e:
            self.logger.error(f"Error in run method: {e}")


if __name__ == "__main__":
    camera = WirelessCamera()
    camera.run()
