import yaml
from time import sleep
from pathlib import Path
from .server import CameraServer
from .PCA9685 import PCA9685

# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent

# Open and read the JSON file
with open(script_dir / 'server_settings.yaml', 'r') as file:
    data = yaml.safe_load(file)
    buffer_size = data["BufferSize"]
    server_port = data["ServerPort"]


class PHTestServer(CameraServer):
    def __init__(self, host="0.0.0.0", port=server_port, init_camera=True, init_motor=True, resolution=None):
        super().__init__(host, port, init_camera, resolution)
        self.motor_driver = self._init_motor_driver() if init_motor else None
        self.PWMA = 0
        self.AIN1 = 1
        self.AIN2 = 2

    def _init_motor_driver(self):
        # Initialize PCA9685 and motor
        # Assumes default I2C address 0x40
        try:
            self.logger.info("Attempting to initialize motor driver...")
            pwm = PCA9685(0x40, debug=False)
            pwm.setPWMFreq(50)
            self.logger.info("Motor driver initialized successfully.")
            return pwm
        except Exception as e:
            self.logger.error(f"Failed to initialize motor driver: {e}")
            self.logger.warning("Motor functionality will be disabled")
            raise

    def run_motor(self):
        """
        Run motor A for 1 seconds at 50% speed
        """
        try:
            if self.motor_driver is None:
                self.logger.error("Motor driver not initialized - cannot run motor")
                return False
                
            self.logger.info("Running motor...")
            # Set speed to 50%
            self.motor_driver.setDutycycle(self.PWMA, 50)
            
            # Set direction to forward
            self.motor_driver.setLevel(self.AIN1, 0)
            self.motor_driver.setLevel(self.AIN2, 1)
            sleep(1)
            
            # Stop motor
            self.motor_driver.setDutycycle(self.PWMA, 0)
            self.logger.info("Motor run complete.")
            return True
        except Exception as e:
            self.logger.error(f"Motor run failed: {e}")
            return False

    def handle_client(self, conn):
        """Handle client connection in a thread-safe manner"""
        try:
            while True:
                msg = conn.recv(buffer_size).decode('utf-8').strip()
                if not msg:
                    break
                self.logger.info(f"Received message: {msg}.")

                if msg == "TAKE_PHOTO":
                    image_path = self.take_photo()
                    if image_path:
                        self.send_photo(conn, image_path)

                elif msg == "CHANGE_COLOR":
                    # Request color coordinates from client
                    conn.sendall("PLEASE SEND RGB".encode('utf-8'))
                    self.logger.info("Sent color request to client.")

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
                            self.logger.info(f"LED color changed to ({r},{g},{b}).")
                        else:
                            raise ValueError("Values out of range (0-255).")
                    except Exception as e:
                        conn.sendall(f"INVALID_RGB: {e}".encode('utf-8'))
                        self.logger.error(f"Invalid RGB values: {rgb_data}.")

                elif msg == "RUN_MOTOR":
                    if self.run_motor():
                        conn.sendall("MOTOR_RUN_COMPLETE".encode('utf-8'))
                    else:
                        conn.sendall("MOTOR_RUN_FAILED".encode('utf-8'))

        except Exception as e:
            self.logger.error(f"Handle client error: {e}.")
        finally:
            conn.close()
            self.logger.info("Client connection closed.")
            self.logger.info("Waiting for new connection.")


if __name__ == "__main__":
    # Test PH test server
    ph_test_server = PHTestServer()
    ph_test_server.test_led(ph_test_server.led)
    ph_test_server.start_server() 