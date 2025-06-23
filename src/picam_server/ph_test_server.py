import yaml
from time import sleep
from pathlib import Path
from server import CameraServer
from PCA9685 import PCA9685

# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent

# Open and read the JSON file
with open(script_dir / 'server_settings.yaml', 'r') as file:
    data = yaml.safe_load(file)
    buffer_size = data["BufferSize"]
    server_port = data["ServerPort"]


class PHTestServer(CameraServer):
    def __init__(self, host="0.0.0.0", port=server_port):
        super().__init__(host, port)
        self.motor_driver = self._init_motor_driver()

    def _init_motor_driver(self):
        # Initialize PCA9685 and motor
        # Assumes default I2C address 0x40
        pwm = PCA9685(0x40, debug=False)
        pwm.setPWMFreq(50)
        self.logger.info("Motor driver initialized.")
        return pwm

    def run_motor(self):
        """
        Run motor A for 2 seconds
        """
        try:
            self.logger.info("Running motor...")
            # Motor A: pins IN1, IN2 are connected to PWM channels 1, 2
            # Set IN1 high, IN2 low to run forward
            self.motor_driver.setMotorPwm(0, 4095) # Full speed
            self.motor_driver.setMotorPwm(1, 0)
            self.motor_driver.setMotorPwm(2, 0) # Make sure motor B is off
            sleep(2)
            # Stop motor
            self.motor_driver.setMotorPwm(0, 0)
            self.logger.info("Motor run complete.")
        except Exception as e:
            self.logger.error(f"Motor run failed: {e}")

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
                    self.run_motor()
                    conn.sendall("MOTOR_RUN_COMPLETE".encode('utf-8'))

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