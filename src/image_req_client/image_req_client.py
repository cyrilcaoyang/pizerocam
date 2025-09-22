import os
import socket
from pathlib import Path
from time import sleep
from dotenv import load_dotenv
from logger import get_logger
from socket_utils import connect_socket, send_file_name, receive_file_name
from socket_utils import send_file_size, receive_file_size, receive_file

from .ph_grid_color_reader import ph_from_image

# Load environment variables from .env file
load_dotenv()

# Get settings from environment variables
buffer_size = int(os.getenv("BUFFER_SIZE", 2048))
chunk_size = int(os.getenv("CHUNK_SIZE", 1024))
default_port = int(os.getenv("SERVER_PORT", 2222))


class ImageReqClient:
    """
    A client for requesting images from the image server and analyzing them for pH values.
    
    This class provides a clean interface for:
    - Connecting to an image server
    - Requesting photos
    - Analyzing photos for pH values
    - Managing the connection lifecycle
    """
    
    def __init__(self, port=default_port, logger=None):
        """
        Initialize the ImageReqClient.
        
        Args:
            port (int): Server port to connect to
            logger: Optional logger instance
        """
        self.port = port
        self.server_ip = None
        self.socket = None
        self.connected = False
        self.logger = logger or self._setup_logger()
        
    @staticmethod
    def _setup_logger():
        """Set up the logger for the client."""
        return get_logger("ImageReqClientLogger")
    
    def connect(self, ip):
        """
        Connect to the image server at the specified IP address.
        
        Args:
            ip (str): IP address of the server
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.server_ip = ip
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket = connect_socket(self.socket, ip, self.port, self.logger)
            
            if self.socket is None:
                self.logger.error(f"Failed to connect to server at {ip}:{self.port}")
                return False
                
            self.connected = True
            self.logger.info(f"Successfully connected to server at {ip}:{self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to server: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """
        Disconnect from the image server.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            if self.socket:
                self.socket.close()
                self.socket = None
            self.connected = False
            self.server_ip = None
            self.logger.info("Disconnected from server")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from server: {e}")
            return False
    
    def request_photo(self):
        """
        Request a photo from the connected server.
        
        Returns:
            str or None: Filepath of the received image, or None if failed
        """
        if not self.connected or not self.socket:
            self.logger.error("Not connected to server. Call connect() first.")
            return None
            
        try:
            # Send photo request
            self.socket.sendall("TAKE_PHOTO".encode('utf-8'))
            sleep(5)  # Wait for photo capture
            
            # Receive the photo
            success, image_path = self._receive_photo()
            
            if success:
                self.logger.info(f"Photo received and saved as {image_path}")
                return image_path
            else:
                self.logger.error("Failed to receive complete photo")
                return None
                
        except Exception as e:
            self.logger.error(f"Error requesting photo: {e}")
            return None
    
    def _receive_photo(self):
        """
        Internal method to receive photo from server.
        
        Returns:
            tuple: (success: bool, image_path: str)
        """
        try:
            # Create the photos directory in ~/Pictures/pH_photos/
            home_dir = Path.home()
            output_dir = home_dir / "Pictures" / "pH_photos"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Receive the image name and echo back to confirm
            img_name = receive_file_name(self.socket, self.logger)
            self.logger.info(f"Server sent file name {img_name}.")
            send_file_name(self.socket, img_name, self.logger)
            self.logger.info("Echoed the file name back to server.")
            img_path = output_dir / img_name

            # Receive ASCII-based file size and echo back to confirm
            file_size = receive_file_size(self.socket, self.logger)
            self.logger.info(f"Server sent file size: {file_size} bytes.")
            send_file_size(self.socket, file_size, self.logger)
            self.logger.info("Echoed the file size back to server.")

            # Now receive the actual file data in chunks, and write the file to disk
            received_data = receive_file(self.socket, file_size, chunk_size, self.logger)
            with open(img_path, "wb") as f:
                f.write(received_data)
            self.logger.info(f"File {img_name} saved to: {output_dir}")
            return True, str(img_path)
            
        except Exception as e:
            self.logger.error(f"Error receiving photo: {e}")
            return False, None
    
    def analyze_photo(self, filepath):
        """
        Analyze a photo for pH value using the pH grid color reader.
        
        Args:
            filepath (str): Path to the image file to analyze
            
        Returns:
            str: pH value as string, or "NULL" if analysis failed
        """
        try:
            if not os.path.exists(filepath):
                self.logger.error(f"Image file not found: {filepath}")
                return "NULL"
                
            ph_result = ph_from_image(filepath)
            self.logger.info(f"pH analysis result: {ph_result}")
            return ph_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing photo: {e}")
            return "NULL"
    
    def request_and_analyze_photo(self):
        """
        Convenience method to request a photo and immediately analyze it.
        
        Returns:
            tuple: (image_path: str or None, ph_value: str)
        """
        image_path = self.request_photo()
        if image_path:
            ph_value = self.analyze_photo(image_path)
            return image_path, ph_value
        else:
            return None, "NULL"
    
    def change_led_color(self, r, g, b):
        """
        Change the LED color on the server.
        
        Args:
            r (int): Red value (0-255)
            g (int): Green value (0-255)
            b (int): Blue value (0-255)
            
        Returns:
            bool: True if color change successful, False otherwise
        """
        if not self.connected or not self.socket:
            self.logger.error("Not connected to server. Call connect() first.")
            return False
            
        try:
            # Validate RGB values
            if not all(0 <= val <= 255 for val in (r, g, b)):
                self.logger.error("RGB values must be between 0-255")
                return False
            
            # Send color change request
            self.socket.sendall("CHANGE_COLOR".encode('utf-8'))
            
            # Wait for server's RGB request
            response = self.socket.recv(buffer_size).decode('utf-8').strip()
            
            if response == "PLEASE SEND RGB":
                # Send RGB values
                self.socket.sendall(f"{r},{g},{b}".encode('utf-8'))
                
                # Get response from server
                result = self.socket.recv(buffer_size).decode('utf-8').strip()
                if result == "COLOR_CHANGED":
                    self.logger.info(f"Successfully changed LED color to RGB({r},{g},{b})")
                    return True
                else:
                    self.logger.error(f"Error changing color: {result}")
                    return False
            else:
                self.logger.error(f"Unexpected server response: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error changing LED color: {e}")
            return False
    
    def run_motor(self):
        """
        Run the motor on the server.
        
        Returns:
            bool: True if motor run successful, False otherwise
        """
        if not self.connected or not self.socket:
            self.logger.error("Not connected to server. Call connect() first.")
            return False
            
        try:
            self.socket.sendall("RUN_MOTOR".encode('utf-8'))
            response = self.socket.recv(buffer_size).decode('utf-8').strip()
            
            if response == "MOTOR_RUN_COMPLETE":
                self.logger.info("Motor run complete")
                return True
            else:
                self.logger.error(f"Unexpected server response: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error running motor: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically disconnect."""
        self.disconnect()


# For backward compatibility, also expose the original PhotoClient
from .photo_client import PhotoClient

__all__ = ['ImageReqClient', 'PhotoClient'] 