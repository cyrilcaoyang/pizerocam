import os
import socket
import yaml
from datetime import datetime
from pathlib import Path
from time import sleep
from PIL import Image
from sdl_utils import get_logger
from sdl_utils import connect_socket, send_file_name, receive_file_name
from sdl_utils import send_file_size, receive_file_size, receive_file

# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent

# Open and read the JSON file
with open(script_dir / 'image_client_settings.yaml', 'r') as file:
    data = yaml.safe_load(file)
server_ip = data['Server_IP']
server_port = data['ServerPort']
buffer_size = data['BufferSize']
chunk_size = data['ChunkSize']


class PhotoClient:
    """
    This is a client that requests and receives images
    More to be added
    """
    def __init__(self, host="0.0.0.0", port=server_port, logger=None):
        self.host = host
        self.port = port
        self.server_ip = server_ip
        self.logger = self.setup_logger()

    @staticmethod
    def setup_logger():
        # Create the logger and file handler
        logger = get_logger("ImageClientLogger")
        return logger

    def update_server_ip(self):
        ip_up_to_date = input(f"Is the server IP address: {self.server_ip}? [Y]: ")
        while True:
            if ip_up_to_date in ['', 'y', 'Y', 'yes', 'Yes']:
                break
            elif ip_up_to_date in ['n', 'N', 'No', 'no']:
                new_server_ip = input("What is the new ip address")
                self.logger.info(f"IP address updated to {new_server_ip}")
                self.server_ip = new_server_ip
                break

    def receive_photo(self, sock):
        """
        :param sock:
        :param photo_dir: the directory to save photos
        :return: absolute image path
        """
        # Create the photos directory if it does not exist already
        output_dir = "photos"
        os.makedirs(output_dir, exist_ok=True)
        
        # Receive the image name and echo back to confirm
        img_name = receive_file_name(sock, self.logger)
        self.logger.info(f"Server sent file name {img_name}.")
        send_file_name(sock, img_name, self.logger)
        self.logger.info("Echoed the file name back to server.")
        img_path = os.path.join(output_dir, img_name)

        # Receive ASCII-based file size  and echo back to confirm
        file_size = receive_file_size(sock, self.logger)
        self.logger.info(f"Server sent file size: {file_size} bytes.")
        send_file_size(sock, file_size, self.logger)
        self.logger.info("Echoed the file size back to server.")

        # Now receive the actual file data in chunks, and write the file to disk
        received_data = receive_file(sock, file_size, chunk_size, self.logger)
        with open(img_path, "wb") as f:
            f.write(received_data)
        self.logger.info(f"File {img_name} saved to: {output_dir}")
        return True, img_path

    def interactive_client_session(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s = connect_socket(s, server_ip, server_port, self.logger)
            if s == None:
                return
            while True:
                print("Options:\n1. Request photo\n2. Change LED color\n3. Exit")
                option = input("Enter your choice: ").strip()

                if option == "1":
                    try:
                        s.sendall("TAKE_PHOTO".encode('utf-8'))
                        sleep(5)

                        success, image_path = self.receive_photo(s)
                        if success:
                            self.logger.info(f"Photo received and saved as {image_path}")
                        else:
                            self.logger.info("Failed to receive complete photo")
                        continue
                    except Exception as e:
                        print(f"Error during photo request: {e}")

                elif option == '2':
                    try:
                        # Send color change request
                        s.sendall("CHANGE_COLOR".encode('utf-8'))

                        # Wait for server's RGB request
                        response = s.recv(buffer_size).decode('utf-8').strip()

                        if response == "PLEASE SEND RGB":
                            while True:
                                rgb_input = input("Enter RGB values (0-255) as R,G,B: ").strip()
                                parts = rgb_input.split(',')

                                if len(parts) != 3:
                                    print("Invalid format. Please use R,G,B format")
                                    continue
                                try:
                                    r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                                    if not all(0 <= val <= 255 for val in (r, g, b)):
                                        print("Values must be between 0-255")
                                        continue

                                    # Send validated RGB values
                                    s.sendall(f"{r},{g},{b}".encode('utf-8'))
                                    break

                                except ValueError:
                                    print("Invalid input. Please enter integers only")
                                    continue

                            # Get response from server
                            result = s.recv(buffer_size).decode('utf-8').strip()
                            if result == "COLOR_CHANGED":
                                print("Successfully changed LED color!")
                            else:
                                print(f"Error changing color: {result}")

                        else:
                            print(f"Unexpected server response: {response}")

                    except Exception as e:
                        print(f"Error during color change: {e}")

                elif option == '3':
                    self.logger.info('Exiting')
                    s.close()
                    break
                else:
                    print("Invalid option. Please try again.")


if __name__ == "__main__":
    client = PhotoClient()
    
    # Please confirm that you have the right server IP address
    client.update_server_ip()
    client.interactive_client_session()
