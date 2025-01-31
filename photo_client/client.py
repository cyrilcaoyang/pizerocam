import os
import sys
import json
import logging
import socket
from datetime import datetime
from time import sleep
from PIL import Image

# Open and read the JSON file
with open('client_settings.json', 'r') as file:
    data = json.load(file)
server_ip = data['Server_IP']
server_port = data['ServerPort']
buffer_size = data['BufferSize']
chunk_size = data['ChunkSize']


class ImageClient:
    """
    This is a client that requests and receives images
    """
    def __init__(self, host="0.0.0.0", port=server_port):
        self.host = host
        self.port = port
        self.server_ip = server_ip
        self.logger = self.setup_logger()

    @staticmethod
    def setup_logger():
        # Create a directory to store logs
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)

        # Generate a timestamped log filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = os.path.join(log_dir, f"{timestamp}.log")

        # Create the logger and file handler
        logger = logging.getLogger("ImageClientLogger")
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_filename)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))
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
        return self.server_ip

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

    def _send_ascii_length(self, conn, value):
        """
        Helper: Sends the integer 'value' as ASCII digits followed by a newline.
        """
        message = f"{value}\n"
        conn.sendall(message.encode("utf-8"))

    def receive_photo(self, sock):
        """
        :param sock:
        :return: absolute image path
        """
        # Create the photos directory if it does not exist already
        photo_dir = 'photos'
        os.makedirs(photo_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        img_path = os.path.join(photo_dir, f"{timestamp}.jpg")

        # 1) Receive ASCII-based size until newline
        size_str = self._recv_until_newline(sock)
        if not size_str:
            raise ConnectionError("Did not receive file size from server (connection closed).")

        self.logger.info(f"Server reports file size: {size_str} bytes (ASCII)")

        # 2) Parse it into an integer
        try:
            file_size = int(size_str)
        except ValueError:
            raise ValueError(f"Could not parse file size as int: '{size_str}'")

        # 3) Echo the size back to server (ASCII + newline)
        self._send_ascii_length(sock, file_size)
        self.logger.info("Echoed the file size back to server.")

        # 4) Now receive the actual file data in chunks
        received_data = b''
        bytes_received = 0
        self.logger.info(f"Receiving file of size {file_size} bytes...")
        while bytes_received < file_size:
            chunk = sock.recv(min(chunk_size, file_size - bytes_received))
            if not chunk:
                raise ConnectionError("Connection lost while receiving file data.")
            received_data += chunk
            bytes_received += len(chunk)

        self.logger.info("Received the entire file from server.")

        # 5) Write the file to disk
        output_dir = "photos"
        os.makedirs(output_dir, exist_ok=True)

        with open(img_path, "wb") as f:
            f.write(received_data)

        self.logger.info(f"File saved to: {img_path}")

        return True, img_path

    def client_session(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((server_ip, server_port))
            self.logger.info("Connected to server")

            while True:
                print("Options:")
                print("1. Request photo")
                print("2. Exit")
                option = input("Enter your choice: ")

                if option == "1":
                    try:
                        s.sendall("TAKE_PHOTO".encode('utf-8'))
                        sleep(10)

                        # TODO also get the filename
                        success, image_path = self.receive_photo(s)
                        if success:
                            self.logger.info("Photo received and saved as received_photo.jpg")
                        else:
                            self.logger.info("Failed to receive complete photo")

                        image = Image.open(image_path)
                        image.show()
                        continue
                    except Exception as e:
                        print(f"Error during photo request: {e}")

                elif option == '2':
                    self.logger.info('Exiting')
                    s.close()
                    break
                else:
                    print("Invalid option. Please try again.")


if __name__ == "__main__":
    client = ImageClient()
    client.client_session()
