"""
This example will explore the (r, g, b) color space with N intevals on each dimension
N can be any of  [2, 4, 8, 16, 32, 64, 128, 256]
A total of n^3 photos will be taken and stored in the 'photos' folder
Filename is in the format of color-mapping_time-stamp_color-coordination.png

"""
import yaml
import socket
from time import sleep
from pathlib import Path
from photo_client import ImageClient
from sdl_utils import get_logger, connect_socket, send_file_name, receive_file_name
from sdl_utils import send_file_size, receive_file_size, receive_file

# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent

# Open and read the JSON file
with open(script_dir / 'client_settings.yaml', 'r') as file:
    data = yaml.safe_load(file)
server_ip = data['Server_IP']
server_port = data['ServerPort']
buffer_size = data['BufferSize']
chunk_size = data['ChunkSize']

def get_photo(
        conn,
        color_cord: list,
        client: ImageClient
):
    """
    Send command to take a photo using the LED lighting of a color coordination
    And store the photo in the 'photos' folder
    :param conn: Active connection
    :param color_cord: Color coordinate as a list in the format of [255, 255, 255]
    :param client:
    :return:
    """
    # Send color change request
    conn.sendall("CHANGE_COLOR".encode('utf-8'))

    # Wait for server's RGB request
    response = conn.recv(buffer_size).decode('utf-8').strip()
    if response != "PLEASE SEND RGB":
        client.logger.error("Incorrect response from the server.")
        return
    elif len(color_cord) != 3:
        client.logger.error("Invalid format. Please use R,G,B format")
        return

    try:
        r, g, b = int(color_cord[0]), int(color_cord[1]), int(color_cord[2])
        if not all(0 <= val <= 255 for val in (r, g, b)):
            client.logger.error("Invalid values. Values must be between 0-255")
            return

        # Send validated RGB values
        conn.sendall(f"{r},{g},{b}".encode('utf-8'))

        # Get response from server
        result = conn.recv(buffer_size).decode('utf-8').strip()
        if result != "COLOR_CHANGED":
            client.logger.error(f"Error changing color: {result}")
            return
        client.logger.info("Successfully changed LED color!")

        # Request photo
        try:
            conn.sendall("TAKE_PHOTO".encode('utf-8'))
            sleep(3)
            success, image_path = client.receive_photo(conn)
            if success:
                client.logger.info(f"Photo received and saved as {image_path}")
            else:
                client.logger.info("Failed to receive complete photo")

        except Exception as e:
            print(f"Error during photo request: {e}")
            return
    except ValueError:
        client.logger.error("Invalid input. Please enter integers only")
        return

def color_mapping_session(interval, client: ImageClient):
    if interval not in [2, 4, 8, 16, 32, 64, 128, 256]:
        client.logger.error("Invalid input. Intervals must be powers of 2.")
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s = connect_socket(s, server_ip, server_port, client.logger)
        if s == None:
            return
        client.logger.info("Entering the color-mapping experiment.")
        while True:
            option = input("Start?(Y/N): ").strip()
            if option in ['n', 'N', 'No', 'no']:
                client.logger.info("Experiment aborted.")
                return
            elif option in ['y', 'Y', 'Yes', 'yes']:
                break
            else:
                client.logger.info("Invalid input, please try again.")

        # Iterate through color space
        for r in range (interval-1, 256, interval):
            for g in range (interval-1, 256, interval):
                for b in range (interval-1, 256, interval):
                    color = [r, g, b]
                    get_photo(s, color, client)

        # Finishing up by exitingY

        client.logger.info('Exiting')
        s.close()


if __name__ == "__main__":
    client = ImageClient()
    # Please confirm that you have the right server IP address
    client.update_server_ip()

    # Intervals must be powers of 2, from 2 up to 256
    color_mapping_session(interval=32, client=client)
