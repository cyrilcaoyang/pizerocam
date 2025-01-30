import socket
import json
from PIL import Image

# Open and read the JSON file
try:
    with open('client_settings.json', 'r') as file:
        data = json.load(file)
    IP = data['server_ip']
    print('Server IP address is:', IP)
except Exception as e:
    print(f"Error reading JSON file: {e}")
    exit()

# Connect to picam_server
msgFromClient = "Hi. This is Client, are you ready to take a photo?"
bytesToSend = msgFromClient.encode('utf-8')
serverAddress = (IP, 2222)
bufferSize = 1024

try:
    UDPClient = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    UDPClient.sendto(bytesToSend, serverAddress)
    print("Message sent to server")
except Exception as e:
    print(f"Error sending message to server: {e}")
    exit()

# Offer options to user
try:
    while True:
        print("\nOptions:")
        print("\n1. Request photo")
        print("\n2. Exit")
        option = input("\nEnter your choice: ")

        if option == "1":
            # Send TAKE_PHOTO command to the picam_server
            try:
                UDPClient.sendto("TAKE_PHOTO".encode(), serverAddress)
                print("TAKE_PHOTO command sent")

                # Get the size of the image first
                image_size, _ = UDPClient.recvfrom(bufferSize)
                image_size = int(image_size.decode())
                print('The image size to be received is:', image_size)

                # Get the image in chunks
                data = b''
                while len(data) < image_size:
                    chunk, _ = UDPClient.recvfrom(bufferSize)
                    if not chunk:
                        break
                    data += chunk

                with open('image.jpg', 'wb') as file:
                    file.write(data)

                print('Image stored as image.jpg')
                print('Displaying image')
                image = Image.open('image.jpg')
                image.show()
            except Exception as e:
                print(f"Error during photo request: {e}")

        elif option == '2':
            print('Exiting')
            UDPClient.close()
            exit()
        else:
            print("Invalid option. Please try again.")

except KeyboardInterrupt:
    print("Client stopped by user")
    UDPClient.close()