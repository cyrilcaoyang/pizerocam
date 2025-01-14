import socket
import time, os
from picamera2 import Picamera2

bufferSize = 2048
chunkSize = 1024
ServerPort = 2222
ServerIP = '172.31.35.135'

# Start the server
msgFromServer = "Hello, Happy to be Your Server"
bytesToSend=msgFromServer.encode('utf-8')
RPIsocket=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
RPIsocket.bind((ServerIP, ServerPort))
print('Sever is up and listening...')

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()

try:
    while True:

    # Get the message from the client
        message,address=RPIsocket.recvfrom(bufferSize)
        message=message.decode('utf-8')
        print(message)
        print("Client Address",address[0])

        if message == "TAKE_PHOTO":
            # Define the image file path
            image_path = 'image.jpg'
            picam2.capture_file(image_path)
            print(f"Photo captured and saved as {image_path}")

            # Read the image file as binary data
            with open(image_path, 'rb') as file:
                image_data = file.read() 
                image_size = len(image_data)

            # Send the size of the image
            RPIsocket.sendto(str(image_size).encode(),address)
            print('Sending captured image...')

            # Send the image
            for i in range(0, len(image_data), chunkSize):
                RPIsocket.sendto(image_data[i:i+chunkSize],address) 
            time.sleep(2)
            print('Image sent.\nServer listening...')

except KeyboardInterrupt:
    print("Server stopped by user")
    RPIsocket.close()
    picam2.close()
