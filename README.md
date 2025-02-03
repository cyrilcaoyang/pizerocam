# PizeroCam
Use Raspberry Pi Zero 2W as a Server. 
Clients can request photos to be taken and sent over.

# Wireless pi-camera module on Raspberry Pi Zero 2 WH (Server)

  1. Assemble the hardware components
     If you are using PiSugar:
     - wget https://cdn.pisugar.com/release/pisugar-power-manager.sh
     - bash pisugar-power-manager.sh -c release
     navigate to the link at the end of the installation to check battery status

  2. Create a folder and put the Server code in this folder
    
  3. Create a Python environment:
     - python -m venv picam_env --system-site-packages"
     Make sure to use the flag at the end, otherwise picamera2 will not work
     
  5. Activate the venv
      - source picam_env/bin/activate
    
  6. Install the LED driver code
     - sudo pip3 install adafruit-circuitpython-neopixel
     
  7. Now you are ready to start the server
     python client.py

# Installing of the Client on the PC that runs the workflows (Client)
  1. Install the required packages on PC: 
cv2==4.10.0
numpy==2.2.1
pytesseract==0.3.13
