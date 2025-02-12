# PizeroCam
- This project uses Raspberry Pi Zero 2W as a Server that takes still images with Pi Camera 3.
- This project uses the Neopixel ring LED for illumination and the PiSugar project for UPS (optional).
- Clients on any platform can request photos to be taken and sent over.
- One use case is to read the pH according to the text and colour scheme on the pH scale.

## Wireless pi-camera module on Raspberry Pi Zero 2 WH (Server)
Please follow the instructions (to be added #TODO) to set up the Pi Zero

Pi Zero has limited RAM; SHH using the command-line interface (CLI) is recommended. 

However, you can also use the Pi Connect Service to connect to Pi Zero.

  1. Assemble the hardware components.
     
     If you are using PiSugar (Wireless):
     ```
     wget https://cdn.pisugar.com/release/pisugar-power-manager.sh
     bash pisugar-power-manager.sh -c release
     ```
     PiSugar will establish a web server through which you can check your battery status.
     Use the link at the end of the installation.

  3. Clone this repo; 
     ```
     clone https://github.com/cyrilcaoyang/pizerocam/
     ```
     DO NOT enter the project folder yet
     
  5. Create a Python environment outside of the project folder
     ```
     python -m venv picam_env --system-site-packages
     ```
     Make sure to use the flag at the end, otherwise, picamera2 package cannot be recognized.
     
  7. Activate the venv
     ```
     source picam_env/bin/activate
     ```
    
  8. Install the LED driver code
     ```
     pip3 install adafruit-circuitpython-neopixel
     ```
    
  9. DO NOT pip install this package

     Just navigate to the server.py code and start the server
     ```
     python server.py
     ```

## Installing of the Client on the PC that runs the workflows (Client)

  1. It is recommended to use the Conda environment (to be added)
  2. Clone and install the sdl_utils package.
     ```
     clone ~~~~
     pip install -e .
     ```
  3. For the demo, Python Imaging Library (PIL) (or the Pillow fork)
     ```
     pip install pillow
     ```
     Download and install Tesseract for OCR (which requires Python 3.6+)
      - github.com/UB-Mannheim/tesseract/wiki

  4. Clone and install this package. DO pip install so you get all the required packages.
     ```
     clone https://github.com/cyrilcaoyang/pizerocam/
     pip install -e .
     ```
    
  5. Navigate to the client.py code and start the client
     ```
     python client.py
     ```
