# PizeroCam
- This project uses Raspberry Pi Zero 2W as a Server that takes still images with Pi Camera 3.
- This project uses the Neopixel ring LED for illumination and the PiSugar project for UPS (optional).
- Clients on any platform can request photos to be taken and sent over.
- One use case is to read the pH according to the text and colour scheme on the pH scale.

## Installation

This package is composed of two main components: a `server` to be run on a Raspberry Pi with a camera, and a `client` that can be run on any computer to request and analyze photos.

You can install the components separately depending on your needs.

### Server-side Installation (on Raspberry Pi)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/cyrilcaoyang/pizerocam/
    cd pizerocam
    ```

2.  **Create a virtual environment:**
    It is recommended to use the `--system-site-packages` flag to have access to system-level packages like `picamera2`.
    ```bash
    python3 -m venv venv --system-site-packages
    source venv/bin/activate
    ```

3.  **Install the server package:**
    This will install the server and all its dependencies.
    ```bash
    pip install .[server]
    ```

### Client-side Installation (on your computer)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/cyrilcaoyang/pizerocam/
    cd pizerocam
    ```

2.  **Create a Conda environment:**
    ```bash
    conda create -n pizerocam-client python=3.9
    conda activate pizerocam-client
    ```

3.  **Install the client package:**
    You have two options for installing the client, depending on which OCR engine you want to use.

    **Option 1: Tesseract (Local OCR)**
    This option uses the Tesseract engine to perform OCR locally on your machine.
    ```bash
    pip install .[client-tesseract]
    ```
    **Note on Tesseract:** You need to have Google's Tesseract OCR engine installed on your system. You can find installation instructions here: [Tesseract Installation](https://github.com/tesseract-ocr/tesseract/wiki)

    **Option 2: Google Cloud Vision (Cloud-based OCR)**
    This option uses the Google Cloud Vision API for OCR.
    ```bash
    pip install .[client-gcloud]
    ```
    **Note on Google Cloud:** You will need to set up a Google Cloud Platform project with the Vision API enabled and configure your authentication.

### Developer Installation

If you want to install all components for development (server, and both client types), you can use Conda for simplicity.

1.  **Create a Conda environment:**
    ```bash
    conda create -n pizerocam-dev python=3.9
    conda activate pizerocam-dev
    ```

2. **Install packages:**
    ```bash
    pip install .[server,client-tesseract,client-gcloud]
    ```

## Usage

### Running the Server

Navigate to the `src/picam_server` directory and run the server:

```bash
python server.py
```

### Running the Client

Navigate to the `src/image_client` directory and run the interactive client:

```bash
python photo_client.py
```

## Wireless pi-camera module on Raspberry Pi Zero 2 WH (Server)
Please follow the instructions (to be added #TODO) to set up the Pi Zero

Pi Zero has limited RAM; SHH using the command-line interface (CLI) is recommended. 

However, you can also use the Pi Connect Service to connect to Pi Zero.

  1. Assemble the hardware components.
  
  2. Make sure to turn on under SPI and I2C: 
     ```
     sudo raspi-config
     ```

     "Interface Options" -> turn on both SPI and I2C options
     
     If you are using PiSugar:
     ```
     wget https://cdn.pisugar.com/release/pisugar-power-manager.sh
     bash pisugar-power-manager.sh -c release
     ```
     PiSugar will establish a web server through which you can check your battery status.
     Use the link at the end of the installation.

  4. Clone this repo; 
     ```
     git clone https://github.com/cyrilcaoyang/pizerocam/
     ```
     
  5. Create a Python environment (it will be outside of the project folder, this will take some time)
     ```
     python -m venv picam_env --system-site-packages
     ```
     Make sure to use the flag at the end, otherwise, picamera2 package cannot be recognized.
     
  6. Activate the venv
     ```
     source picam_env/bin/activate

     ```
  7. (Optional) Activate the environment every login session(SSH):
      
      ```
      nano ~/.profile
      ```
      
      Add the following line:
      ```
      source PATH_TO_ENVIRONMENT/bin/activate
      ```

      Apply changes:
      ```
      source ~/.profile
      ```
      
  8. Install the LED driver code
     ```
     pip3 install adafruit-circuitpython-neopixel pyyaml
     ```
    
  9. Just navigate to the server.py code and start the server
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
