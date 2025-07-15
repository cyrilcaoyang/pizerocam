# PiZeroCam

A Python package for remote camera control and pH analysis using Raspberry Pi Zero 2W.

**Key Features:**
- Remote camera control with LED illumination
- Motor control for automated testing
- pH value analysis from color grid images
- Modular client/server architecture
- Easy installation with optional dependencies
- CLI tools for both client and server

## Architecture

PiZeroCam consists of two main modules:

- **`image_server`**: Runs on Raspberry Pi Zero 2W with camera and motor hardware
- **`image_req_client`**: Runs on any computer to request photos and analyze them

## Installation

### Client Installation (on your computer)

For basic photo requesting and pH analysis:

```bash
git clone https://github.com/cyrilcaoyang/pizerocam.git
cd pizerocam
pip install .[client]
```

Alternative client options:
- `pip install .[client-tesseract]` - for local Tesseract OCR (requires Tesseract installation)
- `pip install .` - base installation without OCR dependencies

### Server Installation (on Raspberry Pi)

```bash
git clone https://github.com/cyrilcaoyang/pizerocam.git
cd pizerocam
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install .[server]
```

**Note:** Use `--system-site-packages` to access system-level packages like `picamera2`.

### Developer Installation

To install both client and server components:

```bash
pip install .[client,server]
```

## Usage

### Quick Start with CLI Tools

**Start the server (on Raspberry Pi):**
```bash
pizerocam-server
```

**Connect from client (on your computer):**
```bash
# Interactive session
pizerocam-client --ip 192.168.1.100

# Request photo and analyze pH
pizerocam-client --ip 192.168.1.100 --photo --analyze

# Change LED color to red
pizerocam-client --ip 192.168.1.100 --led 255,0,0

# Run motor
pizerocam-client --ip 192.168.1.100 --motor
```

### Python API Usage

**Client API:**

```python
from image_req_client import ImageReqClient

# Create and connect to server
client = ImageReqClient()
client.connect("192.168.1.100")

# Request photo and analyze
image_path = client.request_photo()
if image_path:
    ph_value = client.analyze_photo(image_path)
    print(f"pH value: {ph_value}")

# Control LED and motor
client.change_led_color(255, 0, 0)  # Red LED
client.run_motor()

# Disconnect
client.disconnect()

# Or use as context manager
with ImageReqClient() as client:
    client.connect("192.168.1.100")
    image_path, ph_value = client.request_and_analyze_photo()
    print(f"Image: {image_path}, pH: {ph_value}")
```

**Server API:**

```python
from image_server import ImageServer

# Create and start server
server = ImageServer(host="0.0.0.0", port=2222)
print(f"Server IP: {server.get_ip_address()}")
server.start()  # Runs in background by default

# Or use as context manager (blocking)
with ImageServer() as server:
    print(f"Server running at {server.get_ip_address()}")
    # Server runs until context exits
```

**pH Analysis Standalone:**

```python
from image_req_client import ph_from_image

# Analyze existing image
ph_value = ph_from_image("path/to/image.jpg")
print(f"pH: {ph_value}")  # Returns pH value or "NULL"
```

## Hardware Setup (Raspberry Pi Server)

### Required Components
- Raspberry Pi Zero 2W/WH
- Pi Camera 3
- NeoPixel ring LED
- PCA9685 PWM driver (for motor control)
- DC motor
- Optional: PiSugar for UPS functionality

### Setup Instructions

1. **Enable interfaces:**
   ```bash
   sudo raspi-config
   ```
   Enable SPI and I2C under "Interface Options"

2. **Install system dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-picamera2
   ```

3. **Optional - Install PiSugar:**
   ```bash
   wget https://cdn.pisugar.com/release/pisugar-power-manager.sh
   bash pisugar-power-manager.sh -c release
   ```

4. **Install PiZeroCam:**
   ```bash
   git clone https://github.com/cyrilcaoyang/pizerocam.git
   cd pizerocam
   python3 -m venv venv --system-site-packages
   source venv/bin/activate
   pip install .[server]
   ```

5. **Start the server:**
   ```bash
   pizerocam-server
   ```

### Hardware Connections

**NeoPixel LED:**
- Connect to GPIO pin (configured in server code)
- Provides illumination for camera

**Motor (via PCA9685):**
- PWMA: Channel 0
- AIN1: Channel 1  
- AIN2: Channel 2
- I2C address: 0x40

## pH Analysis

The pH analysis tool uses Google Cloud Vision API for OCR text detection and color analysis to determine pH values from test strips.

### Features
- Detects pH numbers (1-12) in grid layouts
- Analyzes color blocks below detected numbers
- Uses BGR color values and Euclidean distance for pH determination
- Generates labeled output images for verification
- Handles multi-digit number splitting intelligently

### Requirements
- Google Cloud Vision API credentials
- Environment variable: `GOOGLE_APPLICATION_CREDENTIALS`

### Configuration
The analysis region is currently configured for coordinates (750, 1100, 150, 300). Modify in the source code as needed for your specific test strips.

## API Reference

### ImageReqClient Methods
- `connect(ip)` - Connect to server
- `disconnect()` - Disconnect from server  
- `request_photo()` - Request and receive photo
- `analyze_photo(filepath)` - Analyze photo for pH
- `request_and_analyze_photo()` - Combined request and analysis
- `change_led_color(r, g, b)` - Change LED color
- `run_motor()` - Run motor

### ImageServer Methods
- `start(background=True)` - Start server
- `stop()` - Stop server
- `get_ip_address()` - Get server IP
- `is_running()` - Check if running

### CLI Options

**pizerocam-client:**
```
--ip IP          Server IP address (required)
--port PORT      Server port (default: 2222)
--photo          Request photo
--analyze        Analyze photo for pH (with --photo)
--led R,G,B      Change LED color
--motor          Run motor
--interactive    Interactive session (default)
```

**pizerocam-server:**
```
--host HOST      Bind interface (default: 0.0.0.0)
--port PORT      Listen port (default: 2222)
--verbose        Enable verbose logging
```

## Development

### Project Structure
```
pizerocam/
├── src/
│   ├── image_req_client/          # Client module
│   │   ├── __init__.py
│   │   ├── image_req_client.py    # Main client class
│   │   ├── ph_grid_color_reader.py # pH analysis
│   │   ├── photo_client.py        # Legacy client
│   │   └── cli.py                 # CLI interface
│   └── image_server/              # Server module
│       ├── __init__.py
│       ├── image_server.py        # Main server class
│       ├── server.py              # Camera server
│       ├── ph_test_server.py      # Extended server with motor
│       ├── PCA9685.py            # Motor driver
│       └── cli.py                # CLI interface
├── examples/                      # Example scripts
├── pyproject.toml                # Package configuration
└── README.md
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: https://github.com/cyrilcaoyang/pizerocam/issues
- Email: cyrilcaoyang@gmail.com
