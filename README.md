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

# Install system dependencies first (if needed)
sudo apt update
sudo apt install python3-picamera2

# Install the server package
pip install ".[server]"
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

# Basic start/stop
server = ImageServer(host="0.0.0.0", port=2222)
print(f"Server IP: {server.get_ip_address()}")

# Start in background (non-blocking)
if server.start(background=True):
    print("Server started successfully")
    
    # Do other work while server runs...
    input("Press Enter to stop server...")
    
    # Stop server
    if server.stop():
        print("Server stopped successfully")

# Start in foreground (blocking)
server.start(background=False)  # This will block until stopped

# Context manager (automatic cleanup)
with ImageServer() as server:
    print(f"Server running at {server.get_ip_address()}")
    # Server runs until context exits

# Status checking
if not server.is_running():
    server.start()
print(f"Server status: {'Running' if server.is_running() else 'Stopped'}")
```

**pH Analysis Standalone:**

```python
from image_req_client import ph_from_image

# Analyze existing image
ph_value = ph_from_image("path/to/image.jpg")
print(f"pH: {ph_value}")  # Returns pH value or "NULL"
```

## Server Control and Management

### Starting and Stopping the Server

The ImageServer provides comprehensive start/stop functionality with multiple usage patterns:

#### 1. Command Line Interface

**Basic server start:**
```bash
# Start with default settings (host: 0.0.0.0, port: 2222)
pizerocam-server

# Custom host and port
pizerocam-server --host 192.168.1.100 --port 3333

# Enable verbose logging
pizerocam-server --verbose
```

**Stopping the server:**
- Press `Ctrl+C` for graceful shutdown
- The CLI handles signal management automatically

#### 2. Python API Control

**Background Mode (Non-blocking):**
```python
from image_server import ImageServer

server = ImageServer()

# Start server in background thread
if server.start(background=True):
    print("Server started successfully")
    print(f"Server IP: {server.get_ip_address()}")
    
    # Your application continues running here
    # Server handles requests in the background
    
    # Stop when needed
    server.stop()
```

**Foreground Mode (Blocking):**
```python
from image_server import ImageServer

server = ImageServer()

try:
    # This will block until server is stopped
    server.start(background=False)
except KeyboardInterrupt:
    print("Server interrupted")
finally:
    server.stop()
```

**Context Manager (Automatic Cleanup):**
```python
from image_server import ImageServer

# Server automatically stops when exiting context
with ImageServer(host="0.0.0.0", port=2222) as server:
    print(f"Server running at {server.get_ip_address()}")
    input("Press Enter to stop...")
# Server automatically stopped here
```

**Status Monitoring:**
```python
from image_server import ImageServer
import time

server = ImageServer()

# Check if server is running
if not server.is_running():
    print("Starting server...")
    server.start()

# Monitor status
while server.is_running():
    print(f"Server status: Running on {server.get_ip_address()}")
    time.sleep(5)
    
    # Stop condition (example)
    if some_stop_condition():
        server.stop()
        break

print("Server stopped")
```

#### 3. Advanced Server Management

**Error Handling:**
```python
from image_server import ImageServer

server = ImageServer()

try:
    if server.start():
        print("Server started successfully")
    else:
        print("Failed to start server - check logs")
        
    # Server operations...
    
except Exception as e:
    print(f"Server error: {e}")
finally:
    if server.is_running():
        if server.stop():
            print("Server stopped gracefully")
        else:
            print("Error stopping server")
```

**Multiple Server Instances:**
```python
from image_server import ImageServer

# Create multiple servers on different ports
server1 = ImageServer(port=2222)
server2 = ImageServer(port=2223)

# Start both
server1.start(background=True)
server2.start(background=True)

print(f"Server 1: {server1.get_ip_address()}:2222")
print(f"Server 2: {server2.get_ip_address()}:2223")

# Stop both
server1.stop()
server2.stop()
```

### Server Features

- **Graceful Shutdown**: Proper cleanup of resources and connections
- **Thread Safety**: Safe concurrent client handling
- **Error Recovery**: Robust error handling and logging
- **Status Monitoring**: Real-time server status checking
- **Flexible Deployment**: Background or foreground operation modes
- **Context Management**: Automatic resource cleanup
- **Signal Handling**: Responds to system signals (Ctrl+C, SIGTERM)

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
   sudo apt install python3-picamera2 portaudio19-dev
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
   pip install ".[server]"
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
- `start(background=True)` - Start server (returns bool)
  - `background=True`: Non-blocking, runs in background thread
  - `background=False`: Blocking, runs in current thread
- `stop()` - Stop server gracefully (returns bool)
- `is_running()` - Check if server is currently running (returns bool)
- `get_ip_address()` - Get server's IP address (returns str)
- `get_logger()` - Get server's logger instance
- Context manager support (`__enter__`/`__exit__`)

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
