"""
Fixed pH client that correctly handles file transfer from the Pi camera.

This fixes the type error in the original ImageReqClient by implementing 
proper socket communication and file transfer.

Author: Yang Cao, Acceleration Consortium (SDL2 Team)
Email: yangcyril.cao@utoronto.ca
Version: 0.3

1. Set up google cloud SDK and CLI
2. Clone and install pizerocam (cyrilcaoyang)
3. pip install both in your chosen env.

Example usage
"""

import os
import socket
import struct
from pathlib import Path
from time import sleep
import sys

# Ensure repo root is on sys.path so top-level imports (e.g. `logger`) work
try:
    repo_root = Path(__file__).resolve().parents[2]
    # The project's Python packages live under the `src/` directory
    src_dir = repo_root / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)
except Exception:
    # Non-fatal; if this fails imports may still work depending on how script is run
    pass


class pHRead:
    """Fixed pH client with proper file transfer handling."""
    
    def __init__(self, server_ip="172.31.34.163", port=2222):
        self.server_ip = server_ip
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Connect to the Pi server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.port))
            self.connected = True
            print(f"[+] Connected to {self.server_ip}:{self.port}")
            return True
        except Exception as e:
            print(f"[!] Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from server."""
        if self.socket:
            try:
                self.socket.close()
                print("Disconnected from server")
            except:
                pass
            finally:
                self.socket = None
                self.connected = False
    
    def _receive_string(self):
        """Receive a newline-terminated string from the server.

        This reads until a "\n" is seen to avoid TCP coalescing issues where
        multiple logical messages arrive in a single recv call.
        """
        try:
            chunks = []
            while True:
                chunk = self.socket.recv(1024)
                if not chunk:
                    # connection closed
                    print("Error receiving string: connection closed by peer")
                    return None
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
            data = b"".join(chunks).decode('utf-8')
            # Split at the first newline and keep remainder in the socket buffer implicitly
            line = data.split('\n', 1)[0].strip()
            return line
        except Exception as e:
            print(f"Error receiving string: {e}")
            return None
    
    def _send_string(self, message):
        """Send a newline-terminated string to the server."""
        try:
            if not message.endswith('\n'):
                message = message + '\n'
            self.socket.sendall(message.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Error sending string: {e}")
            return False
    
    def _receive_file_properly(self, file_size, chunk_size=4096):
        """
        Properly receive file data with correct type handling.
        
        Args:
            file_size (int): Expected file size in bytes
            chunk_size (int): Size of chunks to receive
            
        Returns:
            bytes: The received file data
        """
        received_data = b""
        bytes_received = 0
        
        print(f"Receiving file: 0/{file_size} bytes", end="", flush=True)
        
        while bytes_received < file_size:
            # Calculate how much to receive in this chunk
            remaining = file_size - bytes_received
            to_receive = min(chunk_size, remaining)
            
            try:
                # Receive chunk
                chunk = self.socket.recv(to_receive)
                if not chunk:
                    print(f"\n[!] Connection closed by server at {bytes_received}/{file_size} bytes")
                    break
                    
                received_data += chunk
                bytes_received += len(chunk)
                
                # Show progress
                progress = int((bytes_received / file_size) * 50)
                bar = "#" * progress + "-" * (50 - progress)
                print(f"\rReceiving file: [{bar}] {bytes_received}/{file_size} bytes", end="", flush=True)
                
            except Exception as e:
                print(f"\n[!] Error receiving chunk: {e}")
                break
        
        print()  # New line after progress bar
        return received_data
    
    def request_photo(self):
        """Request a photo from the Pi and save it locally."""
        if not self.connected:
            print("[!] Not connected. Call connect() first.")
            return None, None
        
        try:
            # Create output directory
            home_dir = Path.home()
            output_dir = home_dir / "Pictures" / "pH_photos"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Send photo request
            print("[*] Requesting photo...")
            self._send_string("TAKE_PHOTO")
            sleep(5)  # Wait for photo capture
            
            # Receive filename (newline-terminated)
            filename = self._receive_string()
            if not filename:
                print("[!] Failed to receive filename")
                return None, None

            print(f"[*] Filename received (len={len(filename)}): '{filename}'")

            # Echo filename back (with newline)
            if not self._send_string(filename):
                print("[!] Failed to echo filename back to server")
                return None, None
            
            # Receive file size as string and convert to int
            file_size_str = self._receive_string()
            if not file_size_str:
                print("[!] Failed to receive file size")
                return None, None
                
            try:
                file_size = int(file_size_str)  # This is the key fix!
                print(f"[*] File size string received: '{file_size_str}' -> {file_size:,} bytes")
            except ValueError:
                print(f"[!] Invalid file size received: '{file_size_str}'")
                return None, None
            
            # Echo file size back
            if not self._send_string(file_size_str):
                print("[!] Failed to echo file size back to server")
                return None, None
            
            # Receive the actual file data
            print("[*] Receiving file data...")
            file_data = self._receive_file_properly(file_size)
            print(f"[*] Received raw file data length: {len(file_data)} bytes")
            
            if len(file_data) != file_size:
                print(f"[!] File size mismatch: expected {file_size}, got {len(file_data)}")
                if len(file_data) == 0:
                    print("[!] No file data received!")
                    return None, None
            
            # Save the file
            file_path = output_dir / filename
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            print(f"[+] File saved: {file_path}")
            print(f"[*] File size on disk: {os.path.getsize(file_path):,} bytes")
            
            return str(file_path), self._analyze_ph(str(file_path))
            
        except Exception as e:
            print(f"[!] Error requesting photo: {e}")
            return None, None
    
    def _analyze_ph(self, image_path):
        """Analyze pH from image using the original function."""
        try:
            # Import directly from the module file to avoid circular/relative import issues
            import importlib.util
            module_path = Path(__file__).parent / "ph_grid_color_reader.py"
            spec = importlib.util.spec_from_file_location("ph_grid_color_reader", module_path)
            ph_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ph_module)
            ph_from_image = ph_module.ph_from_image
            
            # Ensure analyzer writes outputs into repository-local output_images/
            repo_root = Path(__file__).resolve().parents[2]
            out_dir = repo_root / "output_images"
            result = ph_from_image(image_path, output_dir=str(out_dir))
            print(f"[*] pH Analysis: {result}")
            return result
        except Exception as e:
            # Print full traceback for easier debugging
            import traceback
            print(f"[!] pH analysis failed: {e}")
            traceback.print_exc()
            return "NULL"
    
    def read(self):
        """Read pH value by taking and analyzing a photo."""
        image_path, ph_value = self.request_photo()
        return ph_value


if __name__ == "__main__":
    ph_read = pHRead()
    ph_read.connect()
    ph = ph_read.read()
    print(f"========= pH is {ph} ===========")
    ph_read.disconnect()

    # to change led color "pizerocam-client --ip 100.64.254.98 --led 10,10,10" (RGB values AT THE END)