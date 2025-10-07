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


class pHRead:
    """
    Fixed pH client with proper file transfer handling.
    
    Always returns pH readings from all three color spaces: RGB, LAB, and HSV.
    """
    
    def __init__(self, server_ip="100.64.254.98", port=2222):
        """
        Initialize the pH reader.
        
        Args:
            server_ip (str): IP address of the Pi server
            port (int): Server port
        """
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
        """Receive a string from the server."""
        try:
            data = self.socket.recv(1024).decode('utf-8').strip()
            return data
        except Exception as e:
            print(f"Error receiving string: {e}")
            return None
    
    def _send_string(self, message):
        """Send a string to the server."""
        try:
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
            
            # Receive filename
            filename = self._receive_string()
            if not filename:
                print("[!] Failed to receive filename")
                return None, None
            
            print(f"[*] Filename: {filename}")
            
            # Echo filename back
            self._send_string(filename)
            
            # Receive file size as string and convert to int
            file_size_str = self._receive_string()
            if not file_size_str:
                print("[!] Failed to receive file size")
                return None, None
                
            try:
                file_size = int(file_size_str)  # This is the key fix!
                print(f"[*] File size: {file_size:,} bytes")
            except ValueError:
                print(f"[!] Invalid file size received: '{file_size_str}'")
                return None, None
            
            # Echo file size back
            self._send_string(file_size_str)
            
            # Receive the actual file data
            print("[*] Receiving file data...")
            file_data = self._receive_file_properly(file_size)
            
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
            
            # Return file path without analyzing yet
            return str(file_path)
            
        except Exception as e:
            print(f"[!] Error requesting photo: {e}")
            return None
    
    def _analyze_ph(self, image_path):
        """
        Analyze pH from image using all three color spaces.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            dict: {'rgb': 7.3, 'lab': 7.1, 'hsv': 7.5, 'distances': {...}}
        """
        try:
            from image_req_client.ph_grid_color_reader import ph_from_image
            
            # Get all color spaces from analysis
            result = ph_from_image(
                image_path,
                return_all_color_spaces=True,
                output_dir="same",
                interpolate=True
            )
            
            if result == "NULL":
                print("[!] pH analysis failed")
                return "NULL"
            
            # Print formatted results
            print("\n" + "="*60)
            print("pH ANALYSIS RESULTS")
            print("="*60)
            print(f"  RGB: pH {result['rgb']:>5.1f}  (distance: {result['distances']['rgb']:>6.2f})")
            print(f"  LAB: pH {result['lab']:>5.1f}  (distance: {result['distances']['lab']:>6.2f})")
            print(f"  HSV: pH {result['hsv']:>5.1f}  (distance: {result['distances']['hsv']:>6.2f})")
            print("="*60 + "\n")
            
            return result
                
        except Exception as e:
            print(f"[!] pH analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return "NULL"
    
    def read(self):
        """
        Read pH value by taking and analyzing a photo.
        
        Returns:
            dict: {'rgb': 7.3, 'lab': 7.1, 'hsv': 7.5, 'distances': {...}}
                 or "NULL" if failed
        """
        image_path = self.request_photo()
        if image_path is None:
            return "NULL"
        
        return self._analyze_ph(image_path)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Simple usage - always returns all three color spaces
    ph_read = pHRead()
    ph_read.connect()
    
    result = ph_read.read()
    
    if result != "NULL":
        # Access individual pH values
        print(f"RGB pH: {result['rgb']:.1f}")
        print(f"LAB pH: {result['lab']:.1f}")
        print(f"HSV pH: {result['hsv']:.1f}")
        print(f"\nDistances: {result['distances']}")
    
    ph_read.disconnect()