import socket
import threading
from .ph_test_server import PHTestServer


class ImageServer:
    """
    A server for handling image requests from clients.
    
    This class provides a clean interface for:
    - Starting and stopping the image server
    - Handling client connections
    - Managing camera operations and motor control
    - Getting the server's IP address
    """
    
    def __init__(self, host="0.0.0.0", port=2222):
        """
        Initialize the ImageServer.
        
        Args:
            host (str): Host address to bind to
            port (int): Port to listen on
        """
        self.host = host
        self.port = port
        self.server = PHTestServer(host, port)
        self.running = False
        self.server_socket = None
        self.server_thread = None
        
    def get_ip_address(self):
        """
        Get the server's IP address.
        
        Returns:
            str: IP address of the server
        """
        return self.server.server_ip
    
    def start(self, background=True):
        """
        Start the image server.
        
        Args:
            background (bool): If True, run server in background thread
                              If False, run in current thread (blocking)
        
        Returns:
            bool: True if server started successfully, False otherwise
        """
        if self.running:
            self.server.logger.warning("Server is already running")
            return False
            
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.running = True
            self.server.logger.info(f"Server listening on {self.host}:{self.port}")
            self.server.logger.info(f"Server IP address: {self.get_ip_address()}")
            
            if background:
                self.server_thread = threading.Thread(target=self._run_server, daemon=True)
                self.server_thread.start()
                return True
            else:
                self._run_server()
                return True
                
        except Exception as e:
            self.server.logger.error(f"Failed to start server: {e}")
            self.running = False
            return False
    
    def _run_server(self):
        """Internal method to run the server loop."""
        try:
            while self.running:
                if self.server_socket:
                    try:
                        conn, addr = self.server_socket.accept()
                        self.server.logger.info(f"Connection from {addr}")
                        
                        # Handle client in a separate thread
                        client_thread = threading.Thread(
                            target=self._handle_client_wrapper, 
                            args=(conn,), 
                            daemon=True
                        )
                        client_thread.start()
                        
                    except socket.error as e:
                        if self.running:  # Only log if we're still supposed to be running
                            self.server.logger.error(f"Socket error: {e}")
                        break
                        
        except Exception as e:
            self.server.logger.error(f"Server error: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
    
    def _handle_client_wrapper(self, conn):
        """Wrapper for handling client connections with proper cleanup."""
        try:
            self.server.handle_client(conn)
        except Exception as e:
            self.server.logger.error(f"Error handling client: {e}")
        finally:
            try:
                conn.close()
            except:
                pass
    
    def stop(self):
        """
        Stop the image server.
        
        Returns:
            bool: True if server stopped successfully, False otherwise
        """
        if not self.running:
            self.server.logger.warning("Server is not running")
            return False
            
        try:
            self.running = False
            
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
                
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=5)
                
            self.server.logger.info("Server stopped")
            return True
            
        except Exception as e:
            self.server.logger.error(f"Error stopping server: {e}")
            return False
    
    def is_running(self):
        """
        Check if the server is currently running.
        
        Returns:
            bool: True if server is running, False otherwise
        """
        return self.running
    
    def get_logger(self):
        """
        Get the server's logger instance.
        
        Returns:
            Logger: The server's logger
        """
        return self.server.logger
    
    def __enter__(self):
        """Context manager entry."""
        self.start(background=False)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically stop server."""
        self.stop()


# For backward compatibility, also expose the original classes
from .server import CameraServer
from .ph_test_server import PHTestServer

__all__ = ['ImageServer', 'CameraServer', 'PHTestServer'] 