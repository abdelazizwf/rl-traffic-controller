from vncdotool import api
from PIL import Image

import logging


logger = logging.getLogger(__name__)


class VNCController:
    """A VNC controller to handle simulation output.
    
    The simulation output is transmitted over a VNC connection. This
    class handles the connection, takes screenshots, and read images.
    
    Attributes:
        vnc_server: The address of the VNC server.
        client: Client connection handler.
        image_path: Path of the output image.
    """
    
    def __init__(self, vnc_server, password, image_path):
        self.vnc_server = vnc_server
        
        try:
            self.client = api.connect(vnc_server, password=password)
            logger.info(f"VNC connection to {vnc_server} established.")
        except Exception:
            logger.exception("VNC connection error.")
            exit()
        
        self.image_path = image_path
    
    def get_image(self, x: int, y: int, w: int, h: int) -> Image.Image:
        """Gets an image of the simulation using the VNC client.
        
        Args:
            x: x position of top-left corner of the simulation window.
            y: y position of top-left corner of the simulation window.
            w: Width of the simulation window.
            h: Height of the simulation window.
        
        Returns:
            The output image as `PIL.Image`.
        """
        try:
            self.client.captureRegion(self.image_path, x, y, w, h)
            logger.debug(f"Captured image at {self.image_path} successfully.")
            return Image.open(self.image_path)
        except Exception:
            logger.exception("Error while capturing image.")
            exit()
    
    def shutdown(self):
        """Disconnect the VNC client."""
        api.shutdown()
        logger.info("VNC connection closed.")
