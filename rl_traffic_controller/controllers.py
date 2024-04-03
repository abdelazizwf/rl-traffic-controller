from vncdotool import api
from PIL import Image


class VNCController:
    """A VNC controller to handle simulation output.
    
    The simulation output is transmitted over a VNC connection. This
    class handles the connection, takes screenshots, and read images.
    
    Attributes:
        client: Client connection handler.
        image_path: Path of the output image.
    """
    # UPDATE dot variables
    client = api.connect("localhost::5901", password="abcabc")
    image_path = "data/simulation.png"
    
    @classmethod
    def get_image(cls, x: int, y: int, w: int, h: int) -> Image.Image:
        """Gets an image of the simulation using the VNC client.
        
        Args:
            x: x position of top-left corner of the simulation window.
            y: y position of top-left corner of the simulation window.
            w: Width of the simulation window.
            h: Height of the simulation window.
        
        Returns:
            The output image as `PIL.Image`.
        """
        cls.client.captureRegion(cls.image_path, x, y, w, h)
        return Image.open(cls.image_path)
    
    @classmethod
    def shutdown(cls):
        """Disconnect the VNC client."""
        api.shutdown()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    w, h = 1100, 960 - 30
    x, y = 800, 0 + 90
    
    image = VNCController.get_image(x, y, w, h)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
    
    VNCController.shutdown()
