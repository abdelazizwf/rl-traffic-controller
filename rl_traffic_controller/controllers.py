import traci.exceptions
from vncdotool import api
from PIL import Image
import traci

import logging
import random
import xml.etree.ElementTree as ET


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
    
    def __init__(
        self,
        vnc_server: str,
        password: str,
        image_path: str
    ) -> None:
        """
        Args:
            vnc_server: The address of the VNC server.
            password: The password of the VNC server.
            image_path: The path to temporarily save the
                screenshot of the simulation.
        """
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
    
    def shutdown(self) -> None:
        """Disconnect the VNC client."""
        api.shutdown()
        logger.info("VNC connection closed.")


class SUMOController:
    """Class to manage SUMO simulation.

    This class provides methods to control SUMO simulations, such as setting traffic
    phases and retrieving vehicle counts.
    
    Examples:
        Using `SUMOController` to run a simulation and print the number
        of vehicles::
        
            simulation = SUMOController(r"./simulation/sumo_config.sumocfg")
            simulation.start()
            done = True
            while done:
                done = simulation.step(seconds=1)
                print(simulation.get_vehicle_count())
            simulation.shutdown()

    Attributes:
        config_file (str): The file path of the SUMO configuration file.
        phase_states (list): List of strings representing traffic light phases.
        edge_ids (list): List of edge IDs for vehicle count retrieval.
        step_time (float): The time length of each step in seconds [0.001, 1].
        prev_phase (int): The index of the last phase activated.
    """

    def __init__(self, config_file: str, step_time: float = 1.0) -> None:
        """
        Args:
            config_file: The file path of the SUMO configuration file.
            step_time: The time length of each step in seconds [0.001, 1].
        """
        self.config_file = config_file

        self.phases = [
            "GGGGGrrrrrrrGGGGGrrrrrrr",
            "rrrrrGrrrrrrrrrrrGrrrrrr",
            "rrrrrrGGGGGrrrrrrrGGGGGr",
            "rrrrrrrrrrrGrrrrrrrrrrrG",
        ]
        self.amber_phases = [
            "yyyyyrrrrrrryyyyyrrrrrrr",
            "rrrrryrrrrrrrrrrryrrrrrr",
            "rrrrrryyyyyrrrrrrryyyyyr",
            "rrrrrrrrrrryrrrrrrrrrrry"
        ]
        
        self.edge_ids = ["E2TL", "N2TL", "S2TL", "W2TL"]
        self.step_time = step_time
        self.prev_phase = -1

    def set_traffic_phase(self, phase_index: int) -> bool:
        """Sets the traffic phase of the simulation.

        Args:
            phase_index: Index of the phase to be set.
        
        Returns:
            A boolean value indicating if the simulation is not over.
        """
        if self.prev_phase != phase_index:
            traci.trafficlight.setRedYellowGreenState("TL", self.amber_phases[self.prev_phase])
            f = self.step(3)
            traci.trafficlight.setRedYellowGreenState("TL", self.phases[phase_index])
            self.prev_phase = phase_index
        else:
            f = self.step(3)
            
        logger.debug(f"Set the traffic light to phase {phase_index}: {self.phases[phase_index]}.")

        return f

    def get_vehicle_count(self) -> int:
        """Retrieves the number of vehicles on each edge and prints the result.
        
        Returns:
            The number of vehicles headed towards the intersection.
        """
        return sum(
            [traci.edge.getLastStepVehicleNumber(x) for x in self.edge_ids]
        )
    
    def start(self) -> None:
        """Starts the simulation using the provided config file."""
        try:
            traci.start(
                [
                    "sumo-gui",
                    "-c", self.config_file,
                    "--step-length", str(self.step_time)
                ]
            )
            logger.info(f"Started up the simulation from the config file {self.config_file}.")
        except traci.exceptions.TraCIException:
            traci.load(["-c", self.config_file, "--step-length", str(self.step_time)])
    
    def step(self, seconds: int = 1) -> bool:
        """Runs the simulation for a given amount of time.
        
        Args:
            seconds: Number of seconds the simulation is run.
        
        Returns:
            A boolean value indicating if the simulation is not over.
        """
        steps = int(seconds / self.step_time)
        
        for _ in range(steps):
            if traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
            else:
                return False
        
        return True
    
    def shutdown(self) -> None:
        """Closes the simulation."""
        traci.close()
        logger.info("Simulation closed.")
    
    def tweak_probability(self) -> None:
        """Change the probabilities of car flows."""
        file_path = "simulation/v1.rou.xml"
        
        tree = ET.parse(file_path)
        root = tree.getroot()

        for flow in root.findall('.//flow'):
            probability = random.uniform(0.05, 0.50)
            flow.set('probability', str(probability))

        tree.write(file_path)
