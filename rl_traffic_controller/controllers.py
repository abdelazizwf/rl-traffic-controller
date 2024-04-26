import logging
import random
import xml.etree.ElementTree as ET

import traci
import traci.exceptions
from PIL import Image

from rl_traffic_controller import consts

logger = logging.getLogger(__name__)


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

        self.phases = consts.SIMULATION_PHASES
        self.amber_phases = consts.SIMULATION_AMBER_PHASES
        
        self.edge_ids = consts.SIMULATION_EDGE_IDS
        self.step_time = step_time
        self.prev_phase = -1
    
    def get_screenshot(self) -> Image.Image:
        """Takes a screenshot of the simulation, saves it to disk, and returns it.
        
        Returns:
            A screenshot of the simulation.
        """
        try:
            traci.gui.screenshot(traci.gui.DEFAULT_VIEW, consts.IMAGE_PATH)
            self.step(1)
            return Image.open(consts.IMAGE_PATH)
        except Exception:
            logger.exception("Error getting screenshot.")

    def set_traffic_phase(self, phase_index: int) -> bool:
        """Sets the traffic phase of the simulation.

        Args:
            phase_index: Index of the phase to be set.
        
        Returns:
            A boolean value indicating if the simulation is not over.
        """
        if self.prev_phase != phase_index:
            traci.trafficlight.setRedYellowGreenState(
                consts.SIMULATION_TRAFFIC_LIGHT_ID,
                self.amber_phases[self.prev_phase]
            )
            f = self.step(3)
            traci.trafficlight.setRedYellowGreenState(
                consts.SIMULATION_TRAFFIC_LIGHT_ID,
                self.phases[phase_index]
            )
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
        commands = [
            "sumo-gui", "--start",
            "--window-size", "1010,882",
            "-c", self.config_file,
            "--step-length", str(self.step_time),
            "--time-to-teleport", str(-1),
        ]

        try:
            traci.start(commands)
            logger.info(f"Started up the simulation from the config file {self.config_file}.")
        except traci.exceptions.TraCIException:
            traci.load(commands[1:])
        except FileNotFoundError:
            logger.error("SUMO is not available.")
            exit(3)
        except Exception:
            logger.exception("Couldn't start the simulation.")
            exit(3)
        
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
        """Changes the probabilities of car flows."""
        tree = ET.parse(consts.SIMULATION_ROUTE_PATH)
        root = tree.getroot()

        for flow in root.findall('.//flow'):
            probability = random.uniform(*consts.SIMULATION_FLOW_PROBABILITY)
            flow.set('probability', str(probability))

        tree.write(consts.SIMULATION_ROUTE_PATH)


class StubController(SUMOController):
    """A class to simulate `SUMOController` for testing only."""
    t = 0
    max_t = 1000
    image = Image.new("RGB", (400, 266))
    
    def get_screenshot(self) -> Image.Image:
        return self.image
    
    def set_traffic_phase(self, phase_index: int) -> bool:
        return self.step(1)
    
    def get_vehicle_count(self) -> int:
        return random.randint(1, 60)
    
    def start(self) -> None:
        self.t = 0
    
    def step(self, seconds: int = 1) -> bool:
        self.t += seconds
        return False if self.t > self.max_t else True
    
    def shutdown(self) -> None:
        return
    
    def tweak_probability(self) -> None:
        return
