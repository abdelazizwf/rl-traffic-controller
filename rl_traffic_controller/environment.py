import torch
import numpy as np

from rl_traffic_controller.controllers import SUMOController
from rl_traffic_controller import consts


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Environment:
    """A class to simulate the interactions of the agent with the environment.
    
    This class manages environment-related operations, such as calculating rewards,
    fetching and preparing observations, and interpreting actions taken by the agent.
    
    Attributes:
        simulation_controller: An instance of `SUMOController` to manage the simulation.
        prev_count: A variable holding the last car count to use it for calculating the
            reward.
    """

    def __init__(self) -> None:
        self.simulation_controller = SUMOController(
            r"./simulation/v1.sumocfg"
        )
        self.prev_count = 0
    
    def get_observation(self) -> torch.Tensor:
        """Fetches a new observation and converts it to a tensor.
        
        Returns:
            The new observation in the form of a `torch.Tensor`.
        """
        image = self.simulation_controller.get_screenshot().resize(consts.IMAGE_SIZE)
        return torch.tensor(
            np.array(image.convert("RGB")),
            dtype=torch.float32,
            device=device
        ).permute(2, 0, 1)
    
    def reset(self) -> torch.Tensor:
        """Resets the environment to prepare for a new episode.
        
        Returns:
            The very first observation.
        """
        self.simulation_controller.tweak_probability()
        self.simulation_controller.start()
        return self.get_observation()
    
    def step(self, action: int) -> tuple[torch.Tensor, int, bool]:
        """Applies the chosen action to the environment and returns the results.
        
        Args:
            action: The index of the action chosen by the agent.
        
        Returns:
            The resulting observation, the reward, and a flag indicating if the episode
            is finished or not.
        """
        done = not self.simulation_controller.set_traffic_phase(action)
        
        state = self.get_observation()
        
        count = self.simulation_controller.get_vehicle_count()
        if count > self.prev_count:
            reward = -1
        elif count < self.prev_count:
            reward = 1
        else:
            reward = 0
        
        self.prev_count = count
        
        if not done:
            self.simulation_controller.step(16)
        
        return state, reward, done
