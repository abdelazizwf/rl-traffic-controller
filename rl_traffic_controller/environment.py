from collections import namedtuple

import numpy as np
import torch
from PIL.Image import Image

from rl_traffic_controller import consts
from rl_traffic_controller.controllers import SUMOController, StubController

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Metrics = namedtuple("Metrics", ["max_queue", "throughput", "avg_delay"])


class Environment:
    """A class to simulate the interactions of the agent with the environment.
    
    This class manages environment-related operations, such as calculating rewards,
    fetching and preparing observations, and interpreting actions taken by the agent.
    
    Attributes:
        simulation_controller: An instance of `SUMOController` to manage the simulation.
        prev_count: A variable holding the last car count to use it for calculating the reward.
        avg_metrics: A variable to hold the average metrics across the entire run.
        episode_metrics: A variable to hold the metrics of the current episode.
    """

    def __init__(self, stub: bool = False) -> None:
        """
        Args:
            stub: Use a stub controller to simulate `SUMOController` for testing.
        """
        if stub is True:
            self.simulation_controller = StubController(
                consts.SIMULATION_CONFIG_PATH
            )
        else:
            self.simulation_controller = SUMOController(
                consts.SIMULATION_CONFIG_PATH
            )
        
        self.prev_count = 0
        
        self.avg_metrics = Metrics([], [], [])
        self.episode_metrics = Metrics([], [], [])
    
    @classmethod
    def image_to_observation(cls, image: Image) -> torch.Tensor:
        """Transforms a raw image into the appropriate observation format.
        
        Args:
            image: The raw image.
        
        Returns:
            The new observation in the form of a `torch.Tensor`.
        """
        image = image.resize(consts.IMAGE_SIZE).convert(consts.IMAGE_FORMAT)
        observation = torch.tensor(
            np.array(image),
            dtype=torch.float32,
            device=device
        )
        
        if len(observation.shape) == 3:
            return observation.permute(2, 0, 1)
        return observation.unsqueeze(0)
    
    def get_observation(self) -> torch.Tensor:
        """Fetches a new observation and converts it to a tensor.
        
        Returns:
            The new observation in the form of a `torch.Tensor`.
        """
        image = self.simulation_controller.get_screenshot()
        return self.image_to_observation(image)
    
    def reset(self) -> torch.Tensor:
        """Resets the environment to prepare for a new episode.
        
        Returns:
            The very first observation.
        """
        self.simulation_controller.tweak_probability()
        self.simulation_controller.start()
        self.prev_count = 0
        self.episode_metrics = Metrics([], [], [])
        return self.get_observation()
    
    def aggregate_metrics(self) -> None:
        """Aggregates the episode metrics to store them in `self.avg_metrics`."""
        n = len(self.episode_metrics.max_queue)
        
        self.avg_metrics.max_queue.append(
            max(self.episode_metrics.max_queue)
        )
        
        self.avg_metrics.throughput.append(
            round(sum(self.episode_metrics.throughput) / n, 3)
        )
        
        self.avg_metrics.avg_delay.append(
            self.episode_metrics.avg_delay[-1]
        )
    
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
            done = not self.simulation_controller.step(16)
        
        # Record metrics
        self.episode_metrics.max_queue.append(self.simulation_controller.get_max_length())
        self.episode_metrics.throughput.append(self.simulation_controller.get_throughput())
        self.episode_metrics.avg_delay.append(
            round(self.simulation_controller.get_avg_delay(), 3)
        )
        
        if done:
            self.aggregate_metrics()
        
        delay_penalty = self.episode_metrics.avg_delay[-1] * 0.01
        max_queue_penalty = self.episode_metrics.max_queue[-1] * 0.01
        throughput_reward = self.episode_metrics.throughput[-1] * 0.05
        
        reward -= (delay_penalty + max_queue_penalty)
        reward += throughput_reward
        
        return state, reward, done
    
    def finish(self) -> None:
        """Cleans up."""
        self.simulation_controller.shutdown()
