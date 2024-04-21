import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from rl_traffic_controller import consts
from rl_traffic_controller.agent import Agent
from rl_traffic_controller.environment import Environment

logger = logging.getLogger(__name__)


def train(
    stack_name: str,
    load_nets: bool = False,
    save: bool = False,
    num_episodes: int = 50,
    image_paths: list[str] = []
) -> None:
    """Trains the agent.
    
    Args:
        stack_name: ID of the layer stack.
        load_nets: Loads a saved network if `True`.
        save: Save the networks if `True`.
        num_episodes: The number of episodes used in training.
        checkpoints: A flag to enable saving the network after each episode.
        image_paths: A list of image paths representing observations to be used
            to evaluate the agent.
    """
    agent = Agent(stack_name, load_nets, save)
    
    env = Environment()
    
    logger.info('Started training.')
    
    agent.train(env, num_episodes)
    
    logger.info('Finished training.')
    
    if len(image_paths) > 0:
        evaluate(stack_name, image_paths, agent)


def demo(stack_name: str) -> None:
    """Runs a demo of the agent.
    
    Args:
        stack_name: ID of the layer stack.
    """
    agent = Agent(stack_name, True)
    
    env = Environment()
    
    logger.info('Starting demo.')
    
    agent.demo(env)
    
    logger.info('Finished demo.')


def display_results(
    image: Image.Image,
    result: Image.Image,
    action_value: float,
    path: str,
) -> None:
    """Displays the input image and the chosen action.
    
    Args:
        image: The input image.
        result: The image of the chosen action.
        action_value: The Q value of the chosen action.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
    ax1.imshow(np.array(image))
    ax1.axis("off")
    ax1.set_title(path)
    
    ax2.imshow(np.array(result))
    ax2.axis("off")
    ax2.set_title(f"Q Value: {str(action_value)}")
    
    fig.tight_layout()
    
    plt.show()


def evaluate(
    stack_name: str,
    image_paths: list[str],
    agent: Agent | None = None
) -> None:
    """Prints the action chosen by the agent given the input observations.
    
    Args:
        stack_name: ID of the layer stack.
        image_paths: A list of image paths representing observations.
        agent: An optional agent that is already initialized.
    """
    if agent is None:
        agent = Agent(stack_name, True)
        
    for path in image_paths:
        is_dir = os.path.isdir(path)
        
        paths = []
        if is_dir is True:
            for file in os.listdir(path):
                _, ext = os.path.splitext(file)
                if ext in consts.IMAGE_EXTENSIONS:
                    paths.append(
                        os.path.join(path, file)
                    )
        else:
            paths.append(path)
        
        for path in paths:
            try:
                image = Image.open(path)
            except Exception:
                logger.exception(f"Failed to open image {path}.")
                continue
            
            state = Environment.image_to_observation(image)
            
            values, action = agent.evaluate(state)
            
            print(
                f"\nAction values for {path} are {values}.\n",
                f"The chosen action's index is {action}.\n"
            )
            
            result = Image.open(f"data/phase{action}.jpg")
            
            display_results(image, result, values[action], path)
