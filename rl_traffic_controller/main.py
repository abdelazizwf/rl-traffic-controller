import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
from rich import print

from rl_traffic_controller import consts
from rl_traffic_controller.agents.dqn import DQNAgent
from rl_traffic_controller.agents.fixed import FixedAgent
from rl_traffic_controller.environment import Environment, Metrics

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')


def get_agent_class(
    agent_name: str
) -> type[DQNAgent] | type[FixedAgent]:
    """Selects the agent class based on the name.
    
    Args:
        agent_name: The name of the agent.
    
    Returns:
        The agent class selected by the user.
    """
    if agent_name.lower() == "dqn":
        return DQNAgent
    elif agent_name.lower() == "fixed":
        return FixedAgent
    else:
        logger.error("Unknown agent option. Use 'python3.11 run.py --help' to know more.")
        exit(-8)


def plot_metrics(metrics: Metrics) -> None:
    """Plots the environment metrics.

    Args:
        metrics: The environment metrics.
    """
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
    n = len(metrics.avg_delay)
    
    # Calculate moving averages
    window_size = n // 10 if n // 10 > 0 else 1
    ma_y1 = moving_average(metrics.avg_delay, window_size)
    ma_y2 = moving_average(metrics.max_queue, window_size)
    ma_y3 = moving_average(metrics.throughput, window_size)
    
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, figsize=(7, 9))
    
    # Average Delay
    ax1.plot(range(1, n + 1), metrics.avg_delay, color="coral")
    ax1.plot(range(window_size, n + 1), ma_y1, color="gray")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Delay [s]")
    ax1.set_title("Average Delay per Episode")
    
    # Max Queue Length
    ax2.plot(range(1, n + 1), metrics.max_queue, color="coral")
    ax2.plot(range(window_size, n + 1), ma_y2, color="gray")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Queue Length [vehicle]")
    ax2.set_title("Max Queue Length per Episode")
    
    # Average Throughput
    ax3.plot(range(1, n + 1), metrics.throughput, color="coral")
    ax3.plot(range(window_size, n + 1), ma_y3, color="gray")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Throughput [vehicle]")
    ax3.set_title("Average Throughput per Episode")
    
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def train(
    agent_name: str,
    stub: bool = False,
    load_nets: bool = False,
    save: bool = False,
    num_episodes: int = 50,
    image_paths: list[str] = [],
    plot: bool = False,
) -> None:
    """Trains the agent.
    
    Args:
        agent_name: The name of the agent to train.
        stub: A flag to use the stub controller for testing.
        load_nets: Loads a saved network if `True`.
        save: Save the networks if `True`.
        num_episodes: The number of episodes used in training.
        image_paths: A list of image paths representing observations to be used
            to evaluate the agent.
        plot: A flag to enable plotting the metrics after training.
    """
    agent_class = get_agent_class(agent_name)
    agent = agent_class(load_nets=load_nets, save=save)
    
    logger.info(f"Initialized agent '{agent_name}'.")
    
    env = Environment(stub)
    
    logger.info('Started training.')
    
    agent.train(env, num_episodes)
    
    logger.info('Finished training.')
    
    if len(image_paths) > 0:
        evaluate(image_paths, agent)
    
    if plot is True:
        plot_metrics(env.avg_metrics)
    
    env.finish()


def demo(agent_name: str, episodes: int = 1, plot: bool = False) -> None:
    """Runs a demo of the agent.
    
    Args:
        agent_name: The name of the agent to demo.
        episodes: The number of episodes to run the demo for.
        plot: A flag to enable plotting the metrics after the demo.
    """
    agent_class = get_agent_class(agent_name)
    agent = agent_class(load_nets=True)
    
    logger.info(f"Initialized agent '{agent_name}'.")
    
    env = Environment()
    
    logger.info('Starting demo.')
    
    agent.demo(env, episodes)
    
    logger.info('Finished demo.')
    
    if plot is True:
        plot_metrics(env.avg_metrics)
    
    env.finish()


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
        path: The image path.
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
    image_paths: list[str],
    agent_name: str
) -> None:
    """Prints the action chosen by the agent given the input observations.
    
    Args:
        image_paths: A list of image paths representing observations.
        agent_name: The name of the agent to evaluate.
    """
    agent_class = get_agent_class(agent_name)
    agent = agent_class(load_nets=True)
        
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
            except FileNotFoundError:
                logger.warning(f"Image {path!r} does not exist.")
                continue
            except UnidentifiedImageError:
                logger.warning(f"{path!r} is not an image or is a corrupted image.")
                continue
            except Exception:
                logger.warning(f"Error while opening {path!r}.")
                continue
            
            state = Environment.image_to_observation(image)
            
            values, action = agent.evaluate(state)
            
            print(
                f"\nAction values for {path!r} are {values!r}.\n",
                f"The chosen action is [bold blue]phase {action!r}[/bold blue].\n"
            )
            
            result = Image.open(f"data/phase{action}.jpg")
            
            display_results(image, result, values[action], path)
