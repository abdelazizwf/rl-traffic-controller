import torch
import logging
import numpy as np
from PIL import Image

from rl_traffic_controller.agent import DQN, Agent
from rl_traffic_controller.environment import Environment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def init_agent(load_nets: bool = False) -> Agent:
    """Initializes the agent with a new or a previously saved network.
    
    Args:
        load_nets: Loads a saved network if `True`.
    
    Returns:
        A new `Agent` instance.
    """
    n_actions = 4
    
    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    
    if load_nets is True:
        try:
            policy_net.load_state_dict(torch.load("models/policy_net.pt"))
            target_net.load_state_dict(torch.load("models/target_net.pt"))
            logger.info("Loaded models successfully")
        except Exception:
            logger.exception("Failed to load models.")
            exit()
    else:
        target_net.load_state_dict(policy_net.state_dict())
    
    return Agent(policy_net, target_net)


def train(
    load_nets: bool = False,
    num_episodes: int = 50,
    checkpoints: bool = True
) -> None:
    """Trains the agent.
    
    Args:
        load_nets: Loads a saved network if `True`.
        num_episodes: The number of episodes used in training.
        checkpoints: A flag to enable saving the network after each episode.
    """
    agent = init_agent(load_nets)
    
    env = Environment()
    
    logger.info('Started training.')
    
    agent.train(env, num_episodes, checkpoints)
    
    logger.info('Finished training.')


def evaluate(image_paths: list[str], agent: Agent | None = None) -> None:
    """Prints the action chosen by the agent given the input observations.
    
    Args:
        image_paths: A list of image paths representing observations.
        agent: An optional agent that is already initialized.
    """
    if agent is None:
        agent = init_agent(True)
    
    for path in image_paths:
        try:
            image = Image.open(path)
        except Exception:
            logger.exception(f"Failed to open image {path}.")
            continue
        
        image = image.resize((220, 186)).convert("RGB")
        
        state = torch.tensor(
            np.array(image),
            dtype=torch.float32,
            device=device
        ).permute(2, 0, 1)
        
        values, action = agent.evaluate(state)
        
        print(
            f"\nAction values for {path} are {values}.\n",
            f"The chosen action's index is {action}.\n"
        )
