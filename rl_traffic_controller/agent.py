import random
import math
import logging
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

from rl_traffic_controller.environment import Environment


# Choose cuda if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)
Transition.__doc__ = """\
A data record of the environment transition.
"""


class ReplayMemory:
    """A memory buffer to store and sample transitions.
    
    Attributes:
        memory: A `collection.deque` object to hold transitions.
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of stored transitions.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition.
        
        Args:
            *args: Transition elements.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        """Samples a random number of transitions.
        
        Args:
            batch_size: Number of randomly sampled transitions.
        
        Returns:
            A list of sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQN(nn.Module):
    """A CNN to predict Q-values.
    
    Attributes:
        layer_stack: A sequence containing the network's layers.
    """

    def __init__(self, n_actions: int):
        """
        Args:
            n_actions: Number of output Q values associated with actions.
        """
        super(DQN, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(3, 16, 7, 3),
            nn.ReLU(),
            nn.Conv2d(16, 64, 5, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 30 * 24, n_actions)
        )
        
        logger.debug(f"Created DQN.\n{self.layer_stack}")

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)


class Agent:
    """The Reinforcement Learning agent.
    
    Attributes:
        BATCH_SIZE: The number of transitions sampled from the replay buffer.
        GAMMA: The discount factor of future state-action values.
        EPS_START: The starting value of epsilon.
        EPS_END: The final value of epsilon.
        EPS_DECAY: Controls the rate of exponential decay of epsilon,
            higher means a slower decay.
        TAU: The update rate of the target network.
        LR: The learning rate of the optimizer
        n_actions: The number of available actions.
        optimizer: The optimization function.
        loss_fn: The loss function.
        memory: The replay memory.
        steps_done: A time step counter used to calculate the epsilon threshold.
    """
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    
    def __init__(self, policy_net: nn.Module, target_net: nn.Module):
        """
        Args:
            policy_net: The policy Q-network.
            target_net: The target Q-network.
        """
        self.policy_net = policy_net
        self.target_net = target_net

        self.optimizer = optim.AdamW(policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayMemory(2000)

        self.steps_done = 0
        
        logger.debug(
            "Created agent with hyperparameters:\n" +
            f"Batch size: {self.BATCH_SIZE}\n" +
            f"Gamma: {self.GAMMA}\n" +
            f"Starting epsilon: {self.EPS_START}\n" +
            f"Final epsilon: {self.EPS_END}\n" +
            f"Epsilon exponential decay rate: {self.EPS_DECAY}\n" +
            f"Tau: {self.TAU}\n" +
            f"Learning rate: {self.LR}"
        )
        
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Given a state, selects an action using epsilon greedy policy.
        
        Args:
            state: A state from the environment.
        
        Returns:
            An action index wrapped in a 2D tensor.
        """
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the largest expected reward.
                action = self.policy_net(state).max(1).indices.view(1, 1)
                logger.debug(f"Selected action {action.item()} using the policy.")
        else:
            action = torch.tensor(
                [[random.randint(0, 3)]], device=device, dtype=torch.long
            )
            logger.debug(f"Selected action {action.item()} randomly.")
        
        return action
    
    def optimize_model(self):
        """Performs the model optimization step using batch gradient descent."""
        if len(self.memory) < self.BATCH_SIZE:
            logger.debug(
                f"Memory size ({len(self.memory)}) is less than the batch size ({self.BATCH_SIZE})."
            )
            return
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.bool
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        logger.info(f"Training loss: {loss}")

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        logger.debug("Finished optimization step.")
    
    def run(self, env: Environment, num_episodes: int = 50, checkpoints: bool = False):
        """Performs the main training loops for the given number of episodes.
        
        Args:
            env: The problem environment.
            num_episodes: Number of episodes to use in training.
            checkpoints: A flag to enable saving of the model after each episode.
        """
        for i_episode in range(num_episodes):
            logger.info(f"Starting episode number {i_episode + 1}.")
            
            # Initialize the environment and get its state
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, done = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                
                logger.debug(f"Received reward {reward.item()}.")

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=device
                    ).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = (policy_net_state_dict[key] * self.TAU) + \
                        (target_net_state_dict[key] * (1 - self.TAU))
                self.target_net.load_state_dict(target_net_state_dict)
                
                logger.debug("Updated target network.")

                if done:
                    logger.debug(f"Finished episode {i_episode + 1}.")
                    break
            
            if checkpoints is True:
                try:
                    torch.save(self.target_net.state_dict(), "models/target_net.pt")
                    torch.save(self.policy_net.state_dict(), "models/policy_net.pt")
                    logger.debug("Saved models.")
                except Exception:
                    logger.exception("Couldn't save models.")

        logger.info('Complete.')
