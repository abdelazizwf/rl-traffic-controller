import logging
import math
import pickle
import random
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim

from rl_traffic_controller import consts
from rl_traffic_controller.environment import Environment
from rl_traffic_controller.utils import ReplayMemory, Transition

# Choose cuda if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Partial functions to set default arguments for layers.
conv2d = partial(nn.Conv2d, device=device, dtype=torch.float32)
batchnorm = partial(nn.BatchNorm2d, device=device, dtype=torch.float32)
linear = partial(nn.Linear, device=device, dtype=torch.float32)


class DQN(nn.Module):
    """A CNN to predict Q-values.
    
    Attributes:
        layer_stack: The network's layer architecture.
    """

    def __init__(self) -> None:
        super(DQN, self).__init__()
        self.layer_stack = nn.Sequential(
            conv2d(1, 16, (8, 6), 2),
            nn.ReLU(),
            batchnorm(16),
            conv2d(16, 32, (5, 7), 2),
            nn.ReLU(),
            batchnorm(32),
            conv2d(32, 64, 3, 1),
            nn.ReLU(),
            batchnorm(64),
            nn.Flatten(),
            linear(78848, 4)
        )
        
        logger.debug(f"Created DQN.\n{self.layer_stack!r}")

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements the forward step for the network.
        
        Args:
            x: The input tensor.
        
        Returns:
            The output of the network in the form of a tensor.
        """
        return self.layer_stack(x)


class DQNAgent:
    """The DQN agent.
    
    Attributes:
        BATCH_SIZE: The number of transitions sampled from the replay buffer.
        GAMMA: The discount factor of future state-action values.
        EPS_START: The starting value of epsilon.
        EPS_END: The final value of epsilon.
        EPS_DECAY: Controls the rate of exponential decay of epsilon,
            higher means a slower decay.
        LR: The learning rate of the optimizer.
        policy_net: The policy Q-network.
        target_net: The target Q-network.
        n_actions: The number of available actions.
        optimizer: The optimization function.
        loss_fn: The loss function.
        memory: The replay memory.
        steps_done: A time step counter used to calculate the epsilon threshold.
        cont: Indicate if the training should continue with the loaded networks.
        save: Save the networks if `True`.
    """
    BATCH_SIZE = consts.BATCH_SIZE
    GAMMA = consts.GAMMA
    EPS_START = consts.EPS_START
    EPS_END = consts.EPS_END
    EPS_DECAY = consts.EPS_DECAY
    LR = consts.LR
    TAU = consts.TAU
    
    def __init__(
        self,
        load_nets: bool = False,
        save: bool = False
    ) -> None:
        """
        Args:
            load_nets: Load a saved network if `True`.
            save: Save the networks if `True`.
        """
        self.cont = load_nets
        self.save = save
    
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        
        if load_nets is True:
            try:
                self.policy_net.load_state_dict(torch.load("models/dqn_policy_net.pt"))
                self.target_net.load_state_dict(torch.load("models/dqn_target_net.pt"))
                logger.info("Loaded models successfully")
            except FileNotFoundError:
                logger.error("Saved models were not found. Consider running without '-c' or '--continue'.")
                exit(-2)
            except Exception:
                logger.exception("Failed to load models.")
                exit(-2)
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.steps_done = 0

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayMemory(consts.MEMORY_SIZE)
        
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
                result = self.policy_net(state).max(1)
                value = result.values.view(1, 1).item()
                action = result.indices.view(1, 1)
            logger.debug(f"Selected action {action.item()!r} with value {round(value, 3)!r} using the policy.")
        else:
            action = torch.tensor(
                [[random.randint(0, 3)]], device=device, dtype=torch.long
            )
            logger.debug(f"Selected action {action.item()!r} randomly.")
        
        return action
    
    def optimize_model(self) -> None:
        """Performs the model optimization step using stochastic gradient descent."""
        if len(self.memory) < self.BATCH_SIZE:
            logger.debug(
                f"Memory size ({len(self.memory)!r}) is less than the batch size ({self.BATCH_SIZE!r})."
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
    
    def train(
        self,
        env: Environment,
        num_episodes: int = 1,
    ) -> None:
        """Performs the main training loops for the given number of episodes.
        
        Args:
            env: The problem environment.
            num_episodes: Number of episodes to sample during training.
        """
        if self.cont is True:
            with open("models/dqn_steps.pkl", "rb") as f:
                self.steps_done = pickle.load(f)
            with open("models/dqn_env_avg_metrics.pkl", "rb") as f:
                env.avg_metrics = pickle.load(f)
        
        for i_episode in range(1, num_episodes + 1):
            logger.info(f"Starting episode number {i_episode!r}.")
            
            # Initialize the environment and get its state
            state = env.reset().unsqueeze(0)
            while True:
                action = self.select_action(state)
                observation, reward, done = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                
                logger.debug(f"Received reward {reward.item()!r}.")

                if done:
                    next_state = None
                else:
                    next_state = observation.unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                if done:
                    logger.debug(f"Finished episode {i_episode!r}.")
                    break
            
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
            self.target_net.load_state_dict(target_net_state_dict)
            
            logger.debug("Updated target network.")
            
            if self.save is True:
                try:
                    torch.save(self.target_net.state_dict(), "models/dqn_target_net.pt")
                    torch.save(self.policy_net.state_dict(), "models/dqn_policy_net.pt")
                    with open("models/dqn_steps.pkl", "wb") as f:
                        pickle.dump(self.steps_done, f)
                    with open("models/dqn_env_avg_metrics.pkl", "wb") as f:
                        pickle.dump(env.avg_metrics, f)
                    logger.debug("Saved models.")
                except Exception:
                    logger.exception("Couldn't save models.")
                    exit(-4)
    
    @torch.inference_mode()
    def evaluate(self, state: torch.Tensor) -> tuple[list[float], int]:
        """Returns the action values of the agent given the input state.
        
        Args:
            state: The input state.
        
        Returns:
            A list of the action values and the index of the chosen action (i.e.
            the action with the maximum value).
        """
        actions = self.target_net(state.unsqueeze(0))
        actions = actions.squeeze(0).tolist()
        return [round(x, 3) for x in actions], actions.index(max(actions))
    
    def demo(self, env: Environment, episodes: int = 1) -> None:
        """Uses the target network to always select the best action.
        
        Args:
            env: The problem environment.
            episodes: The number of episodes to run the demo.
        """
        for i in range(episodes):
            logger.info(f"Starting episode number {i + 1!r}.")
            state = env.reset()
            done = False
            while not done:
                _, action = self.evaluate(state)
                state, _, done = env.step(action)
