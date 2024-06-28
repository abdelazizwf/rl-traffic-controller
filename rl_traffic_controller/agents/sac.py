import logging
import pickle
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_traffic_controller import consts
from rl_traffic_controller.environment import Environment
from rl_traffic_controller.utils import ReplayMemory, Transition

# Choose cuda if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

# Partial functions to set default arguments for layers.
conv2d = partial(nn.Conv2d, device=device, dtype=torch.float32)
batchnorm = partial(nn.BatchNorm2d, device=device, dtype=torch.float32)
linear = partial(nn.Linear, device=device, dtype=torch.float32)


class Actor(nn.Module):
    """A CNN to choose an action.
    
    Attributes:
        layer_stack: The architecture of the network.
    """
    
    def __init__(self) -> None:
        super(Actor, self).__init__()
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
            linear(78848, 4),
        )
        
        logger.debug(f"Created Actor.\n{self.layer_stack!r}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements the forward step for the network.
        
        Args:
            x: The input tensor.
        
        Returns:
            The output of the network in the form of a tensor.
        """
        return self.layer_stack(x)


class Critic(nn.Module):
    """A CNN to estimate the Q-value of actions.
    
    Attributes:
        layer_stack: The architecture of the network.
    """
    
    def __init__(self) -> None:
        super(Critic, self).__init__()
        self.layer_stack = nn.Sequential(
            conv2d(1, 16, 8, 4),
            nn.ReLU(),
            batchnorm(16),
            conv2d(16, 32, 4, 4),
            nn.ReLU(),
            batchnorm(32),
            conv2d(32, 64, 3, 1),
            nn.ReLU(),
            batchnorm(64),
            nn.Flatten(),
            linear(64 * 60, 512),
            nn.ReLU(),
            linear(512, 4),
        )
        
        logger.debug(f"Created Critic.\n{self.layer_stack!r}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements the forward step for the network.
        
        Args:
            x: The input tensor.
        
        Returns:
            The output of the network in the form of a tensor.
        """
        return self.layer_stack(x)


class SACAgent:
    """The SAC agent.
    
    Attributes:
        BATCH_SIZE: The number of transitions sampled from the replay buffer.
        GAMMA: The discount factor of future state-action values.
        LR: The learning rate of the optimizer.
        TAU: The soft update rate of the target networks.
        actor: The policy network that chooses an action.
        local_critic_1: A critic network to estimate the values of actions.
        local_critic_2: Another critic network to estimate the values of actions.
        target_critic_1: A fixed target for `local_critic_1`.
        target_critic_2: A fixed target for `local_critic_2`.
        n_actions: The number of available actions.
        actor_optimizer: The optimizer for the actor.
        critic_1_optimizer: The optimizer for `local_critic_1`.
        critic_2_optimizer: The optimization for the `local_critic_2`.
        loss_fn: The loss function.
        memory: The replay memory.
        cont: Indicate if the training should continue with the loaded networks.
        save: Save the networks if `True`.
    """
    BATCH_SIZE = consts.BATCH_SIZE
    GAMMA = consts.GAMMA
    LR = consts.LR
    TAU = consts.TAU
    
    def __init__(
        self,
        initial_alpha: float = 0.2,
        load_nets: bool = False,
        save: bool = False
    ) -> None:
        """
        Args:
            load_nets: Load a saved network if `True`.
            save: Save the networks if `True`.
            initial_alpha: The initial value of the entropy parameter.
        """
        self.cont = load_nets
        self.save = save
    
        self.actor = Actor().to(device)
        self.local_critic_1 = Critic().to(device)
        self.local_critic_2 = Critic().to(device)
        self.target_critic_1 = Critic().to(device)
        self.target_critic_2 = Critic().to(device)
        
        if load_nets is True:
            try:
                self.actor.load_state_dict(torch.load("models/sac_actor_net.pt"))
                self.local_critic_1.load_state_dict(torch.load("models/sac_local_critic1_net.pt"))
                self.local_critic_2.load_state_dict(torch.load("models/sac_local_critic2_net.pt"))
                self.target_critic_1.load_state_dict(torch.load("models/sac_target_critic1_net.pt"))
                self.target_critic_2.load_state_dict(torch.load("models/sac_target_critic2_net.pt"))
                logger.info("Loaded models successfully")
            except FileNotFoundError:
                logger.error("Saved models were not found. Consider running without '-c' or '--continue'.")
                exit(-2)
            except Exception:
                logger.exception("Failed to load models.")
                exit(-2)
        else:
            self.target_critic_1.load_state_dict(self.local_critic_1.state_dict())
            self.target_critic_2.load_state_dict(self.local_critic_2.state_dict())

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.LR, amsgrad=True)
        self.critic_1_optimizer = optim.AdamW(self.local_critic_1.parameters(), lr=self.LR, amsgrad=True)
        self.critic_2_optimizer = optim.AdamW(self.local_critic_2.parameters(), lr=self.LR, amsgrad=True)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayMemory(consts.MEMORY_SIZE)
        
        # Automatic entropy tuning
        self.target_entropy = -0.98 * torch.log(1 / torch.tensor(4).float())
        self.log_alpha = torch.tensor(torch.log(torch.tensor(initial_alpha)), requires_grad=True)
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=3e-4)
        
        logger.debug(
            "Created agent with hyperparameters:\n" +
            f"Batch size: {self.BATCH_SIZE}\n" +
            f"Gamma: {self.GAMMA}\n" +
            f"Learning rate: {self.LR}" +
            f"Tau: {self.TAU}"
        )
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
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

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)
        done_batch = torch.FloatTensor([s is None for s in batch.next_state]).unsqueeze(1)
        next_state_batch = torch.cat(
            [b if b is not None else torch.zeros(1, 1, 133, 200) for b in batch.next_state]
        )
        
        # Update Q-networks
        with torch.no_grad():
            next_action_probs = F.softmax(self.actor(next_state_batch), dim=-1)
            next_log_probs = F.log_softmax(self.actor(next_state_batch), dim=-1)
            next_q1 = self.target_critic_1(next_state_batch)
            next_q2 = self.target_critic_2(next_state_batch)
            next_q = torch.min(next_q1, next_q2)
            v_next = (next_action_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            q_target = reward_batch + (1 - done_batch) * self.GAMMA * v_next
        
        q1 = self.local_critic_1(state_batch).gather(1, action_batch)
        q2 = self.local_critic_2(state_batch).gather(1, action_batch)
        q1_loss = self.loss_fn(q1, q_target)
        q2_loss = self.loss_fn(q2, q_target)
        
        self.critic_1_optimizer.zero_grad()
        q1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        q2_loss.backward()
        self.critic_2_optimizer.step()

        # Update policy network
        action_probs = F.softmax(self.actor(state_batch), dim=-1)
        log_probs = F.log_softmax(self.actor(state_batch), dim=-1)
        q1 = self.local_critic_1(state_batch)
        q2 = self.local_critic_2(state_batch)
        q = torch.min(q1, q2)
        policy_loss = (action_probs * (self.alpha * log_probs - q)).sum(dim=1).mean()
        
        logger.info(f"Training loss: {policy_loss.item()}")

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature parameter
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
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
            with open("models/sac_env_avg_metrics.pkl", "rb") as f:
                env.avg_metrics = pickle.load(f)
        
        for i_episode in range(1, num_episodes + 1):
            logger.info(f"Starting episode number {i_episode!r}.")
            
            # Initialize the environment and get its state
            state = env.reset().unsqueeze(0)
            while True:
                action = self.actor(state).argmax(1)
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
            
            # Soft update of the target networks' weights
            # θ′ ← τ θ + (1 −τ )θ′
            local_critic_1_state_dict = self.local_critic_1.state_dict()
            local_critic_2_state_dict = self.local_critic_2.state_dict()
            target_critic_1_state_dict = self.target_critic_1.state_dict()
            target_critic_2_state_dict = self.target_critic_2.state_dict()
            for key_1, key_2 in zip(local_critic_1_state_dict, local_critic_2_state_dict):
                target_critic_1_state_dict[key_1] = ((local_critic_1_state_dict[key_1] * self.TAU) +
                                                     target_critic_1_state_dict[key_1] * (1 - self.TAU))
                target_critic_2_state_dict[key_2] = ((local_critic_2_state_dict[key_2] * self.TAU) +
                                                     target_critic_2_state_dict[key_2] * (1 - self.TAU))
            self.target_critic_1.load_state_dict(target_critic_1_state_dict)
            self.target_critic_2.load_state_dict(target_critic_2_state_dict)
            logger.debug("Updated target network.")
            
            if self.save is True:
                try:
                    torch.save(self.actor.state_dict(), "models/sac_actor_net.pt")
                    torch.save(self.local_critic_1.state_dict(), "models/sac_local_critic1_net.pt")
                    torch.save(self.local_critic_2.state_dict(), "models/sac_local_critic2_net.pt")
                    torch.save(self.target_critic_1.state_dict(), "models/sac_target_critic1_net.pt")
                    torch.save(self.target_critic_2.state_dict(), "models/sac_target_critic2_net.pt")
                    with open("models/sac_env_avg_metrics.pkl", "wb") as f:
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
        action = self.actor(state.unsqueeze(0)).argmax(1).item()
        action_values_1 = self.target_critic_1(state.unsqueeze(0))
        action_values_2 = self.target_critic_2(state.unsqueeze(0))
        action_values = torch.min(action_values_1, action_values_2)
        action_values = action_values.squeeze(0).tolist()
        return [round(x, 3) for x in action_values], action
    
    def demo(self, env: Environment, episodes: int = 1) -> None:
        """Uses the target network to always select the best action.
        
        Args:
            env: The problem environment.
        """
        for i in range(episodes):
            logger.info(f"Starting episode number {i + 1!r}.")
            state = env.reset()
            done = False
            while not done:
                _, action = self.evaluate(state)
                state, _, done = env.step(action)
