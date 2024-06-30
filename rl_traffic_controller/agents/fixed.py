import logging

from rl_traffic_controller.environment import Environment

logger = logging.getLogger(__name__)


class FixedAgent:
    """A class simulating a traditional time-based traffic light controller.
    
    Attributes:
        last_action: The last selected action.
    """
    
    def __init__(
        self,
        load_nets: bool = False,
        save: bool = False
    ) -> None:
        self.last_action = 0
    
    def _next_action(self) -> int:
        self.last_action = (self.last_action + 1) % 4
        return self.last_action
    
    def train(
        self,
        env: Environment,
        num_episodes: int = 1,
    ) -> None:
        for i_episode in range(1, num_episodes + 1):
            logger.info(f"Starting episode number {i_episode!r}.")
            env.reset()
            while True:
                action = self._next_action()
                _, _, done = env.step(action)

                if done:
                    break
    
    def demo(self, env: Environment, episodes: int = 1) -> None:
        for i in range(episodes):
            logger.info(f"Starting episode number {i + 1!r}.")
            env.reset()
            done = False
            while not done:
                _, _, done = env.step(self._next_action())
