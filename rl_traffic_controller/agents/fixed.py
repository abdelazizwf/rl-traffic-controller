from rl_traffic_controller.environment import Environment


class FixedAgent:
    """A class simulating a traditional time-based traffic light controller."""
    
    def __init__(
        self,
        load_nets: bool = False,
        save: bool = False
    ) -> None:
        self.current_phase = 0
    
    def _next_phase(self) -> int:
        self.current_phase = (self.current_phase + 1) % 4
        return self.current_phase
    
    def train(
        self,
        env: Environment,
        num_episodes: int = 1,
    ) -> None:
        for _ in range(num_episodes):
            env.reset()
            while True:
                action = self._next_phase()
                _, _, done = env.step(action)

                if done:
                    break
    
    def demo(self, env: Environment) -> None:
        env.reset()
        done = False
        while not done:
            _, _, done = env.step(self._next_phase())
