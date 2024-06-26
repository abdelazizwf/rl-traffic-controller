import random
from collections import deque, namedtuple


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)
Transition.__doc__ = """\
A data record of the environment transition.

Attributes:
    state: Current state.
    action: Action taken.
    next_state: Resulting observation.
    reward: Reward earned.
"""


class ReplayMemory:
    """A memory buffer to store and sample transitions.
    
    Attributes:
        memory: A `collection.deque` object to hold transitions.
    """

    def __init__(self, capacity: int) -> None:
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
