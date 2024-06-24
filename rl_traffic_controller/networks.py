from functools import partial
from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Partial functions to set default arguments for layers.
conv2d = partial(nn.Conv2d, device=device, dtype=torch.float32)
batchnorm = partial(nn.BatchNorm2d, device=device, dtype=torch.float32)
linear = partial(nn.Linear, device=device, dtype=torch.float32)


class DQN(nn.Module):
    """A CNN to predict Q-values."""

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
