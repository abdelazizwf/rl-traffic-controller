from functools import partial
from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    """A CNN to predict Q-values.
    
    Attributes:
        layer_stack: An `nn.Sequence` containing the network's layers.
        name: The ID of the network configuration.
    """

    def __init__(self, layer_stack: nn.Sequential, name: str) -> None:
        """
        Args:
            layer_stack: An `nn.Sequence` containing the network's layers.
            name: The ID of the network configuration.
        """
        super(DQN, self).__init__()
        self.layer_stack = layer_stack
        self.name = name
        
        logger.debug(f"Created DQN with stack {name}.\n{self.layer_stack}")

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


# Partial functions to set default arguments for layers.
conv2d = partial(nn.Conv2d, device=device, dtype=torch.float32)
batchnorm = partial(nn.BatchNorm2d, device=device, dtype=torch.float32)
linear = partial(nn.Linear, device=device, dtype=torch.float32)


stacks = {
    "v1": nn.Sequential(
        conv2d(3, 16, 7, 3),
        nn.ReLU(),
        conv2d(16, 64, 5, 2),
        nn.ReLU(),
        conv2d(64, 128, 3, 1),
        nn.ReLU(),
        conv2d(128, 256, 3, 1),
        nn.ReLU(),
        nn.Flatten(),
        linear(256 * 30 * 24, 4)
    ),
    
    "v2": nn.Sequential(
        conv2d(3, 16, 5, 2),
        nn.ReLU(),
        conv2d(16, 64, 3, 1),
        nn.ReLU(),
        conv2d(64, 128, 3, 1),
        nn.ReLU(),
        nn.Flatten(),
        linear(128 * 94 * 61, 4)
    ),
    
    "v3": nn.Sequential(
        conv2d(3, 8, (5, 3), 2),
        nn.ReLU(),
        conv2d(8, 16, (3, 5), 2),
        nn.ReLU(),
        conv2d(16, 64, 5, 1),
        nn.ReLU(),
        nn.Flatten(),
        linear(78848, 4)
    ),
    
    "v4": nn.Sequential(
        conv2d(1, 16, (8, 6), 2),
        nn.ReLU(),
        conv2d(16, 32, (5, 7), 2),
        nn.ReLU(),
        conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.Flatten(),
        linear(78848, 4)
    ),
    
    "v5": nn.Sequential(
        conv2d(1, 16, 8, 4),
        nn.ReLU(),
        conv2d(16, 32, 4, 4),
        nn.ReLU(),
        conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.Flatten(),
        linear(64 * 60, 512),
        nn.ReLU(),
        linear(512, 4),
    ),
    
    "v6": nn.Sequential(
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
    ),
    
    "v7": nn.Sequential(
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
    ),
}
