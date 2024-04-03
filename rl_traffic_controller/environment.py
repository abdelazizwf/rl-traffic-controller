import torch
import numpy as np

from rl_traffic_controller.controllers import VNCController


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Environment:

    def __init__(self):
        self.t = 0
        self.vnc_controller = VNCController(
            "localhost::5901",
            "abcabc",
            "data/simulation.png"
        )
    
    def _get_state(self) -> torch.Tensor:
        w, h = 1100, 960 - 30
        x, y = 800, 0 + 90
        
        image = self.vnc_controller.get_image(x, y, w, h)
        image = image.resize((220, 186))
        
        return torch.tensor(
            np.array(image),
            dtype=torch.float32,
            device=device
        ).permute(2, 0, 1)
    
    def reset(self) -> torch.Tensor:
        return self._get_state()
    
    def step(self, action):
        state = self._get_state()
        if self.t == 10:
            self.t = 0
            done = True
        else:
            self.t += 1
            done = False
        return state, 1, done

    def destroy(self):
        self.vnc_controller.shutdown()
