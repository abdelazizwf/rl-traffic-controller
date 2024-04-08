import torch
import numpy as np

from rl_traffic_controller.controllers import SUMOController


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Environment:

    def __init__(self) -> None:
        self.simulation_controller = SUMOController(
            r"./simulation/v1.sumocfg"
        )
        self.prev_count = 0
    
    def _get_state(self) -> torch.Tensor:
        image = self.simulation_controller.get_screenshot().resize((220, 186))
        return torch.tensor(
            np.array(image.convert("RGB")),
            dtype=torch.float32,
            device=device
        ).permute(2, 0, 1)
    
    def reset(self) -> torch.Tensor:
        self.simulation_controller.tweak_probability()
        self.simulation_controller.start()
        return self._get_state()
    
    def step(self, action) -> tuple[torch.Tensor, int, bool]:
        done = not self.simulation_controller.set_traffic_phase(action)
        
        state = self._get_state()
        
        count = self.simulation_controller.get_vehicle_count()
        if count > self.prev_count:
            reward = -1
        elif count < self.prev_count:
            reward = 1
        else:
            reward = 0
        
        self.prev_count = count
        
        if not done:
            self.simulation_controller.step(16)
        
        return state, reward, done

    def destroy(self) -> None:
        pass
