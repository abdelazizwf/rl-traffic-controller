import torch

from rl_traffic_controller.agent import DQN, Agent
from rl_traffic_controller.environment import Environment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(
    load_nets: bool = False,
    num_episodes: int = 50,
    checkpoints: bool = True
):
    n_actions = 4
    
    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    
    env = Environment()
    
    if load_nets is True:
        policy_net.load_state_dict(torch.load("models/policy_net.pt"))
        target_net.load_state_dict(torch.load("models/target_net.pt"))
    else:
        target_net.load_state_dict(policy_net.state_dict())
    
    agent = Agent(policy_net, target_net)
    
    agent.run(env, num_episodes, checkpoints)


if __name__ == "__main__":
    run()
