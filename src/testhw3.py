
from homework3 import ValueNetwork
from homework3 import PolicyNetwork
from homework3 import PPOAgent
from homework3 import Hw3Env
import torch
import time
import os


value_networks = [f for f in os.listdir("HW3/network-snapshots/") if f.startswith("value_network")]
policy_networks = [f for f in os.listdir("HW3/network-snapshots/") if f.startswith("policy_network")]

value_networks = sorted(value_networks, key=lambda f: os.path.getmtime(f"HW3/network-snapshots/{f}"), reverse=True)
policy_networks = sorted(policy_networks, key=lambda f: os.path.getmtime(f"HW3/network-snapshots/{f}"), reverse=True)

latest_value_network = f"HW3/network-snapshots/{value_networks[0]}"
latest_policy_network = f"HW3/network-snapshots/{policy_networks[0]}"

# load model
value_network = ValueNetwork()
value_network.load_state_dict(torch.load(latest_value_network))
value_network.eval()

policy_network = PolicyNetwork()
policy_network.load_state_dict(torch.load(latest_policy_network))
policy_network.eval()

agent = PPOAgent(value_network, policy_network)

# Test the agent in the simulation environment

env = Hw3Env(render_mode="gui")

# calculate the reward

n_episodes = 100

for episode in range(n_episodes):
    env.reset()
    done = False
    cum_reward = 0.0
    start = time.time()
    while not done:
        state = env.high_level_state().unsqueeze(0)
        action = policy_network(state)

        action = action[0]  # convert (1, 4) to (4,)
        mean = action[[0, 2]]
        std = action[[1, 3]]
        cov_matrix = torch.diag(std ** 2)
        dist = torch.distributions.MultivariateNormal(mean, cov_matrix)
        action = dist.sample()

        next_state, reward, is_terminal, is_truncated = env.step(action)
        done = is_terminal or is_truncated
        cum_reward += reward
    end = time.time()
    print(f"Episode={episode}, reward={cum_reward}")
    print(f"RF={env.data.time/(end-start):.2f}")




