
from homework3 import ValueNetwork
from homework3 import PolicyNetwork
from homework3 import PPOAgent
from homework3 import Hw3Env
import torch
import time


# load model
value_network = ValueNetwork()
value_network.load_state_dict(torch.load('HW3/value_network.pth'))
value_network.eval()

policy_network = PolicyNetwork()
policy_network.load_state_dict(torch.load('HW3/policy_network.pth'))
policy_network.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
value_network.to(device)
policy_network.to(device)

agent = PPOAgent(value_network, policy_network)

# Test the agent in the simulation environment

env = Hw3Env(render_mode="gui")

# calculate the reward

n_episodes = 100
batch_size = 32

for episode in range(n_episodes):
    env.reset()
    done = False
    cum_reward = 0.0
    start = time.time()
    while not done:
        state = env.high_level_state()
        action = agent.get_action(state)
        next_state, reward, is_terminal, is_truncated = env.step(action)
        done = is_terminal or is_truncated
        cum_reward += reward
    end = time.time()
    print(f"Episode={episode}, reward={cum_reward}")
    print(f"RF={env.data.time/(end-start):.2f}")




