
from HW2 import DQNAgent
from homework2 import Hw2Env
import torch


# load model from model.pth
agent = DQNAgent(6)
agent.load('HW2/model.pth')
agent.set_epsilon_min()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent.q_network.to(device)

# Test the agent in the simulation environment

N_ACTIONS = 8
env = Hw2Env(n_actions=N_ACTIONS, render_mode="gui")
for episode in range(100):
    env.reset()
    done = False
    cum_reward = 0.0
    while not done:
        state = torch.tensor(env.high_level_state(), dtype=torch.float32).to(device)
        action = agent.get_action(state)
        state, reward, is_terminal, is_truncated = env.step(action)
        done = is_terminal or is_truncated
        cum_reward += reward
    print(f"Episode={episode}, reward={cum_reward}")