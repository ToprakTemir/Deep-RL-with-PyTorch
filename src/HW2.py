# ### Deep Q Network for finite action space
# - DQN using Double Q learning with target network, replay buffer, target network, and epsilon greedy policy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

import time
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") # for use in server

from homework2 import Hw2Env



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ### Define the network
# ##### it will take the high level state of the simulation (ee_pos, obj_pos, goal_post) as input and give the Q values for each action as output 

class QNetwork(nn.Module):
    def __init__(self, state_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 8)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Transition class to store in the replay buffer

class Transition:
    def __init__(self, state, action, reward, next_state):
        self.state = torch.tensor(state, dtype=torch.float32).to(device)
        self.action = torch.tensor(action, dtype=torch.long).to(device)
        self.reward = torch.tensor(reward, dtype=torch.float32).to(device)
        self.next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

    def to(self, device):
        self.state = self.state.to(device)
        self.action = self.action.to(device)
        self.reward = self.reward.to(device)
        self.next_state = self.next_state.to(device)
        return self


# ### Define the agent

# HYPERPARAMETERS:
# - lr: learning rate, used in the optimizer
# - gamma: discount factor that describes how much the agent values future rewards
# - epsilon: the probability of selecting a random action instead of the greedy optimal one
# - decay_rate: the rate of exponential decay for epsilon
# - decay_iter: the number of iterations after which epsilon will be decayed
# - target_network_update_period: the number of iterations after which the target network will be updated with the current network

class DQNAgent:
    def __init__(self, state_dim, lr=0.001, gamma=0.99, epsilon_min=0.1, epsilon=1.0, decay_rate=0.999, decay_iter=10):
        self.q_network = QNetwork(state_dim).to(device)
        self.target_network = QNetwork(state_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.decay_iter = decay_iter

        self.replay_buffer = deque(maxlen=10000)

        self.update_count = 0
        self.target_network_update_period = 1000
        
    # epsilon greedy policy to select the action
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 8)
        else:
            return torch.argmax(self.q_network(state)).item()

    # Since we will use Double Q learning, we want to use different networks to compute the reward and the target
    # We will use the target network for the target estimation, and as input to the target network we will give the argmax action of the current network, decoupling the action selection (now done by the current network) from the target estimation (done by the target network)
    # def estimate_target(self, reward, next_state):
    #     # next_state = next_state.clone().detach().to(device)
    #     action = torch.argmax(self.q_network(next_state)).item()
    #     y = reward + (self.gamma * self.target_network(next_state)[action].item())
    #     return y
    
    def update_q_network(self, state, action, target):
        # state = state.clone().detach().to(device)
        # target = target.clone().detach().to(device)
        loss = nn.MSELoss()(self.q_network(state)[action], target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def load_target_network_with_current_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    # def update_target_network_using_polyak_avg(self):
    #     for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
    #         target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
            
    def add_to_replay_buffer(self, transition):
        self.replay_buffer.append(transition)

    def sample_replay_buffer(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return self.replay_buffer
        return random.sample(self.replay_buffer, batch_size)

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        transition_batch = self.sample_replay_buffer(batch_size)
        states = torch.stack([transition.state for transition in transition_batch])
        actions = torch.stack([transition.action for transition in transition_batch])
        rewards = torch.stack([transition.reward for transition in transition_batch])
        next_states = torch.stack([transition.next_state for transition in transition_batch])

        # Compute targets using double Q-learning
        with torch.no_grad():
            next_actions = torch.argmax(self.q_network(next_states), dim=1)
            next_action_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + (self.gamma * next_action_values)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(q_values, targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network and epsilon
        self.update_count += 1
        if self.update_count % self.target_network_update_period == 0:
            self.load_target_network_with_current_network()
        if self.update_count % self.decay_iter == 0:
            self.decay_epsilon()

        # old, inefficient code
        # transitions = self.sample_replay_buffer(batch_size)
        # for transition in transitions:
        #     state = transition.state
        #     action = transition.action
        #     reward = transition.reward
        #     next_state = transition.next_state
        #     target = self.estimate_target(reward, next_state)
        #     self.update_q_network(state, action, target)
        ##     self.update_count += batch_size
        ##     if self.update_count % self.target_network_update_period == 0:
        ##        self.load_target_network_with_current_network()
        ##     if self.update_count % self.decay_iter == 0:
        ##         self.decay_epsilon()

    def decay_epsilon(self):
        self.decay_epsilon_exponential()

    def decay_epsilon_linear(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.decay_rate)

    def decay_epsilon_exponential(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)
    
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))


# Train the agent in the simulation environment

if __name__ == "__main__":
    N_ACTIONS = 8
    env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")

    state_dim = 6
    agent = DQNAgent(state_dim)

    # HYPERPARAMETERS: batch_size, update_freq
    n_episodes = 1000
    batch_size = 128
    update_freq = 4
    reward_list = []
    for episode in range(n_episodes):
        env.reset()
        done = False
        cum_reward = 0.0
        start = time.time()
        action_count = 0
        while not done:
            state = torch.tensor(env.high_level_state(), dtype=torch.float32).to(device)
            action = agent.get_action(state)
            next_state, reward, is_terminal, is_truncated = env.step(action)
            transition = Transition(state, action, reward, next_state)
            agent.add_to_replay_buffer(transition)
            action_count += 1
            if action_count % update_freq == 0:
                agent.train(batch_size)

            done = is_terminal or is_truncated
            cum_reward += reward
        end = time.time()
        print(f"Episode={episode}, reward={cum_reward}, RF={env.data.time/(end-start):.2f}")

        # plot episode-reward graph
        reward_list.append(cum_reward)

    plt.plot(reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # save the plot instead of displaying it
    plt.savefig('reward_plot.png')

    # save the model
    agent.save('model.pth')

