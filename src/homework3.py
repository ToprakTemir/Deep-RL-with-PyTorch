import time
from absl.logging import flush
from jedi.api.strings import complete_dict
from queue import Queue
from collections import deque
from datetime import datetime

import torch
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import softplus

import environment

timestep_multiplier = 4

class Hw3Env(environment.BaseEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # divide the action space into n_actions
        self._delta = 0.05

        self._goal_thresh = 0.01
        self._max_timesteps = 50 * timestep_multiplier

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [np.random.uniform(0.25, 0.75),
                   np.random.uniform(-0.3, 0.3),
                   1.5]
        goal_pos = [np.random.uniform(0.25, 0.75),
                    np.random.uniform(-0.3, 0.3),
                    1.025]
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels.float() / 255.0

    def _high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    def high_level_state(self):
        return torch.tensor(self._high_level_state(), dtype=torch.float32)

    def reward(self):
        state = self._high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]
        ee_to_obj = max(100*np.linalg.norm(ee_pos - obj_pos), 1)
        obj_to_goal = max(100*np.linalg.norm(obj_pos - goal_pos), 1)
        return 1/ee_to_obj + 1/obj_to_goal

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps

    def step(self, action):
        action = action.clamp(-1, 1).detach().cpu().numpy() * self._delta
        ee_pos = self.data.site(self._ee_site).xpos[:2] # ee_pos = [x, y]
        target_pos = np.concatenate([ee_pos, [1.06]])   # set z to 1.06, just above the table
        target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3]) # add the action
        self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)
        self._t += 1

        state = self.high_level_state()
        reward = self.reward()
        terminal = self.is_terminal()
        truncated = self.is_truncated()
        return state, reward, terminal, truncated



class PolicyNetwork(torch.nn.Module):
    r""" Takes the image as the state and outputs the policy action """
    def __init__(self) -> None: # for learning from raw pixels
        super(PolicyNetwork, self).__init__()

        # for learning directly from raw pixels
        # self.conv = torch.nn.Sequential(
        #     torch.nn.Conv2d(3, 8, 3, 2, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(8, 16, 3, 2, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(16, 32, 3, 2, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(32, 64, 3, 2, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(64, 128, 3, 2, 1),
        #     torch.nn.ReLU()
        # )
        # self.linear = torch.nn.Linear(128, 4) # mean and std for 2D action

        # for learning from high-level state
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(6, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 4)
        )


    def forward(self, x):
        # if x.ndim == 3:
        #     x = x.unsqueeze(0)
        # x = self.conv(x)
        # x = x.mean(dim=[2, 3])

        x = self.linear(x)
        std_x1 = softplus(x[:, 1].clone()) + 1e-4
        std_x3 = softplus(x[:, 3].clone()) + 1e-4
        x = torch.cat([x[:, 0].unsqueeze(1), std_x1.unsqueeze(1), x[:, 2].unsqueeze(1), std_x3.unsqueeze(1)], dim=1)
        return x

    def copy(self):
        new_model = PolicyNetwork()
        new_model.load_state_dict(self.state_dict())
        return new_model

class ValueNetwork(torch.nn.Module):
    r""" Takes the image as the state and outputs the state value """
    def __init__(self) -> None:
        super(ValueNetwork, self).__init__()

        # for learning directly from raw pixels
        # self.conv = torch.nn.Sequential(
        #     torch.nn.Conv2d(3, 8, 3, 2, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(8, 16, 3, 2, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(16, 32, 3, 2, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(32, 64, 3, 2, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(64, 128, 3, 2, 1),
        #     torch.nn.ReLU()
        # )
        # self.linear = torch.nn.Linear(128, 1)

        # for learning from high-level state
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(6, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        # if x.ndim == 3:
        #     x = x.unsqueeze(0)
        # x = self.conv(x)
        # x = x.mean(dim=[2, 3])

        if x.ndim == 6:
            x = x.unsqueeze(0)
        x = self.linear(x)
        return x

    def copy(self):
        new_model = ValueNetwork()
        new_model.load_state_dict(self.state_dict())
        return new_model


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    last_value = last_value.detach()
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t+1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages

def collector(policy_network, value_network, shared_queue, is_collecting, is_finished, shared_reward_list):
    env = Hw3Env(render_mode="offscreen")

    while not is_finished.is_set():
        while is_collecting.is_set():
            env.reset()
            state = env.high_level_state()

            done = False
            cum_reward = 0.0
            last_value = 0.0 # the value estimation of the last state

            rollout = {
                "states": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "log_probs": [],
                "values": [],
                "advantages": []
            }
            while not done:
                with torch.no_grad():
                    action = policy_network(state.unsqueeze(0))
                    value = value_network(state.unsqueeze(0))

                action = action[0] # convert (1, 4) to (4,)
                mean = action[[0, 2]]
                std = action[[1, 3]]

                cov_matrix = torch.diag(std**2)
                dist = torch.distributions.MultivariateNormal(mean, cov_matrix)
                action = dist.sample()

                log_prob = dist.log_prob(action) # the log probability of the executed action being selected

                next_state, reward, is_terminal, is_truncated = env.step(action)
                cum_reward += reward
                done = is_terminal or is_truncated
                if done:
                    last_value = value

                rollout["states"].append(state)
                rollout["actions"].append(action)
                rollout["rewards"].append(reward)
                rollout["dones"].append(done)
                rollout["log_probs"].append(log_prob)
                rollout["values"].append(value)

                state = next_state
                if is_finished.is_set():
                    break

            # post-processing the advantages
            advantages = compute_gae(rollout["rewards"], rollout["values"], rollout["dones"], last_value)
            rollout["advantages"] = advantages

            for i in np.random.permutation(len(rollout["states"])):
                shared_queue.put([rollout["states"][i], rollout["actions"][i], rollout["rewards"][i], rollout["dones"][i], rollout["log_probs"][i], rollout["values"][i], rollout["advantages"][i]])

            print(cum_reward)
            flush()
            shared_reward_list.append(cum_reward)


            if is_finished.is_set():
                break
        is_collecting.wait()

class RolloutBuffer:
    def __init__(self, buffer_length=None):
        self.buffer = {
            "states": deque(maxlen=buffer_length),
            "actions": deque(maxlen=buffer_length),
            "rewards": deque(maxlen=buffer_length),
            "dones": deque(maxlen=buffer_length),
            "log_probs": deque(maxlen=buffer_length),
            "values": deque(maxlen=buffer_length),
            "advantages": deque(maxlen=buffer_length)
        }
        self.keys = self.buffer.keys()

    def __len__(self):
        return len(self.buffer["states"])

    def clear(self):
        self.buffer[self.keys].clear()

    def append_as_list(self, dic):
        i = 0
        for key in self.keys:
            self.buffer[key].append(dic[i])
            i += 1

    def get(self, sample_size):
        idx = np.random.choice(len(self), sample_size, replace=False)
        return self.get_by_idx(idx)

    def get_by_idx(self, idx):
        res = {}
        for key in self.keys:
            res[key] = torch.stack([torch.tensor(self.buffer[key][i]) for i in idx])
        return res


class PPOAgent:
    def __init__(self, policy_network, value_network) -> None:

        # HYPERPARAMETERS
        self.policy_lr = 3e-4 # learning rate
        self.value_lr = 3e-4 # learning rate
        self.n_steps = 2048  # number of steps to collect before updating the model
        self.batch_size = 64  # batch size for training
        self.num_epochs = 10  # number of epochs to train per update
        self.gamma = 0.99  # discount factor
        self.gae_lambda = 0.95  # for GAE (Generalized Advantage Estimation)
        self.epsilon = 0.2  # clip ratio
        self.ent_coeff = 0.01 # entropy coefficient for the policy loss
        self.max_grad_norm = 0.5 # max value for the gradient clipping
        self.num_workers = 6
        self.rollout_buffer = RolloutBuffer(self.n_steps)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = policy_network
        self.value_network = value_network
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.policy_lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.value_lr)

    def add_to_buffer(self, observation):
        assert len(observation) == len(self.rollout_buffer.keys)
        self.rollout_buffer.append_as_list(observation)

    def to(self, device):
        self.device = device
        self.policy_network.to(device)
        self.value_network.to(device)
        return self

    def train(self):
        for _ in range(self.num_epochs):
            rollout = self.rollout_buffer.get(self.batch_size)

            states = rollout["states"].to(self.device).float()
            actions = rollout["actions"].to(self.device).float()
            rewards = rollout["rewards"].to(self.device).float()
            dones = rollout["dones"].to(self.device).float()
            old_log_probs = rollout["log_probs"].to(self.device).float()
            values = rollout["values"].to(self.device).float()
            advantages = rollout["advantages"].to(self.device).float()

            print(f"mean advantages={advantages.mean()}")
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # compute policy loss
            policy_output = self.policy_network(states)
            mean = policy_output[:, [0, 2]]
            std = policy_output[:, [1, 3]]
            cov_matrix = torch.diag_embed(std**2)
            dist = torch.distributions.MultivariateNormal(mean, cov_matrix)
            print(f"policy entropy={dist.entropy().mean()}")
            log_probs = dist.log_prob(actions) # log prob of action being sampled from the new policy

            entropy = dist.entropy().mean()
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean() - self.ent_coeff * entropy

            # update policy network
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()

            # compute value loss
            values_predicted = self.value_network(states).squeeze()
            returns = (advantages + values.squeeze()).squeeze()
            print(f"values_predicted={values_predicted.mean()}, rewards={rewards.mean()}, returns={returns.mean()}")
            mse_loss = torch.nn.MSELoss()
            value_loss = mse_loss(values_predicted, returns)

            # update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
            self.value_optimizer.step()


if __name__ == "__main__":

    start_time = datetime.now()

    # multiprocessing setup
    mp.set_start_method("spawn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_network = PolicyNetwork()
    policy_network.share_memory()  # share model parameters across processes
    value_network = ValueNetwork()
    value_network.share_memory()
    PPOAgent = PPOAgent(policy_network, value_network).to(device)

    shared_queue = mp.Queue()
    shared_reward_list = mp.Manager().list()
    is_collecting = mp.Event()
    is_finished = mp.Event()

    worker_processes = []
    for i in range(PPOAgent.num_workers):
        p = mp.Process(target=collector, args=(policy_network, value_network, shared_queue, is_collecting, is_finished, shared_reward_list))
        p.start()
        worker_processes.append(p)

    rewards_file_path = f"HW3/rewards/rewards_list_{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}-x{timestep_multiplier}.pth"
    rewards_file = open(rewards_file_path, "w")


    num_updates = 10000
    for i in range(num_updates):

        # collecting data
        is_collecting.set()
        start = time.time()

        buffer_feeded = 0
        while buffer_feeded < PPOAgent.n_steps:
            if not shared_queue.empty():

                state, action, reward, done, log_prob, value, advantage = shared_queue.get()
                PPOAgent.add_to_buffer([state, action, reward, done, log_prob, value, advantage])
                del state, action, reward, done, log_prob, value, advantage
                buffer_feeded += 1

                # save the rewards
                for item in shared_reward_list:
                    rewards_file.write(f"{item}\n")
                shared_reward_list[:] = [] # clear the list
                rewards_file.flush()


        end = time.time()
        is_collecting.clear()
        print(f"{PPOAgent.n_steps / (end-start)} samples/sec for {end-start} seconds. Updating model.")

        # training
        PPOAgent.train()
        torch.save(policy_network.state_dict(), f"HW3/network-snapshots/policy_network_{start_time}.pth")
        torch.save(value_network.state_dict(), f"HW3/network-snapshots/value_network_{start_time}.pth")
        print(f"Model updated in {time.time() - end} seconds")


    # save the models
    torch.save(policy_network.state_dict(), "HW3/policy_network.pth")
    torch.save(value_network.state_dict(), "HW3/value_network.pth")


    is_collecting.clear()
    is_finished.set()
    for p in worker_processes:
        p.kill()
