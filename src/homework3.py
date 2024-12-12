import time
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


class Hw3Env(environment.BaseEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # divide the action space into n_actions
        self._delta = 0.05

        self._goal_thresh = 0.01
        self._max_timesteps = 2048 # !! MANUAL HYPERPARAMETER FOR ROLLOUT BUFFER

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
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.float32).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.float32).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels.float()

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    def reward(self):
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]
        ee_to_obj = max(100*np.linalg.norm(ee_pos - obj_pos), 1)
        obj_to_goal = max(100*np.linalg.norm(obj_pos - goal_pos), 1)
        return 1/(ee_to_obj) + 1/(obj_to_goal)

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps

    def step(self, action):
        action = action.clamp(-1, 1).cpu().numpy() * self._delta
        ee_pos = self.data.site(self._ee_site).xpos[:2] # ee_pos = [x, y]
        target_pos = np.concatenate([ee_pos, [1.06]])   # set z to 1.06, just above the table
        target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3]) # add the action
        self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)
        self._t += 1

        state = self.state()
        reward = self.reward()
        terminal = self.is_terminal()
        truncated = self.is_truncated()
        return state, reward, terminal, truncated


class Memory:
    def __init__(self, keys, buffer_length=None):
        self.buffer = {}
        self.keys = keys
        for key in keys:
            self.buffer[key] = deque(maxlen=buffer_length)

    def clear(self):
        for key in self.keys:
            self.buffer[key].clear()

    def append(self, dic):
        for key in self.keys:
            self.buffer[key].append(dic[key])

    def sample_n(self, n):
        r = torch.randperm(len(self))
        idx = r[:n]
        return self.get_by_idx(idx)

    def get_by_idx(self, idx):
        res = {}
        for key in self.keys:
            res[key] = torch.stack([self.buffer[key][i] for i in idx])
        return res

    def get_all(self):
        idx = list(range(len(self)))
        return self.get_by_idx(idx)

    def __len__(self):
        return len(self.buffer[self.keys[0]])


class PolicyNetwork(torch.nn.Module):
    r""" Takes the image as the state and outputs the policy action """
    def __init__(self) -> None:
        super(PolicyNetwork, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, 2, 1),
            torch.nn.ReLU()
        )
        self.linear = torch.nn.Linear(128, 4) # mean and std for 2D action

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = self.conv(x)
        x = x.mean(dim=[2, 3])
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
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, 2, 1),
            torch.nn.ReLU()
        )
        self.linear = torch.nn.Linear(128, 1)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = self.conv(x)
        x = x.mean(dim=[2, 3])
        x = self.linear(x)
        return x

    def copy(self):
        new_model = ValueNetwork()
        new_model.load_state_dict(self.state_dict())
        return new_model


def collector(model, shared_queue, is_collecting, is_finished, device, shared_reward_list):
    env = Hw3Env(render_mode="offscreen")

    while not is_finished.is_set():
        while is_collecting.is_set():
            env.reset()
            state = env.state()
            done = False
            cum_reward = 0.0
            while not done:
                with torch.no_grad():
                    action = model(state.to(device))

                if torch.isnan(action).any():
                    print("NaN detected in policy network output. Skipping this sample.")
                    break

                # !! below way ignores the correlations between x and y
                # x = torch.normal(action[0], action[1])
                # y = torch.normal(action[2], action[3])
                # action = torch.stack([x, y], dim=0)

                action = action[0] # convert (1, 4) to (4,)
                mean = action[[0, 2]]
                std = action[[1, 3]]

                if torch.isnan(mean).any() or torch.isnan(std).any():
                    print(f"NaN detected in mean or std: mean={mean}, std={std}. Skipping this sample.")
                    break

                cov_matrix = torch.diag(std**2)
                dist = torch.distributions.MultivariateNormal(mean, cov_matrix)
                action = dist.sample()

                next_state, reward, is_terminal, is_truncated = env.step(action)
                cum_reward += reward
                done = is_terminal or is_truncated
                shared_queue.put((state.float(), action, reward, next_state.float(), done))
                state = next_state
                if is_finished.is_set():
                    break

            shared_reward_list.append(cum_reward)
            print(cum_reward)

            if is_finished.is_set():
                break
        is_collecting.wait()

num_epochs = 10000

class PPOAgent:
    def __init__(self, policy_network, value_network) -> None:

        # HYPERPARAMETERS
        self.policy_lr = 3e-4 # learning rate
        self.value_lr = 3e-4 # learning rate
        self.n_steps = 2048 # number of steps to collect per worker before updating the model
        self.gamma = 0.99  # discount factor
        self.gae_lambda = 0.95  # for GAE (Generalized Advantage Estimation)
        self.epsilon = 0.2  # clip ratio
        self.initial_beta = 0.03 # entropy coefficient for the policy loss
        self.polyak_tau = 0.002 # for updating the target value network
        self.max_grad_norm = 0.5 # max value for the gradient clipping
        self.num_updates = 10  # number of updates given one big batch
        self.batch_size = 1024 # number of samples in one big batch
        self.mini_batch_size = 64 # number of samples in one mini-batch
        self.buffer_length = 10000 # replay buffer length

        self.current_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = policy_network
        self.value_network = value_network
        self.value_target_network = value_network.copy()
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.policy_lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.value_lr)
        self.replay_buffer = Memory(keys=["state", "action", "reward", "next_state", "done"], buffer_length=self.buffer_length)

    def to(self, device):
        self.device = device
        self.policy_network.to(device)
        self.value_network.to(device)
        return self

    def add_to_replay_buffer(self, transition):
        self.replay_buffer.append(transition)

    def sample_n(self, n):
        return self.replay_buffer.sample_n(n)

    # PPO:
    # 1) sample transitions
    # 2) compute advantage
    # 3) compute policy loss with clipped surrogate objective
    # 4) update policy
    # 4) compute value loss, fitting value function to the new policy
    # 5) update value
    def train(self):

        self.current_epoch += 1
        samples = self.replay_buffer.sample_n(self.batch_size)

        old_policy = self.policy_network.copy().to(self.device)
        for _ in range(self.num_updates):

            # sample mini-batch
            indices = torch.randperm(self.batch_size)[:self.mini_batch_size]
            states = samples["state"][indices].to(self.device)
            actions = samples["action"][indices].to(self.device)
            rewards = samples["reward"][indices].to(self.device)
            next_states = samples["next_state"][indices].to(self.device)
            dones = samples["done"][indices].to(self.device)

            # compute advantage
            # TODO: implement GAE (Generalized Advantage Estimation) for computing advantage
            with torch.no_grad():
                values = self.value_network(states).squeeze()
                next_values = self.value_network(next_states).squeeze()
                advantages = rewards + (1 - dones.float()) * self.gamma * next_values - values # one-step TD error
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages = advantages.detach()

            # compute policy loss

            new_policy_output = self.policy_network(states)
            mean = new_policy_output[:, [0, 2]]
            std = new_policy_output[:, [1, 3]]
            cov_matrix = torch.diag_embed(std**2)
            dist = torch.distributions.MultivariateNormal(mean, cov_matrix)
            print(f"mean={mean.mean()}, std={std.mean()}, entropy={dist.entropy().mean()}")
            new_log_probs = dist.log_prob(actions) # log prob of action being sampled from the new policy

            with torch.no_grad():
                old_policy_outputs = old_policy(states)
                mean = old_policy_outputs[:, [0, 2]]
                std = old_policy_outputs[:, [1, 3]]
                cov_matrix = torch.diag_embed(std**2)
                dist = torch.distributions.MultivariateNormal(mean, cov_matrix)
                old_log_probs = dist.log_prob(actions) # log prob of action being sampled from the old policy

            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
            policy_entropy = dist.entropy().mean()
            beta = max(self.initial_beta * (1 - self.current_epoch / num_epochs), 0.002)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean() - beta * policy_entropy

            # update policy network
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()

            # compute value loss
            # TODO: bootstrapped returns should be computed by advantages acquired by GAE
            values = self.value_network(states).squeeze()
            next_values = self.value_target_network(next_states).squeeze()
            bootstrapped_returns = rewards + (1 - dones.float()) * self.gamma * next_values
            print(f"values={values.mean()}, next_values={next_values.mean()}, rewards={rewards.mean()} returns={bootstrapped_returns.mean()}")
            bootstrapped_returns.detach()
            mse_loss = torch.nn.MSELoss()
            value_loss = mse_loss(values, bootstrapped_returns)

            # update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
            self.value_optimizer.step()

            # update the target network
            for target_param, param in zip(self.value_target_network.parameters(), self.value_network.parameters()):
                target_param.data.copy_(self.polyak_tau * param.data + (1 - self.polyak_tau) * target_param.data)



if __name__ == "__main__":

    start_time = datetime.now()

    # multiprocessing setup
    mp.set_start_method("spawn")
    simulation_device = torch.device("cpu")
    policy_network = PolicyNetwork().to(simulation_device)
    policy_network.share_memory()  # share model parameters across processes
    shared_reward_list = mp.Manager().list()
    shared_queue = mp.Queue()
    is_collecting = mp.Event()
    is_finished = mp.Event()

    procs = []
    num_workers = 6
    for i in range(num_workers):
        p = mp.Process(target=collector, args=(policy_network, shared_queue, is_collecting, is_finished, simulation_device, shared_reward_list))
        p.start()
        procs.append(p)

    now = datetime.now()
    # print working directory
    data_file_path = "HW3/rewards_list_" + now.strftime("%Y.%m.%d-%H:%M:%S") + ".pth"
    data_file = open(data_file_path, "w")

    # training agent setup
    training_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value_network = ValueNetwork().to(training_device)
    PPOAgent = PPOAgent(policy_network, value_network).to(training_device)

    for i in range(num_epochs):
        is_collecting.set()

        start = time.time()
        buffer_feeded = 0
        while buffer_feeded < 400:
            # print(f"Buffer size={len(PPOAgent.replay_buffer)}", end="\r")
            if not shared_queue.empty():
                # unfortunately, you can't feed the replay buffer as fast as you collect

                # save the rewards to a file
                for item in shared_reward_list:
                    data_file.write(f"{item}\n")
                shared_reward_list[:] = [] # ! don't write the same rewards again
                data_file.flush()

                state, action, reward, next_state, done = shared_queue.get()

                PPOAgent.add_to_replay_buffer({
                    "state": torch.tensor(state.clone(), dtype=torch.float32),
                    "action": torch.tensor(action.clone(), dtype=torch.float32),
                    "reward": torch.tensor(reward, dtype=torch.float32),
                    "next_state": torch.tensor(next_state.clone(), dtype=torch.float32),
                    "done": torch.tensor(done)
                })
                del state, action, reward, next_state, done
                buffer_feeded += 1
        end = time.time()

        is_collecting.clear()
        print(f"{400/(end-start):.2f} samples/sec for {end-start} seconds. Updating model.")

        if len(PPOAgent.replay_buffer) < PPOAgent.batch_size:
            continue

        # do your updates here
        PPOAgent.train()
        torch.save(policy_network.state_dict(), f"HW3/policy_network_{start_time}.pth")
        torch.save(value_network.state_dict(), f"HW3/value_network_{start_time}.pth")
        print(f"Model updated in {time.time()-end:.2f} seconds")


    reward_data = np.loadtxt(data_file_path)
    reward_list = reward_data.tolist()
    data_file.close()

    # plot the reward
    plt.plot(reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    # plot smoothed reward
    smoothed_reward = np.convolve(reward_list, np.ones(100)/100, mode='valid')
    plt.plot(smoothed_reward)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    # save the models
    torch.save(policy_network.state_dict(), "HW3/policy_network.pth")
    torch.save(value_network.state_dict(), "HW3/value_network.pth")


    is_collecting.clear()
    is_finished.set()
    for p in procs:
        p.kill()
