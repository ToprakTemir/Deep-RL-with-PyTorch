import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn.functional import softplus
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
from datetime import datetime

# Import your custom environment base class
# Ensure 'environment.py' is in the same directory or properly installed as a package
import environment  # Replace with the correct import if different

import homework3

class Hw3Env(gym.Env):
    """
    Custom Environment that follows OpenAI Gymnasium interface.
    This environment is based on your existing Hw3Env.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None, seed=None, **kwargs):
        super(Hw3Env, self).__init__()

        # Initialize your base environment
        self.base_env = homework3.Hw3Env(render_mode=render_mode, seed=seed, **kwargs)
        self.base_env._create_scene(seed)  # Initialize the scene

        # Define action and observation space
        # Example: Continuous action space with 2 dimensions (modify as needed)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: Image-based observations
        # Assuming images are 128x128 with 3 color channels and normalized to [0,1]
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(3, 128, 128), dtype=np.float32)

        # Additional environment parameters
        self._delta = 0.05
        self._goal_thresh = 0.01
        self._max_timesteps = 50
        self._t = 0  # Initialize timestep
        self.viewer = None  # Initialize viewer if needed
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state and return the initial observation.
        """
        super().reset(seed=seed)
        self.base_env.reset(seed=seed, options=options)  # Reset BaseEnv
        self._t = 0
        self.base_env._create_scene(seed)  # Re-create the scene with the seed
        initial_state = self.state()
        info = {}
        return initial_state, info

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            observation (object): Agent's observation of the current environment.
            reward (float): Amount of reward returned after previous action.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (dict): Contains auxiliary diagnostic information.
        """
        # Convert action from numpy to torch tensor
        action = torch.tensor(action, dtype=torch.float32)

        # Clamp and scale action
        action = action.clamp(-1, 1) * self._delta
        action_np = action.cpu().numpy()

        # Execute action in the base environment
        ee_pos = self.base_env.data.site(self.base_env._ee_site).xpos[:2]  # [x, y]
        target_pos = np.concatenate([ee_pos, [1.06]])  # Set z to 1.06
        target_pos[:2] = np.clip(target_pos[:2] + action_np, [0.25, -0.3], [0.75, 0.3])

        # Apply the target position to the end-effector
        self.base_env._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180],
                                           n_splits=30, threshold=0.04)
        self._t += 1

        # Get next state
        state = self.state()

        # Calculate reward
        reward = self.reward()

        # Check if done
        done = self.is_terminal()
        truncated = self.is_truncated()

        info = {}

        return state, reward, done, truncated, info

    def render(self, mode='human'):
        """
        Render the environment.

        Args:
            mode (str): The mode to render with. 'human' for visual rendering and 'rgb_array' for image.
        """
        if mode == 'rgb_array':
            return self.state().permute(1, 2, 0).numpy()
        elif mode == 'human':
            # Implement human rendering if necessary
            if self.viewer is None:
                # Initialize your viewer here if using a specific rendering tool
                pass
            # Update and render the viewer
            pass

    def close(self):
        """Clean up resources."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def state(self):
        """Return the current state as an image tensor normalized to [0,1]."""
        if self.base_env._render_mode == "offscreen":
            self.base_env.viewer.update_scene(self.base_env.data, camera="topdown")
            pixels = torch.tensor(self.base_env.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.base_env.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return (pixels.float() / 255.0).numpy()  # Normalize to [0,1]

    def high_level_state(self):
        """Return high-level state information."""
        ee_pos = self.base_env.data.site(self.base_env._ee_site).xpos[:2]
        obj_pos = self.base_env.data.body("obj1").xpos[:2]
        goal_pos = self.base_env.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    def reward(self):
        """Compute and return the reward for the current state."""
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]
        ee_to_obj = max(100 * np.linalg.norm(ee_pos - obj_pos), 1)
        obj_to_goal = max(100 * np.linalg.norm(obj_pos - goal_pos), 1)
        return 1 / ee_to_obj + 1 / obj_to_goal

    def is_terminal(self):
        """Check if the episode has reached a terminal state."""
        obj_pos = self.base_env.data.body("obj1").xpos[:2]
        goal_pos = self.base_env.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        """Check if the episode has been truncated due to max timesteps."""
        return self._t >= self._max_timesteps

def main():
    # Create directories for saving models and rewards if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("rewards", exist_ok=True)

    # Instantiate the environment
    env = Hw3Env(render_mode="offscreen")  # Use "human" if you want to render

    # Optional: Check if the environment follows the Gymnasium interface
    try:
        check_env(env, warn=True)
        print("Environment is valid!")
    except AssertionError as e:
        print("Environment check failed:", e)
        return
    except TypeError as e:
        print("TypeError during environment check:", e)
        return

    # Define the PPO model
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,  # Adjust entropy coefficient as needed
        tensorboard_log="./ppo_hw3_tensorboard/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Define total timesteps
    total_timesteps = 1_000_000  # Adjust based on your computational resources

    # Start training
    start_time = datetime.now()
    print(f"Training started at {start_time}")
    model.learn(total_timesteps=total_timesteps)
    end_time = datetime.now()
    print(f"Training finished at {end_time}, duration: {end_time - start_time}")

    # Save the trained model
    model_save_path = os.path.join("models", f"ppo_hw3_model_{start_time.strftime('%Y%m%d_%H%M%S')}.zip")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Optionally, evaluate the trained agent
    evaluate(model, env)

def evaluate(model, env, num_episodes=10):
    """
    Evaluate the trained agent.

    Args:
        model: Trained PPO model.
        env: Gymnasium environment.
        num_episodes: Number of episodes to run for evaluation.
    """
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        cum_reward = 0.0
        while not (done or truncated):
            # Predict action using the trained model
            action, _states = model.predict(obs, deterministic=True)

            # Take action in the environment
            obs, reward, done, truncated, info = env.step(action)

            cum_reward += reward

            # Render the environment (optional)
            # env.render()

        print(f"Episode {episode + 1} finished with cumulative reward: {cum_reward}")

    env.close()

if __name__ == "__main__":
    main()