from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
import gymnasium as gym
import numpy as np
from boop_env import BoopEnv
import random
import torch as th
from torch import nn

class CustomCNN(nn.Module):
    def __init__(self, observation_space, features_dim=128):
        super().__init__()
        # Input channels: 5 (as defined in BoopEnv)
        n_input_channels = observation_space.shape[0]
        self.features_dim = features_dim
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate output size of CNN
        with th.no_grad():
            sample = th.zeros((1, n_input_channels, 6, 6))  # Sample input
            cnn_output = self.cnn(sample)
            n_flatten = cnn_output.shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        return self.linear(self.cnn(observations))

class SelfPlayBoopEnv(gym.Env):
    def __init__(self, opponent_model=None):
        super().__init__()
        self.env = BoopEnv()
        self.opponent_model = opponent_model
        # Use the same observation and action spaces as the wrapped environment
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, action):
        if self.env.current_player_num == 0:
            info = {}
            reward = 0.0

            # Check if the action is legal for player 0
            if not self.env.is_legal(action):
                info["invalid"] = True
                reward = -0.1  # small penalty
                legal_actions = self.env.legal_actions()
                if legal_actions:
                    action = random.choice(legal_actions)
                else:
                    return self.env.get_observation(), reward, True, False, {"reason": "no_legal_moves"}

            # Always advance the game state
            obs, env_reward, terminated, truncated, env_info = self.env.step(action)
            
            # If invalid, keep penalty; else use env reward
            reward = reward if "invalid" in info else env_reward
            info.update(env_info)
            
            # If game ended after player 0's move
            if terminated:
                return obs, reward, terminated, truncated, info

        # Opponent's turn (player 1)
        # Note: We only reach here if player 0's move didn't end the game
        obs = self.env.get_observation()
        
        # Get legal moves for opponent
        legal_actions = self.env.legal_actions()
        if not legal_actions:
            # If no legal moves exist for opponent, player 0 wins
            return obs, 1.0, True, False, {"reason": "opponent_no_legal_moves"}
        
        # Make opponent move (either from model or randomly)
        try:
            if self.opponent_model:
                opponent_action, _ = self.opponent_model.predict(obs, deterministic=False)
                # Check if the action is legal, if not use random legal move
                if not self.env.is_legal(opponent_action):
                    opponent_action = random.choice(legal_actions)
            else:
                # Just use random legal move
                opponent_action = random.choice(legal_actions)
            
            # Execute opponent move
            obs, env_reward, terminated, truncated, info = self.env.step(opponent_action)
            
            # Invert reward since we're training player 0
            # If opponent wins, player 0 gets negative reward
            reward = -env_reward if env_reward != 0 else 0
            
        except Exception as e:
            print(f"Error during opponent move: {e}")
            # Fall back to random move
            opponent_action = random.choice(legal_actions)
            obs, env_reward, terminated, truncated, info = self.env.step(opponent_action)
            reward = -env_reward if env_reward != 0 else 0
            info["note"] = "fallback opponent move"

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

# Simple random opponent model for self-play
class RandomOpponent:
    def __init__(self):
        self.action_space = gym.spaces.MultiDiscrete([2, 6, 6, 2])
        
    def predict(self, obs, deterministic=True):
        # Just return a random action - the environment will handle legality
        action = np.array([
            random.randint(0, 1),              # action_type
            random.randint(0, 5),              # row
            random.randint(0, 5),              # col
            random.randint(0, 1)               # piece_type
        ])
        return action, None

from stable_baselines3.common.callbacks import BaseCallback

class OverfittingTracker(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.entropies = []
        self.ep_len_mean = []
        self.ep_rew_mean = []

    def _on_step(self) -> bool:
        if "infos" in self.locals and len(self.locals["infos"]) > 0:
            ep_info = self.locals["infos"][0]
            if "episode" in ep_info:
                self.ep_len_mean.append(ep_info["episode"]["l"])
                self.ep_rew_mean.append(ep_info["episode"]["r"])
                
        if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
            entropy = self.model.logger.name_to_value.get("train/entropy_loss")
            if entropy is not None:
                self.entropies.append(entropy)
        return True

if __name__ == "__main__":
    
    timesteps = 20000
    # 1000000 is the optimal
    
    # Set up learning environment
    callback = OverfittingTracker()
    env = SelfPlayBoopEnv(opponent_model=RandomOpponent())
    
    # Define CNN feature extractor parameters
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    model = PPO(
        "CnnPolicy",  # Use CNN policy
        env, 
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )
    
    try:
        model.learn(total_timesteps=timesteps, callback=callback)
        model.save("ppo_boop_cnn_v0")
        print("Model training completed successfully")
    except Exception as e:
        print(f"Error during training: {e}")

    try:
        import matplotlib.pyplot as plt

        # Entropy plot
        if callback.entropies:
            plt.figure(figsize=(10, 6))
            plt.plot(callback.entropies)
            plt.title("Entropy over Training")
            plt.xlabel("Step")
            plt.ylabel("Entropy")
            plt.grid(True)
            plt.savefig("entropy_plot.png")
            plt.show()

        # Ep Len
        if callback.ep_len_mean:
            plt.figure(figsize=(10, 6))
            plt.plot(callback.ep_len_mean)
            plt.title("Mean Episode Length")
            plt.xlabel("Episode")
            plt.ylabel("Mean Length")
            plt.grid(True)
            plt.savefig("episode_length_plot.png")
            plt.show()

        # Ep Reward
        if callback.ep_rew_mean:
            plt.figure(figsize=(10, 6))
            plt.plot(callback.ep_rew_mean)
            plt.title("Mean Reward")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True)
            plt.savefig("reward_plot.png")
            plt.show()
    except Exception as e:
        print(f"Error during plotting: {e}")

# Function to load and use the model
def load_and_use_model():
    try:
        model = PPO.load("ppo_boop_cnn_v0")
        env = BoopEnv()
        obs, _ = env.reset()
        
        terminated = False
        while not terminated:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated:
                print("Game finished with info:", info)
                break
    except Exception as e:
        print(f"Error loading or using model: {e}")
