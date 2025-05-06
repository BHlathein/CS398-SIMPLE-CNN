from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
from boop_env import BoopEnv
import random

class SelfPlayBoopEnv(gym.Env):
    def __init__(self, opponent_model=None):
        super().__init__()
        self.env = BoopEnv()
        self.opponent_model = opponent_model
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6, 6, 5), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2, 6, 6, 2])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.env.reset()

    def step(self, action):
        if self.env.current_player_num == 0:
            info = {}
            reward = 0.0

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

        else: # opponent
            obs = self.env.observation
            retries = 0
            max_retries = 100
            while retries < max_retries:
                opponent_action, _ = self.opponent_model.predict(obs, deterministic=False)
                if self.env.is_legal(opponent_action):
                    obs, reward, terminated, truncated, info = self.env.step(opponent_action)
                    reward = -reward if reward != 0 else 0
                    break
                retries += 1

            if retries >= max_retries:
                # Fallback: sample random legal move
                legal_actions = self.env.legal_actions()
                if legal_actions:
                    fallback_action = random.choice(legal_actions)
                    obs, reward, terminated, truncated, info = self.env.step(fallback_action)
                    reward = -reward if reward != 0 else 0
                    info["note"] = "opponent fallback move"
                else:
                    # If no legal moves exist at all (unlikely), end the game
                    reward = 1
                    terminated = True
                    truncated = False
                    info = {"reason": "opponent_no_legal_moves"}

        return obs, reward, terminated, truncated, info

# Opponent is just a random agent
class RandomOpponent:
    def predict(self, obs, deterministic=True):
        return SelfPlayBoopEnv().action_space.sample(), None

from stable_baselines3.common.callbacks import BaseCallback

class OverfittingTracker(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.entropies = []
        self.ep_len_mean = []
        self.ep_rew_mean = []

    def _on_step(self) -> bool:
        ep_info = self.locals.get("infos", [{}])[0]
        if "episode" in ep_info:
            self.ep_len_mean.append(ep_info["episode"]["l"])
            self.ep_rew_mean.append(ep_info["episode"]["r"])
        entropy = self.model.logger.name_to_value.get("train/entropy_loss")
        if entropy is not None:
            self.entropies.append(entropy)
        return True

if __name__ == "__main__":
    
    timesteps = 20000
    # 1000000 is the optimal

    callback = OverfittingTracker()
    env = SelfPlayBoopEnv(opponent_model=RandomOpponent())
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save("ppo_boop_v0")

    import matplotlib.pyplot as plt

    # Entropy plot
    plt.plot(callback.entropies)
    plt.title("Entropy over Training")
    plt.xlabel("Step")
    plt.ylabel("Entropy")
    plt.grid(True)
    plt.show()

    # Ep Len
    plt.plot(callback.ep_len_mean)
    plt.title("Mean Episode Length")
    plt.xlabel("Episode")
    plt.ylabel("Mean Length")
    plt.grid(True)
    plt.show()

    # Ep Reward
    plt.plot(callback.ep_rew_mean)
    plt.title("Mean Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()