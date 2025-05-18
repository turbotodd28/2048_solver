import numpy as np
import gymnasium as gym
from src.gym_env import Game2048Env
import matplotlib.pyplot as plt
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO


def run_random_agent(episodes=5):
    print("\nRunning random agent...")
    env = Game2048Env()
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        print(f"Episode {ep+1}: Total reward={total_reward:.1f}, Steps={steps}, Max tile={int(2**np.max(obs))}")
    env.close()


class PlottingCallback(BaseCallback):
    def __init__(self, plot_interval=10000, verbose=0):
        super().__init__(verbose)
        self.plot_interval = plot_interval
        self.rewards = []
        self.max_tiles = []
        self.steps = []
        self.best_reward = float('-inf')
        self.best_max_tile = float('-inf')
        plt.ion()

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None and any(self.locals['dones']):
            infos = self.locals['infos']
            for info in infos:
                if 'episode' in info:
                    reward = info['episode']['r']
                    self.rewards.append(reward)
                    self.steps.append(self.num_timesteps)
                    if 'max_tile' in info:
                        max_tile = info['max_tile']
                        self.max_tiles.append(max_tile)
                        if max_tile > self.best_max_tile:
                            self.best_max_tile = max_tile
                            print(f"[Step {self.num_timesteps}] New best max tile: {max_tile}")
                    if reward > self.best_reward:
                        self.best_reward = reward
                        print(f"[Step {self.num_timesteps}] New best episode reward: {reward}")
        if len(self.rewards) > 0 and self.num_timesteps % self.plot_interval < self.locals['env'].num_envs:
            avg_reward = np.mean(self.rewards[-self.plot_interval//10:]) if len(self.rewards) >= self.plot_interval//10 else np.mean(self.rewards)
            print(f"[Step {self.num_timesteps}] Recent avg reward: {avg_reward:.1f}")
            plt.clf()
            plt.plot(self.steps, self.rewards, label='Episode Reward')
            if self.max_tiles:
                plt.plot(self.steps[:len(self.max_tiles)], self.max_tiles, label='Max Tile')
            plt.xlabel('Steps')
            plt.ylabel('Reward / Max Tile')
            plt.title('Learning Curve')
            plt.legend()
            plt.pause(0.01)
        return True


def make_env():
    def _init():
        return Game2048Env()
    return _init


def run_sb3_ppo(train_steps=1_000_000, reward_variant='default', plot_interval=10000, save_path=None):
    print(f"\nTraining PPO agent with Stable Baselines3 for {train_steps} steps (reward: {reward_variant})...")
    num_envs = 2  # Parallel environments, suitable for 4-thread CPU
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        device='cpu'
    )
    callback = PlottingCallback(plot_interval=plot_interval)
    model.learn(total_timesteps=train_steps, callback=callback)
    plt.ioff()
    plt.show()
    if save_path:
        model.save(save_path)
        print(f"Model saved to {save_path}")
    obs, info = env.reset()
    done = [False] * num_envs
    total_rewards = [0] * num_envs
    steps_taken = [0] * num_envs
    while not all(done):
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, truncateds, infos = env.step(actions)
        for i in range(num_envs):
            if not done[i]:
                total_rewards[i] += rewards[i]
                steps_taken[i] += 1
                done[i] = dones[i]
    print(f"PPO agent finished across {num_envs} envs. Avg reward={np.mean(total_rewards):.1f}, Avg steps={np.mean(steps_taken):.1f}")
    env.close()


if __name__ == "__main__":
    print("=== 2048 Gym Environment Demo ===")
    print("1. Random agent: takes random moves, no learning.")
    print("2. PPO agent (Stable Baselines3): learns from experience.")
    run_random_agent(episodes=5)
    run_sb3_ppo(train_steps=1_000_000, reward_variant='default', plot_interval=20000, save_path='ppo_2048_model')
