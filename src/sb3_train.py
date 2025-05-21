import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO, DQN
from src.gym_env import Game2048Env


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
    def __init__(self, plot_interval=10000, verbose=0, env_ref=None, store_data=None, sweep_id=None, show_plot=True):
        super().__init__(verbose)
        self.plot_interval = plot_interval
        self.rewards = []
        self.max_tiles = []
        self.steps = []
        self.best_reward = float('-inf')
        self.best_max_tile = float('-inf')
        self.episode_count = 0
        self.env_ref = env_ref  # Reference to env for reward structure string
        self.show_plot = show_plot
        self.store_data = store_data  # Optional dict to store results for sweep
        self.sweep_id = sweep_id      # Optional identifier for sweep run
        if self.show_plot:
            plt.ion()

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None and any(self.locals['dones']):
            infos = self.locals['infos']
            for info in infos:
                if 'episode' in info:
                    reward = info['episode']['r']
                    self.rewards.append(reward)
                    self.steps.append(self.num_timesteps)
                    self.episode_count += 1
                    if 'max_tile' in info:
                        max_tile = info['max_tile']
                        self.max_tiles.append(max_tile)
                        if max_tile > self.best_max_tile:
                            self.best_max_tile = max_tile
                            print(f"[Step {self.num_timesteps}] New best max tile: {max_tile}")
                    if reward > self.best_reward:
                        self.best_reward = reward
                        print(f"[Step {self.num_timesteps}] New best episode reward: {reward}")
        # Only plot on the first call, and reuse the same figure/axes
        if not hasattr(self, 'fig'):
            self.fig, self.ax1 = plt.subplots()
            self.ax2 = self.ax1.twinx()
        if len(self.rewards) > 0 and self.num_timesteps % self.plot_interval < getattr(self.locals['env'], 'num_envs', 1):
            avg_reward = np.mean(self.rewards[-self.plot_interval//10:]) if len(self.rewards) >= self.plot_interval//10 else np.mean(self.rewards)
            print(f"[Step {self.num_timesteps}] Recent avg reward: {avg_reward:.1f}")
            if self.show_plot:
                self.ax1.clear()
                self.ax2.clear()
                # Smoothing for reward and max tile
                window = min(51, max(3, len(self.rewards)//20))
                if window > 1:
                    def smooth(x, w):
                        return np.convolve(x, np.ones(w)/w, mode='valid')
                    rewards_smooth = smooth(self.rewards, window)
                    steps_smooth = self.steps[window-1:]
                    max_tiles_smooth = smooth(self.max_tiles, window) if len(self.max_tiles) >= window else self.max_tiles
                    self.ax1.plot(steps_smooth, rewards_smooth, label='Episode Reward', color='tab:blue')
                    if len(self.max_tiles) >= window:
                        self.ax2.plot(steps_smooth, max_tiles_smooth, label='Max Tile', color='tab:orange', alpha=0.7)
                else:
                    self.ax1.plot(self.steps, self.rewards, label='Episode Reward', color='tab:blue')
                    self.ax2.plot(self.steps[:len(self.max_tiles)], self.max_tiles, label='Max Tile', color='tab:orange', alpha=0.7)
                self.ax1.set_xlabel('Steps')
                self.ax1.set_ylabel('Episode Reward', color='tab:blue')
                self.ax2.set_ylabel('Max Tile', color='tab:orange')
                self.ax1.tick_params(axis='y', labelcolor='tab:blue')
                self.ax2.tick_params(axis='y', labelcolor='tab:orange')
                self.ax1.legend(loc='upper left')
                self.ax2.legend(loc='upper right')
                # Show reward weights in the plot title
                reward_title = self.env_ref.get_reward_structure_str() if self.env_ref is not None else 'Reward structure unknown'
                self.fig.suptitle(f'Learning Curve\n{reward_title}')
                self.fig.tight_layout(rect=[0, 0.07, 1, 0.95])
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        return True

    def _on_training_end(self) -> None:
        # Store data for sweep if requested
        if self.store_data is not None and self.sweep_id is not None:
            self.store_data[self.sweep_id] = {
                'rewards': self.rewards.copy(),
                'max_tiles': self.max_tiles.copy(),
                'steps': self.steps.copy(),
                'reward_structure': self.env_ref.get_reward_structure_str() if self.env_ref else None
            }
        if self.show_plot:
            plt.ioff()
            plt.show()


def run_sb3_ppo(train_steps=100_000, plot_interval=20000, save_path=None):
    print(f"\nTraining PPO agent with Stable Baselines3 for {train_steps} steps...")
    env = SubprocVecEnv([lambda: Game2048Env() for _ in range(2)])
    # Use a single env instance for reward structure string
    env_for_title = Game2048Env()
    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        device='cpu'
    )
    callback = PlottingCallback(plot_interval=plot_interval, env_ref=env_for_title)
    model.learn(total_timesteps=train_steps, callback=callback)
    print(f"Total episodes completed during training: {callback.episode_count}")
    plt.ioff()
    plt.show()
    if save_path:
        model.save(save_path)
        print(f"Model saved to {save_path}")
    eval_env = Game2048Env()
    for ep in range(20):
        obs, info = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward
            steps += 1
        print(f"Eval Episode {ep+1}: Total reward={total_reward:.1f}, Steps={steps}, Max tile={int(2**np.max(obs))}")
        print("Final board state:")
        print(eval_env.game.board)
    eval_env.close()


def run_sb3_dqn(train_steps=100_000, plot_interval=20000, save_path=None):
    print(f"\nTraining DQN agent with Stable Baselines3 for {train_steps} steps...")
    env = Game2048Env()
    model = DQN(
        'MlpPolicy',
        env,
        verbose=0,
        learning_rate=1e-3,
        batch_size=256,
        buffer_size=50000,
        learning_starts=1000,
        train_freq=4,
        target_update_interval=1000,
        device='cpu'
    )
    callback = PlottingCallback(plot_interval=plot_interval, env_ref=env)
    model.learn(total_timesteps=train_steps, callback=callback)
    print(f"Total episodes completed during training: {callback.episode_count}")
    plt.ioff()
    plt.show()
    if save_path:
        model.save(save_path)
        print(f"Model saved to {save_path}")
    eval_env = Game2048Env()
    for ep in range(20):
        obs, info = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward
            steps += 1
        print(f"Eval Episode {ep+1}: Total reward={total_reward:.1f}, Steps={steps}, Max tile={int(2**np.max(obs))}")
        print("Final board state:")
        print(eval_env.game.board)
    eval_env.close()


if __name__ == "__main__":
    print("=== 2048 Gym Environment Demo ===")
    print("1. Random agent: takes random moves, no learning.")
    print("2. PPO agent (Stable Baselines3): learns from experience.")
    print("3. DQN agent (Stable Baselines3): learns from experience.")
    run_random_agent(episodes=5)
    run_sb3_dqn(train_steps=200_000, plot_interval=20000, save_path='dqn_2048_model')
