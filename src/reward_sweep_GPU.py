import os
import time
import torch
import json
import numpy as np
import multiprocessing as mp
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from src.gym_env import Game2048Env


# === Config ===
# Summary of reward sweep setup and evaluation (from notes):
#
# - We define a small set of hard-coded reward configurations (not a full grid sweep).
# - For each configuration, we create N_ENVS (e.g., 32 or 128) parallel environments.
# - For each configuration, its environments share a total pool of TOTAL_STEPS timesteps for training (e.g., 100,000 or 1,000,000).
# - Each action in the environment is immediately assessed for reward using the current config's weights.
# - After training, we evaluate the trained model by playing EVAL_EPISODES (e.g., 128) games per config.
# - Each evaluation game runs until it ends or hits a move limit (e.g., 1000 moves).
# - For each config, we record metrics like average reward, stddev, and possibly other stats (e.g., max tile, score).
# - This allows us to compare how well each reward configuration enables the agent to learn to play 2048.

N_ENVS = 64            # example: 64 parallel environments for each config
TOTAL_STEPS = 1_000_000 # example: the 64 parallel environments will share this pool of 1,000,000 timesteps
EVAL_EPISODES = 256     # example: 256 games for each config. this is the "final exam" after training
DEBUG = True
RESULTS_PATH = "reward_sweep_gpu_results.json"

REWARD_VARIANTS = {
    "baseline": {},
    "tile_boost": {"tile_bonus": 1.5},
    "corner_penalty": {"corner_penalty": True},
    "combo": {"tile_bonus": 1.5, "corner_penalty": True},
}


def atomic_save_json(obj, filename):
    """Safely save JSON to file without risk of corruption"""
    temp_filename = filename + ".tmp"
    with open(temp_filename, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(temp_filename, filename)


def make_env(reward_kwargs):
    def _init():
        env = Game2048Env()
        env.set_reward_weights(**reward_kwargs)
        return env
    return _init


def load_or_init_results(force_fresh=False):
    if not force_fresh and os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def evaluate_with_gpu_batch_inference(model, reward_kwargs, episodes=EVAL_EPISODES):
    rewards = []
    for _ in range(episodes):
        env = Game2048Env()
        env.set_reward_weights(**reward_kwargs)
        obs, _ = env.reset()
        done = False
        total_reward = 0
        moves = 0
        while not done and moves < 1000:
            obs_array = np.array(obs)  # shape (17,)
            with torch.no_grad():
                action, _ = model.predict(obs_array.reshape(1, -1), deterministic=True)
            obs, reward, done, _, _ = env.step(action.item())
            total_reward += reward
            moves += 1
        rewards.append(total_reward)
    return rewards


def run_sweep_variant(sweep_id, reward_kwargs, force_fresh=False):
    if DEBUG:
        print(f"[DEBUG] Running sweep: {sweep_id} ...")

    result_log = load_or_init_results(force_fresh=force_fresh)

    if not force_fresh and sweep_id in result_log:
        print(f"[INFO] Skipping {sweep_id} (already completed)")
        return result_log[sweep_id]

    vec_env = SubprocVecEnv([make_env(reward_kwargs) for _ in range(N_ENVS)])

    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=1e-4,
        buffer_size=200_000,
        batch_size=1024,
        tensorboard_log=None,
        device="cuda"
    )

    model.learn(total_timesteps=TOTAL_STEPS)

    rewards = evaluate_with_gpu_batch_inference(model, reward_kwargs)
    avg_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))

    result_data = {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "reward_kwargs": reward_kwargs,
    }

    print(f"[✓] {sweep_id}: avg {avg_reward:.2f} ± {std_reward:.2f}")

    result_log[sweep_id] = result_data
    atomic_save_json(result_log, RESULTS_PATH)
    return result_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Force fresh sweep results")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    start_time = time.time()
    all_results = {}

    for sweep_id, reward_kwargs in REWARD_VARIANTS.items():
        result = run_sweep_variant(
            sweep_id,
            reward_kwargs=reward_kwargs,
            force_fresh=args.fresh
        )
        all_results[sweep_id] = result

    print("\n--- Sweep Summary ---")
    for k, v in all_results.items():
        print(f"{k}: {v['avg_reward']:.2f} ± {v['std_reward']:.2f}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
