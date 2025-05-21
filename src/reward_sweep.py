import itertools
import numpy as np
from src.sb3_train import PlottingCallback
from src.gym_env import Game2048Env
import csv
import json
import plotly.graph_objs as go
import plotly.offline as pyo
import time
import os
import tempfile
import shutil
from stable_baselines3 import DQN

# --- Utility functions: clean_for_json and atomic_save_json ---
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items() if k not in ('env_ref', 'callback', 'model', 'env', 'eval_env')}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)

def atomic_save_json(obj, filename):
    with tempfile.NamedTemporaryFile('w', delete=False, dir='.') as tf:
        json.dump(obj, tf, indent=2)
        tempname = tf.name
    shutil.move(tempname, filename)

# --- rich dashboard helpers (after all utility function defs, before sweep logic) ---
def try_import_rich():
    try:
        from rich.console import Console
        from rich.table import Table
        return Console, Table
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rich'])
        from rich.console import Console
        from rich.table import Table
        return Console, Table

Console, Table = try_import_rich()

def print_sweep_status(results, sweep_ids, step_dict, reward_dict, best_so_far_dict, stopped_dict):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Sweep ID")
    table.add_column("Steps", justify="right")
    table.add_column("Mean Reward", justify="right")
    table.add_column("Best So Far", justify="right")
    table.add_column("Stopped", justify="center")
    for sid in sweep_ids:
        table.add_row(
            sid,
            str(step_dict.get(sid, "")),
            f"{reward_dict.get(sid, 0):.2f}",
            f"{best_so_far_dict.get(sid, 0):.2f}",
            "✅" if stopped_dict.get(sid, False) else ""
        )
    console.clear()
    console.print(table)

# Restore a full grid with multiple values for each parameter
grid = {
    # 'max_in_corner': [5.0, 10.0], 
    # 'not_in_corner': [-2.0, -5.0],
    # 'moved_out_of_corner': [-30.0, -50.0],
    # 'max_tile_up': [0.0, 10.0],  # 0.0 disables this reward
    # 'combine_top_two': [10.0, 15.0, 20.0],  # focus on higher weights
    # 'monotonicity_weight': [2.0, 4.0],

    'max_in_corner': [5.0, 10.0], 
    'not_in_corner': [-1.0, -3.0],
    'moved_out_of_corner': [-20.0],
    'max_tile_up': [0.0, 10.0],  # 0.0 disables this reward
    'combine_top_two': [10.0, 15.0, 20.0],  # focus on higher weights
    'monotonicity_weight': [2.0],
}

param_names = list(grid.keys())
param_values = list(grid.values())

results = {}

INIT_STEPS = 20_000
CHUNK_STEPS = 20_000
MIN_STEPS = 40_000
MAX_STEPS = 250_000
CUTOFF_FRAC = 0.5
EVAL_EPISODES = 50  # Increased for stability

# Add a flag to force a fresh sweep (ignore old results and delete old files)
import glob
FORCE_FRESH_SWEEP = True  # Set to True to always start fresh

if FORCE_FRESH_SWEEP:
    # Remove results file and all sweep init zips
    for fname in glob.glob('reward_sweep_full_results*.json'):
        try:
            os.remove(fname)
        except Exception:
            pass
    for fname in glob.glob('sweep_*_init.zip'):
        try:
            os.remove(fname)
        except Exception:
            pass
    results = {}
else:
    if os.path.exists('reward_sweep_full_results.json'):
        with open('reward_sweep_full_results.json') as f:
            results = json.load(f)

total_start_time = time.time()

for i, values in enumerate(itertools.product(*param_values)):
    params = dict(zip(param_names, values))
    sweep_id = f'sweep_{i+1}'
    if sweep_id in results and results[sweep_id].get('init_done'):
        continue
    print(f"\n[INIT] {sweep_id}: {params}")
    env = Game2048Env()
    env.set_reward_weights(**params)
    callback = PlottingCallback(plot_interval=INIT_STEPS, env_ref=env, sweep_id=sweep_id, show_plot=False)
    model = DQN('MlpPolicy', env, learning_rate=1e-3, batch_size=256, buffer_size=50000, learning_starts=1000, train_freq=4, target_update_interval=1000, device='cpu')
    model.learn(total_timesteps=INIT_STEPS, callback=callback)
    model.save(f"{sweep_id}_init.zip")

    eval_rewards, eval_max_tiles = [], []
    eval_env = Game2048Env()
    eval_env.set_reward_weights(**params)
    for _ in range(EVAL_EPISODES):
        obs, _ = eval_env.reset()
        done, total_reward, max_tile = False, 0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = eval_env.step(action)
            total_reward += reward
            max_tile = max(max_tile, np.max(eval_env.game.board))
        eval_rewards.append(total_reward)
        eval_max_tiles.append(max_tile)
    results[sweep_id] = {'eval_rewards': eval_rewards, 'eval_max_tiles': eval_max_tiles, 'params': params, 'init_done': True}
    atomic_save_json(clean_for_json(results), 'reward_sweep_full_results.json')
    elapsed = time.time() - total_start_time
    print(f"[INIT] {sweep_id} complete. Elapsed time: {elapsed/60:.1f} min ({elapsed:.1f} sec)")

batting_order = sorted(results.keys(), key=lambda k: np.mean(results[k]['eval_rewards']), reverse=True)

best_so_far = float('-inf')
# Track sweep progress for dashboard
total_step_dict = {sid: INIT_STEPS for sid in batting_order}
mean_reward_dict = {sid: np.mean(results[sid]['eval_rewards']) for sid in batting_order}
best_so_far_dict = {sid: float('-inf') for sid in batting_order}
stopped_dict = {sid: False for sid in batting_order}

for sweep_id in batting_order:
    if results[sweep_id].get('adaptive_done'):
        continue
    params = results[sweep_id]['params']
    print(f"\n[ADAPTIVE] {sweep_id}: {params}")
    env = Game2048Env()
    env.set_reward_weights(**params)
    callback = PlottingCallback(plot_interval=CHUNK_STEPS, env_ref=env, sweep_id=sweep_id, show_plot=False)
    model = DQN.load(f"{sweep_id}_init.zip", env=env, device='cpu')
    total_steps = INIT_STEPS
    all_rewards, all_max_tiles = results[sweep_id]['eval_rewards'], results[sweep_id]['eval_max_tiles']

    while total_steps < MAX_STEPS:
        model.learn(total_timesteps=CHUNK_STEPS, callback=callback, reset_num_timesteps=False)
        total_steps += CHUNK_STEPS
        eval_rewards, eval_max_tiles = [], []
        eval_env = Game2048Env()
        eval_env.set_reward_weights(**params)
        for _ in range(EVAL_EPISODES):
            obs, _ = eval_env.reset()
            done, total_reward, max_tile = False, 0, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = eval_env.step(action)
                total_reward += reward
                max_tile = max(max_tile, np.max(eval_env.game.board))
            eval_rewards.append(total_reward)
            eval_max_tiles.append(max_tile)

        mean_reward = np.mean(eval_rewards)
        best_so_far = max(best_so_far, mean_reward)
        best_so_far_dict[sweep_id] = max(best_so_far_dict[sweep_id], mean_reward)
        mean_reward_dict[sweep_id] = mean_reward
        total_step_dict[sweep_id] = total_steps
        print_sweep_status(results, batting_order, total_step_dict, mean_reward_dict, best_so_far_dict, stopped_dict)
        elapsed = time.time() - total_start_time
        print(f"{sweep_id}: steps={total_steps}, mean_reward={mean_reward:.2f}, best_so_far={best_so_far:.2f}")
        print(f"{sweep_id}: Elapsed time: {elapsed/60:.1f} min ({elapsed:.1f} sec)")

        if total_steps >= MIN_STEPS and mean_reward < CUTOFF_FRAC * best_so_far:
            print(f"{sweep_id}: Stopped early (mean_reward<{CUTOFF_FRAC*100:.0f}% best)")
            stopped_dict[sweep_id] = True
            print_sweep_status(results, batting_order, total_step_dict, mean_reward_dict, best_so_far_dict, stopped_dict)
            break
        all_rewards.extend(eval_rewards)
        all_max_tiles.extend(eval_max_tiles)

    results[sweep_id].update({'eval_rewards': all_rewards, 'eval_max_tiles': all_max_tiles, 'adaptive_done': True})
    atomic_save_json(clean_for_json(results), 'reward_sweep_full_results.json')
    elapsed = time.time() - total_start_time
    print(f"[ADAPTIVE] {sweep_id} complete. Elapsed time: {elapsed/60:.1f} min ({elapsed:.1f} sec)")
    print_sweep_status(results, batting_order, total_step_dict, mean_reward_dict, best_so_far_dict, stopped_dict)

final_elapsed = time.time() - total_start_time
print(f"\nTotal sweep time elapsed: {final_elapsed/60:.1f} min ({final_elapsed:.1f} sec)")
