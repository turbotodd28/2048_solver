#!/usr/bin/env python3
"""
Reward sweep without premature stopping - allows full training for all configurations
"""

import os
import time
import json
import glob
import itertools
import numpy as np
from stable_baselines3 import DQN
from .gym_env import Game2048Env
from .sb3_train import PlottingCallback

def clean_for_json(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

def atomic_save_json(obj, filename):
    """Save JSON atomically to avoid corruption"""
    temp_filename = filename + '.tmp'
    with open(temp_filename, 'w') as f:
        json.dump(obj, f, indent=2)
    os.rename(temp_filename, filename)

def try_import_rich():
    """Try to import rich for better console output"""
    try:
        from rich.console import Console
        from rich.table import Table
        return Console(), Table
    except ImportError:
        return None, None

def print_sweep_status(results, sweep_ids, step_dict, reward_dict, best_so_far_dict, stopped_dict):
    """Print current status of all sweeps"""
    console, Table = try_import_rich()
    if console is None:
        print("Install rich for better status display: pip install rich")
        return
    
    table = Table(title="Reward Sweep Status")
    table.add_column("Sweep", style="cyan")
    table.add_column("Merge Power", style="magenta")
    table.add_column("Steps", style="green")
    table.add_column("Mean Reward", style="yellow")
    table.add_column("Best So Far", style="blue")
    table.add_column("Stopped", style="red")
    
    for sweep_id in sweep_ids:
        params = results[sweep_id]['params']
        merge_power = params.get('merge_power', 'N/A')
        steps = step_dict.get(sweep_id, 0)
        mean_reward = reward_dict.get(sweep_id, 0)
        best_so_far = best_so_far_dict.get(sweep_id, 0)
        stopped = "Yes" if stopped_dict.get(sweep_id, False) else "No"
        
        table.add_row(sweep_id, str(merge_power), str(steps), 
                     f"{mean_reward:.1f}", f"{best_so_far:.1f}", stopped)
    
    console.clear()
    console.print(table)

# --- Grid definition ---
grid = {
    'empty_tile': [1.25],
    'corner_bonus': [10.0],
    'corner_penalty': [-2.0],
    'merge_power': [2.1, 2.2, 2.25, 2.3, 2.4],
    'milestone_1024': [0],
    'milestone_2048': [0],
    'milestone_4096': [0],
}

param_names = list(grid.keys())
param_values = list(grid.values())

total_sweeps = 1
for v in param_values:
    total_sweeps *= len(v)
print(f"Running reward sweep with {total_sweeps} configurations.")

results = {}

# --- Sweep settings (MODIFIED: No premature stopping) ---
INIT_STEPS = 20_000
CHUNK_STEPS = 20_000
MIN_STEPS = 40_000
MAX_STEPS = 250_000
# CUTOFF_FRAC = 0.5  # REMOVED: No more premature stopping
EVAL_EPISODES = 20

# --- Reset flag ---
FORCE_FRESH_SWEEP = True
if FORCE_FRESH_SWEEP:
    for fname in glob.glob('reward_sweep_full_results*.json'):
        try: os.remove(fname)
        except Exception: pass
    for fname in glob.glob('sweep_*_init.zip'):
        try: os.remove(fname)
        except Exception: pass
    results = {}
elif os.path.exists('reward_sweep_full_results.json'):
    with open('reward_sweep_full_results.json') as f:
        results = json.load(f)

# --- Grid Search Loop ---
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
    model = DQN('MlpPolicy', env, learning_rate=1e-3, batch_size=256, buffer_size=50000,
                learning_starts=1000, train_freq=4, target_update_interval=1000, device='cpu')
    model.learn(total_timesteps=INIT_STEPS, callback=callback)
    model.save(f"{sweep_id}_init.zip")

    # Evaluate
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
    results[sweep_id] = {
        'eval_rewards': eval_rewards,
        'eval_max_tiles': eval_max_tiles,
        'params': params,
        'init_done': True
    }
    atomic_save_json(clean_for_json(results), 'reward_sweep_full_results.json')
    elapsed = time.time() - total_start_time
    print(f"[INIT] {sweep_id} complete. Elapsed time: {elapsed/60:.1f} min")

# --- Adaptive extension loop (MODIFIED: No premature stopping) ---
batting_order = sorted(results.keys(), key=lambda k: np.mean(results[k]['eval_rewards']), reverse=True)

best_so_far = float('-inf')
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

    # MODIFIED: Train for full duration without premature stopping
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

        # REMOVED: No more premature stopping logic
        # if total_steps >= MIN_STEPS and mean_reward < CUTOFF_FRAC * best_so_far:
        #     print(f"{sweep_id}: Stopped early (mean_reward < {CUTOFF_FRAC*100:.0f}% of best)")
        #     stopped_dict[sweep_id] = True
        #     print_sweep_status(results, batting_order, total_step_dict, mean_reward_dict, best_so_far_dict, stopped_dict)
        #     break
        
        all_rewards.extend(eval_rewards)
        all_max_tiles.extend(eval_max_tiles)

    results[sweep_id].update({
        'eval_rewards': all_rewards,
        'eval_max_tiles': all_max_tiles,
        'adaptive_done': True
    })
    atomic_save_json(clean_for_json(results), 'reward_sweep_full_results.json')
    elapsed = time.time() - total_start_time
    print(f"[ADAPTIVE] {sweep_id} complete. Elapsed time: {elapsed/60:.1f} min")
    print_sweep_status(results, batting_order, total_step_dict, mean_reward_dict, best_so_far_dict, stopped_dict)

final_elapsed = time.time() - total_start_time
print(f"\nTotal sweep time elapsed: {final_elapsed/60:.1f} min") 