import os
import torch
import json
import numpy as np
import multiprocessing as mp
import argparse
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
# Support both package and script execution
try:
    from src.gym_env import Game2048Env
    from src.timer_utils import Timer
except ModuleNotFoundError:
    import sys as _sys
    _THIS_DIR = os.path.abspath(os.path.dirname(__file__))
    _PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
    for _p in (_THIS_DIR, _PROJECT_ROOT):
        if _p not in _sys.path:
            _sys.path.append(_p)
    try:
        from src.gym_env import Game2048Env
        from src.timer_utils import Timer
    except ModuleNotFoundError:
        from gym_env import Game2048Env
        from timer_utils import Timer
import plotly.graph_objects as go
from plotly.subplots import make_subplots



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

N_ENVS = 32            # default parallel environments per config
TOTAL_STEPS = 1_000_000 # total timesteps per config
EVAL_EPISODES = 256     # evaluation episodes per config
DEBUG = True
RESULTS_PATH = "reward_sweep_gpu_results.json"
PLOTS_DIR = "plots"
MAX_EPISODE_STEPS = 1000

REWARD_VARIANTS = {
    # Baseline as defined in Game2048Env
    "baseline": {},

    # Emphasize merges (bigger boosts for creating larger tiles), slightly de-emphasize empties
    "merge_emphasis": {"merge_power": 2.0, "empty_tile": 0.05},

    # Encourage keeping the max tile anchored in the corner
    "corner_strategy": {"corner_bonus": 25.0, "corner_penalty": -15.0},

    # Encourage mobility (more empty tiles) and reduce per-step penalty to allow longer planning
    "mobility_strategy": {"empty_tile": 0.4, "step_penalty": -0.25},
}


def atomic_save_json(obj, filename):
    """Safely save JSON to file without risk of corruption"""
    temp_filename = filename + ".tmp"
    with open(temp_filename, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(temp_filename, filename)


def mask_fn(env: Game2048Env):
    # Unwrap common gym wrappers to reach the base env exposing get_action_mask
    base = getattr(env, 'unwrapped', env)
    if hasattr(base, 'get_action_mask'):
        return base.get_action_mask()
    inner = getattr(env, 'env', None)
    if inner is not None and hasattr(inner, 'get_action_mask'):
        return inner.get_action_mask()
    # Fallback: try the original attribute and let it raise if missing
    return env.get_action_mask()


def make_env(reward_kwargs):
    def _init():
        env = Game2048Env()
        env.set_reward_weights(**reward_kwargs)
        # Enforce a step cap with proper truncated flag
        env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


def load_or_init_results(force_fresh=False):
    if not force_fresh and os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def evaluate_with_gpu_batch_inference(model, reward_kwargs, episodes=EVAL_EPISODES):
    rewards = []
    max_tiles = []
    num_moves_list = []
    invalid_move_rates = []
    cap_hit_flags = []
    avg_empty_tiles_per_step = []
    corner_hold_rates = []
    for _ in range(episodes):
        env = Game2048Env()
        env.set_reward_weights(**reward_kwargs)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env = ActionMasker(env, mask_fn)
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        moves = 0
        max_tile = 0
        invalid_moves = 0
        empty_tiles_sum = 0.0
        max_in_corner_sum = 0.0
        while not (done or truncated):
            obs_array = np.array(obs)  # shape (17,)
            mask = mask_fn(env)
            with torch.no_grad():
                action, _ = model.predict(obs_array.reshape(1, -1), deterministic=True, action_masks=mask)
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
            max_tile = max(max_tile, info.get("max_tile", 0))
            if info.get("invalid_move", False):
                invalid_moves += 1
            empty_tiles_sum += float(info.get("empty_tiles", 0))
            max_in_corner_sum += 1.0 if bool(info.get("max_in_corner", False)) else 0.0
            moves += 1
        rewards.append(total_reward)
        max_tiles.append(max_tile)
        num_moves_list.append(moves)
        invalid_move_rate = (invalid_moves / moves) if moves > 0 else 0.0
        invalid_move_rates.append(invalid_move_rate)
        cap_hit_flags.append(bool(truncated))
        avg_empty_tiles_per_step.append(empty_tiles_sum / max(1, moves))
        corner_hold_rates.append(max_in_corner_sum / max(1, moves))
        # Explicitly close env per episode to avoid descriptor/memory buildup
        try:
            env.close()
        except Exception:
            pass

    return {
        "rewards": rewards, 
        "max_tiles": max_tiles,
        "num_moves_list": num_moves_list,
        "invalid_move_rates": invalid_move_rates,
        "cap_hit_flags": cap_hit_flags,
        "avg_empty_tiles_per_step": avg_empty_tiles_per_step,
        "corner_hold_rates": corner_hold_rates,
    }


class InvalidMoveTrackingCallback(BaseCallback):
    def __init__(self, check_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.timesteps = []
        self.invalid_move_rates = []
        self.empty_tiles_avgs = []
        self.max_in_corner_rates = []
        self.max_tile_avgs = []

    def _on_training_start(self) -> None:
        self._reset_accumulators()

    def _reset_accumulators(self):
        self._steps_in_window = 0
        self._invalid_in_window = 0
        self._empty_tiles_sum = 0.0
        self._max_in_corner_sum = 0.0
        self._max_tile_sum = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is not None:
            # VecEnv provides a list of info dicts
            for info in infos:
                if isinstance(info, dict):
                    if info.get("invalid_move", False):
                        self._invalid_in_window += 1
                    # Aggregate additional metrics
                    if "empty_tiles" in info:
                        self._empty_tiles_sum += float(info["empty_tiles"])  # 0..16
                    if "max_in_corner" in info:
                        self._max_in_corner_sum += 1.0 if bool(info["max_in_corner"]) else 0.0
                    if "max_tile" in info:
                        self._max_tile_sum += float(info["max_tile"])  # actual tile value
                    self._steps_in_window += 1

        if self.num_timesteps > 0 and self.num_timesteps % self.check_freq == 0:
            rate = (self._invalid_in_window / max(1, self._steps_in_window))
            self.timesteps.append(self.num_timesteps)
            self.invalid_move_rates.append(rate)
            self.empty_tiles_avgs.append(self._empty_tiles_sum / max(1, self._steps_in_window))
            self.max_in_corner_rates.append(self._max_in_corner_sum / max(1, self._steps_in_window))
            self.max_tile_avgs.append(self._max_tile_sum / max(1, self._steps_in_window))
            self._reset_accumulators()
        return True

def run_sweep_variant(
    sweep_id,
    reward_kwargs,
    force_fresh=False,
    num_envs: int = N_ENVS,
    use_subproc: bool = False,
    ppo_n_steps: int = 2048,
    ppo_batch_size: int = 4096,
    policy_arch=(256, 256),
):
    if DEBUG:
        print(f"[DEBUG] Running sweep: {sweep_id} ...")

    result_log = load_or_init_results(force_fresh=force_fresh)

    if not force_fresh and sweep_id in result_log:
        print(f"[INFO] Skipping {sweep_id} (already completed)")
        return result_log[sweep_id]

    if use_subproc:
        vec_env = SubprocVecEnv([make_env(reward_kwargs) for _ in range(num_envs)], start_method="spawn")
    else:
        vec_env = DummyVecEnv([make_env(reward_kwargs) for _ in range(num_envs)])

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=1e-4,
        n_steps=ppo_n_steps,
        batch_size=ppo_batch_size,
        tensorboard_log=None,
        device="cuda",
        policy_kwargs={"net_arch": list(policy_arch)},
    )

    callback = InvalidMoveTrackingCallback(check_freq=50_000)
    model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

    metrics = evaluate_with_gpu_batch_inference(model, reward_kwargs)
    avg_reward = float(np.mean(metrics["rewards"]))
    std_reward = float(np.std(metrics["rewards"]))
    avg_max_tile = float(np.mean(metrics["max_tiles"]))
    avg_moves = float(np.mean(metrics["num_moves_list"]))
    avg_invalid_rate = float(np.mean(metrics["invalid_move_rates"]))
    cap_hit_rate = float(np.mean(metrics["cap_hit_flags"]))
    avg_empty_tiles_eval = float(np.mean(metrics["avg_empty_tiles_per_step"]))
    corner_hold_rate_eval = float(np.mean(metrics["corner_hold_rates"]))

    result_data = {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_max_tile": avg_max_tile,
        "avg_moves": avg_moves,
        "avg_invalid_rate_eval": avg_invalid_rate,
        "cap_hit_rate_eval": cap_hit_rate,
        "avg_empty_tiles_per_step_eval": avg_empty_tiles_eval,
        "corner_hold_rate_eval": corner_hold_rate_eval,
        "reward_kwargs": reward_kwargs,
        "train_metrics_curve": {
            "timesteps": getattr(callback, "timesteps", []),
            "invalid_move_rates": getattr(callback, "invalid_move_rates", []),
            "empty_tiles_avgs": getattr(callback, "empty_tiles_avgs", []),
            "max_in_corner_rates": getattr(callback, "max_in_corner_rates", []),
            "max_tile_avgs": getattr(callback, "max_tile_avgs", []),
        },
    }

    print(
        f"[✓] {sweep_id}: avg {avg_reward:.2f} ± {std_reward:.2f} | max tile: {avg_max_tile:.1f} | "
        f"moves: {avg_moves:.1f} | eval invalid%: {avg_invalid_rate*100:.1f}% | cap-hit: {cap_hit_rate*100:.1f}%"
    )

    result_log[sweep_id] = result_data
    atomic_save_json(result_log, RESULTS_PATH)

    # Cleanup between variants to reduce memory pressure
    try:
        vec_env.close()
    except Exception:
        pass
    del model
    import gc, torch as _torch
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    return result_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Force fresh sweep results")
    parser.add_argument("--num-envs", type=int, default=max(8, (os.cpu_count() or 8) // 2), help="Number of parallel envs")
    parser.add_argument("--subproc", action="store_true", help="Use SubprocVecEnv for multi-core CPU stepping")
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO n_steps (rollout length per env)")
    parser.add_argument("--batch-size", type=int, default=8192, help="PPO batch_size for updates")
    parser.add_argument("--arch", type=str, default="256,256", help="Policy hidden layers, comma-separated")
    args = parser.parse_args()

    # Reduce CPU threading footprint; helps memory and stability under many envs
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with Timer("Total runtime"):
        all_results = {}

        for sweep_id, reward_kwargs in REWARD_VARIANTS.items():
            arch = tuple(int(x) for x in args.arch.split(",") if x.strip())
            result = run_sweep_variant(
                sweep_id,
                reward_kwargs=reward_kwargs,
                force_fresh=args.fresh,
                num_envs=args.num_envs,
                use_subproc=args.subproc,
                ppo_n_steps=args.n_steps,
                ppo_batch_size=args.batch_size,
                policy_arch=arch,
            )
            all_results[sweep_id] = result

        print("\n--- Sweep Summary ---")
        for k, v in all_results.items():
            print(f"{k}: {v['avg_reward']:.2f} ± {v['std_reward']:.2f}")

        # Plot training metrics curves per variant (assumes Plotly installed)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            "Invalid Move Rate",
            "Avg Empty Tiles per Step",
            "Max-in-Corner Rate",
            "Avg Max Tile Value",
        ))
        for sweep_id, result in all_results.items():
            tm = result.get("train_metrics_curve", {})
            # Backward-compat: support older result files
            if not tm:
                tm = {
                    "timesteps": result.get("train_invalid_curve", {}).get("timesteps", []),
                    "invalid_move_rates": result.get("train_invalid_curve", {}).get("invalid_move_rates", []),
                    "empty_tiles_avgs": [],
                    "max_in_corner_rates": [],
                    "max_tile_avgs": [],
                }
            x = tm.get("timesteps", [])
            if len(x) == 0:
                continue
            fig.add_trace(go.Scatter(x=x, y=tm.get("invalid_move_rates", []), mode="lines+markers", name=f"{sweep_id}"), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=tm.get("empty_tiles_avgs", []), mode="lines+markers", name=f"{sweep_id}"), row=1, col=2)
            fig.add_trace(go.Scatter(x=x, y=tm.get("max_in_corner_rates", []), mode="lines+markers", name=f"{sweep_id}"), row=2, col=1)
            fig.add_trace(go.Scatter(x=x, y=tm.get("max_tile_avgs", []), mode="lines+markers", name=f"{sweep_id}"), row=2, col=2)
        fig.update_layout(title="Training Metrics", template="plotly_white")
        fig.update_xaxes(title_text="Timesteps", row=1, col=1)
        fig.update_xaxes(title_text="Timesteps", row=1, col=2)
        fig.update_xaxes(title_text="Timesteps", row=2, col=1)
        fig.update_xaxes(title_text="Timesteps", row=2, col=2)
        fig.update_yaxes(title_text="Rate", row=1, col=1)
        fig.update_yaxes(title_text="Avg Empty Tiles", row=1, col=2)
        fig.update_yaxes(title_text="Rate", row=2, col=1)
        fig.update_yaxes(title_text="Avg Max Tile", row=2, col=2)
        html_path = os.path.join(PLOTS_DIR, "invalid_move_rate_training.html")
        fig.write_html(html_path)
        # Try to auto-open the plot in the default browser
        try:
            import webbrowser, os as _os
            abs_path = _os.path.abspath(html_path)
            print(f"[INFO] Opening plot: {abs_path}")
            webbrowser.open(f"file://{abs_path}")
        except Exception as e:
            print(f"[WARN] Could not auto-open plot: {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
