import os
import math
import json
import argparse
import numpy as np
import gymnasium as gym
import torch
from typing import Dict, Any, Tuple
import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# Robust imports: allow running as module or script
try:
    from src.game import Game2048
except ModuleNotFoundError:
    import sys as _sys
    _THIS_DIR = os.path.abspath(os.path.dirname(__file__))
    _PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
    for _p in (_THIS_DIR, _PROJECT_ROOT):
        if _p not in _sys.path:
            _sys.path.append(_p)
    from src.game import Game2048


class GodMode2048Env(gym.Env):
    """Environment optimized for 'god mode' training: reward only for max-tile progress.

    - Observation: log2(board)/11 flattened to shape (16,). No engineered features.
    - Action masking supported via get_action_mask() (invalid moves masked externally).
    - Reward: delta in log2(max_tile), i.e., reward = log2(max_after) - log2(max_before).
      This gives +1 when a new power-of-two milestone is reached, 0 otherwise.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.game = Game2048()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)
        # Curriculum controls
        self._target_tile: int | None = None
        self._terminal_bonus: float = 0.0

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.game = Game2048()
        return self._get_obs(), {}

    def step(self, action: int):
        move = ['up', 'down', 'left', 'right'][int(action)]

        board_before = self.game.board.copy()
        max_tile_before = int(np.max(board_before))
        log_before = math.log2(max_tile_before) if max_tile_before > 0 else 0.0

        try:
            self.game.move(move)
        except ValueError:
            # Should be masked out; return no-op penalty 0 for robustness
            info = {
                "invalid_move": True,
                "max_tile": max_tile_before,
                "action_mask": self.get_action_mask(),
            }
            return self._get_obs(), 0.0, False, False, info

        max_tile_after = int(np.max(self.game.board))
        log_after = math.log2(max_tile_after) if max_tile_after > 0 else 0.0
        reward = float(log_after - log_before)

        done = self.is_game_over()
        reached_target = False
        if self._target_tile is not None and max_tile_after >= int(self._target_tile):
            reward += float(self._terminal_bonus)
            done = True
            reached_target = True
        info = {
            "invalid_move": False,
            "max_tile": max_tile_after,
            "action_mask": self.get_action_mask(),
            "reached_target": reached_target,
        }
        return self._get_obs(), reward, done, False, info

    def _get_obs(self) -> np.ndarray:
        with np.errstate(divide='ignore'):
            obs = np.where(self.game.board > 0, np.log2(self.game.board) / 11, 0).astype(np.float32)
        return obs.flatten()

    def get_action_mask(self) -> np.ndarray:
        mask_bool = self.game.get_valid_action_mask()
        return mask_bool.astype(np.float32)

    def is_game_over(self) -> bool:
        return self.game.is_game_over()

    def render(self):
        print(self.game.board)

    # Curriculum API
    def set_target_tile(self, target: int | None):
        self._target_tile = int(target) if target is not None else None

    def get_target_tile(self) -> int | None:
        return self._target_tile

    def set_terminal_bonus(self, bonus: float):
        self._terminal_bonus = float(bonus)


def mask_fn(env: GodMode2048Env):
    base = getattr(env, 'unwrapped', env)
    if hasattr(base, 'get_action_mask'):
        return base.get_action_mask()
    inner = getattr(env, 'env', None)
    if inner is not None and hasattr(inner, 'get_action_mask'):
        return inner.get_action_mask()
    return env.get_action_mask()


def make_env(max_episode_steps: int):
    def _init():
        env = GodMode2048Env()
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


class GodModeCallback(BaseCallback):
    def __init__(self, check_freq: int = 50_000, verbose: int = 0, total_timesteps_target: int | None = None, progress_interval_sec: int = 60,
                 curriculum: bool = False, target_start: int = 256, promote_rate: float = 0.6, window_episodes: int = 512, terminal_bonus: float = 1.0,
                 entropy_start: float = 0.01, entropy_end: float = 0.005):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.timesteps: list[int] = []
        self.avg_max_tile: list[float] = []
        self.total_timesteps_target = total_timesteps_target
        self.progress_interval_sec = progress_interval_sec
        self._start_time: float = 0.0
        self._last_log_time: float = 0.0
        # Additional curves
        self.positive_reward_rate: list[float] = []
        self.avg_valid_actions: list[float] = []
        # Curriculum
        self.enable_curriculum = curriculum
        self.curr_target = target_start if curriculum else None
        self.curr_promote_rate = promote_rate
        self.curr_window = window_episodes
        self.curr_terminal_bonus = terminal_bonus
        self._ep_window_count = 0
        self._ep_window_success = 0
        # Entropy decay schedule
        self.entropy_start = float(entropy_start)
        self.entropy_end = float(entropy_end)

    def _on_training_start(self) -> None:
        self._reset()
        self._start_time = time.time()
        self._last_log_time = self._start_time
        if self.total_timesteps_target is not None:
            print(f"[GodMode] Training started: target {self.total_timesteps_target:,} steps", flush=True)
        # Initialize curriculum target across envs
        if self.enable_curriculum and self.training_env is not None and self.curr_target is not None:
            try:
                self.training_env.env_method('set_target_tile', self.curr_target)
                self.training_env.env_method('set_terminal_bonus', float(self.curr_terminal_bonus))
                print(f"[GodMode] Curriculum target initialized at {self.curr_target}", flush=True)
            except Exception:
                pass

    def _reset(self):
        self._steps = 0
        self._max_tile_sum = 0.0
        self._positive_reward_steps = 0
        self._valid_actions_sum = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is not None:
            for info in infos:
                if isinstance(info, dict) and "max_tile" in info:
                    self._max_tile_sum += float(info["max_tile"])
                    self._steps += 1
                    # Action mask based valid-action count
                    mask = info.get("action_mask")
                    if mask is not None:
                        try:
                            self._valid_actions_sum += float(np.sum(mask))
                        except Exception:
                            pass
                # Curriculum episode accounting: look for episode termination flags
                if isinstance(info, dict) and (info.get('terminal_observation') is not None or info.get('TimeLimit.truncated') is not None or info.get('reached_target') is not None):
                    # We don't have a reliable per-env done signal here; approximate using info keys
                    if self.enable_curriculum:
                        self._ep_window_count += 1
                        if info.get('reached_target', False):
                            self._ep_window_success += 1
        # Rewards array from vec env
        rewards = self.locals.get("rewards")
        if rewards is not None:
            try:
                self._positive_reward_steps += int(np.sum((rewards > 0).astype(np.int32)))
            except Exception:
                pass

        # Periodic metric snapshot
        if self.num_timesteps > 0 and self.num_timesteps % self.check_freq == 0:
            avg_tile = self._max_tile_sum / max(1, self._steps)
            pos_rate = float(self._positive_reward_steps) / max(1, self._steps)
            avg_valid_actions = self._valid_actions_sum / max(1, self._steps)
            self.timesteps.append(self.num_timesteps)
            self.avg_max_tile.append(avg_tile)
            self.positive_reward_rate.append(pos_rate)
            self.avg_valid_actions.append(avg_valid_actions)
            # Log to TensorBoard if available
            try:
                self.logger.record("custom/avg_max_tile_window", avg_tile)
                self.logger.record("custom/positive_reward_rate", pos_rate)
                self.logger.record("custom/avg_valid_actions", avg_valid_actions)
            except Exception:
                pass
            self._reset()
        # Curriculum promotion check based on episode window
        if self.enable_curriculum and self._ep_window_count >= max(1, self.curr_window):
            success_rate = self._ep_window_success / max(1, self._ep_window_count)
            try:
                self.logger.record("custom/curr_success_rate", success_rate)
            except Exception:
                pass
            if success_rate >= self.curr_promote_rate and self.curr_target is not None:
                next_targets = [256, 512, 1024, 2048, 4096, 8192, 16384]
                try:
                    idx = next_targets.index(int(self.curr_target))
                    if idx + 1 < len(next_targets):
                        self.curr_target = next_targets[idx + 1]
                        if self.training_env is not None:
                            self.training_env.env_method('set_target_tile', self.curr_target)
                            print(f"[GodMode] Curriculum promoted to {self.curr_target}", flush=True)
                except ValueError:
                    pass
            # reset window
            self._ep_window_count = 0
            self._ep_window_success = 0

        # Entropy decay (manual schedule as SB3 expects a float, not a callable)
        if self.total_timesteps_target is not None and self.total_timesteps_target > 0:
            progress_remaining = max(0.0, 1.0 - (float(self.num_timesteps) / float(self.total_timesteps_target)))
            current_ent = self.entropy_end + (self.entropy_start - self.entropy_end) * progress_remaining
            try:
                if hasattr(self.model, 'ent_coef'):
                    self.model.ent_coef = float(current_ent)
                self.logger.record("custom/ent_coef", float(current_ent))
            except Exception:
                pass

        # Periodic progress indicator (at least once per minute)
        now = time.time()
        if now - self._last_log_time >= self.progress_interval_sec:
            elapsed = now - self._start_time
            steps = int(self.num_timesteps)
            sps = steps / max(1e-6, elapsed)
            if self.total_timesteps_target is not None and self.total_timesteps_target > 0:
                pct = min(100.0, 100.0 * steps / self.total_timesteps_target)
                remaining = max(0, self.total_timesteps_target - steps)
                eta_sec = remaining / max(1e-6, sps)
                eta_min = eta_sec / 60.0
                print(f"[GodMode] {steps:,}/{self.total_timesteps_target:,} steps ({pct:5.1f}%) | {sps:,.0f} steps/s | ETA ~ {eta_min:,.1f} min", flush=True)
            else:
                print(f"[GodMode] {steps:,} steps | {sps:,.0f} steps/s | elapsed {elapsed/60.0:,.1f} min", flush=True)
            self._last_log_time = now
        return True


def evaluate(model: MaskablePPO, episodes: int = 256, obs_rms=None, max_episode_steps: int = 1000) -> Dict[str, Any]:
    thresholds = [256, 512, 1024, 2048, 4096, 8192]
    results = {
        "max_tiles": [],
        "moves": [],
        "success_rates": {str(t): 0 for t in thresholds},
    }
    for _ in range(episodes):
        if obs_rms is None:
            env = GodMode2048Env()
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
            env = ActionMasker(env, mask_fn)
            obs, _ = env.reset()
            done = False
            truncated = False
            max_tile = 0
            steps = 0
            while not (done or truncated):
                mask = mask_fn(env)
                with torch.no_grad():
                    action, _ = model.predict(np.array(obs).reshape(1, -1), deterministic=True, action_masks=mask)
                action_idx = int(np.asarray(action).item())
                obs, reward, done, truncated, info = env.step(action_idx)
                max_tile = max(max_tile, info.get("max_tile", 0))
                steps += 1
            results["max_tiles"].append(max_tile)
            results["moves"].append(steps)
            for t in thresholds:
                if max_tile >= t:
                    results["success_rates"][str(t)] += 1
            try:
                env.close()
            except Exception:
                pass
        else:
            # Use VecNormalize with loaded obs stats
            venv = DummyVecEnv([make_env(max_episode_steps)])
            venv = VecNormalize(venv, norm_obs=True, norm_reward=False, training=False)
            venv.obs_rms = obs_rms
            obs = venv.reset()
            done_vec = [False]
            max_tile = 0
            steps = 0
            while not done_vec[0]:
                # Derive mask from the underlying env
                base_env = venv.venv.envs[0]
                try:
                    mask = mask_fn(base_env)
                except Exception:
                    mask = np.ones(4, dtype=np.float32)
                with torch.no_grad():
                    action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                action_idx = int(np.asarray(action).item())
                obs, rewards, done_vec, infos = venv.step([action_idx])
                info = infos[0]
                max_tile = max(max_tile, info.get("max_tile", 0))
                steps += 1
            results["max_tiles"].append(max_tile)
            results["moves"].append(steps)
            for t in thresholds:
                if max_tile >= t:
                    results["success_rates"][str(t)] += 1
            try:
                venv.close()
            except Exception:
                pass
    # Normalize success counts to rates
    for k in list(results["success_rates"].keys()):
        results["success_rates"][k] /= float(episodes)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Ignore previous results and overwrite")
    parser.add_argument("--num-envs", type=int, default=max(16, (os.cpu_count() or 20)), help="Number of parallel envs")
    parser.add_argument("--subproc", action="store_true", help="Use SubprocVecEnv for multi-core stepping")
    parser.add_argument("--total-steps", type=int, default=2_000_000, help="Total timesteps")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO n_steps (rollout length per env)")
    parser.add_argument("--batch-size", type=int, default=65_536, help="PPO batch size")
    parser.add_argument("--arch", type=str, default="1024,1024", help="Policy hidden layers, comma-separated")
    parser.add_argument("--eval-episodes", type=int, default=512, help="Evaluation episodes")
    parser.add_argument("--results", type=str, default="god_mode_results.json", help="Output JSON path")
    # PPO hyperparameters exposed for tuning
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of PPO epochs per update")
    parser.add_argument("--entropy-start", type=float, default=0.01, help="Entropy coef start (schedule)")
    parser.add_argument("--entropy-end", type=float, default=0.005, help="Entropy coef end (schedule)")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--tb", type=str, default="runs/god_mode", help="TensorBoard log dir")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--vecnorm", action="store_true", help="Enable VecNormalize for observations")
    parser.add_argument("--episode-steps", type=int, default=2000, help="Max steps per episode")
    # Curriculum flags
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum on target tile")
    parser.add_argument("--curr-target-start", type=int, default=256, help="Initial curriculum target tile")
    parser.add_argument("--curr-promote", type=float, default=0.6, help="Promotion success rate threshold")
    parser.add_argument("--curr-window", type=int, default=512, help="Episodes per curriculum window")
    parser.add_argument("--curr-bonus", type=float, default=1.0, help="Terminal bonus when reaching target")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    def make_vec_env(num_envs: int):
        if args.subproc:
            return SubprocVecEnv([make_env(args.episode_steps) for _ in range(num_envs)], start_method="spawn")
        return DummyVecEnv([make_env(args.episode_steps) for _ in range(num_envs)])

    vec_env = make_vec_env(args.num_envs)
    if args.seed is not None:
        try:
            vec_env.seed(args.seed)
        except Exception:
            pass
    arch = tuple(int(x) for x in args.arch.split(",") if x.strip())

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=float(args.entropy_start),
        vf_coef=args.vf_coef,
        clip_range=args.clip_range,
        gamma=args.gamma,
        device="cuda",
        policy_kwargs={"net_arch": list(arch)},
        tensorboard_log=args.tb,
    )

    callback = GodModeCallback(
        check_freq=max(50_000, args.n_steps * args.num_envs),
        total_timesteps_target=args.total_steps,
        progress_interval_sec=60,
        curriculum=args.curriculum,
        target_start=args.curr_target_start,
        promote_rate=args.curr_promote,
        window_episodes=args.curr_window,
        terminal_bonus=args.curr_bonus,
        entropy_start=args.entropy_start,
        entropy_end=args.entropy_end,
    )
    model.learn(total_timesteps=args.total_steps, callback=callback)

    obs_rms = vec_env.obs_rms if isinstance(vec_env, VecNormalize) else None
    metrics = evaluate(model, episodes=args.eval_episodes, obs_rms=obs_rms, max_episode_steps=args.episode_steps)
    summary = {
        "avg_max_tile": float(np.mean(metrics["max_tiles"])),
        "p95_max_tile": float(np.percentile(metrics["max_tiles"], 95)),
        "avg_moves": float(np.mean(metrics["moves"])),
        "success_rates": metrics["success_rates"],
        "train_curve": {
            "timesteps": callback.timesteps,
            "avg_max_tile": callback.avg_max_tile,
            "positive_reward_rate": callback.positive_reward_rate,
            "avg_valid_actions": callback.avg_valid_actions,
        },
        "config": {
            "num_envs": args.num_envs,
            "subproc": args.subproc,
            "total_steps": args.total_steps,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "arch": list(arch),
            "lr": args.lr,
            "n_epochs": args.n_epochs,
            "entropy_start": args.entropy_start,
            "entropy_end": args.entropy_end,
            "vf_coef": args.vf_coef,
            "clip_range": args.clip_range,
            "tensorboard_log": args.tb,
            "seed": args.seed,
            "gamma": args.gamma,
            "vecnorm": args.vecnorm,
            "episode_steps": args.episode_steps,
            "curriculum": args.curriculum,
            "curr_target_start": args.curr_target_start,
            "curr_promote": args.curr_promote,
            "curr_window": args.curr_window,
            "curr_bonus": args.curr_bonus,
        },
    }

    # Atomic save
    tmp = args.results + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, args.results)

    print(
        f"[GodMode] avg max tile: {summary['avg_max_tile']:.1f} | p95: {summary['p95_max_tile']:.1f} | "
        + " ".join([f">={k}: {v*100:.1f}%" for k, v in summary['success_rates'].items()])
    )


if __name__ == "__main__":
    main()


