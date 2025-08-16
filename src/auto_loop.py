import argparse
import datetime as dt
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    EventAccumulator = None  # TensorBoard not available; analysis will be limited


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def launch_tensorboard(logdir: str, port: int = 6006):
    if not is_port_open(port):
        cmd = f"tensorboard --logdir {shlex.quote(logdir)} --port {port}"
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        time.sleep(1.0)


def read_tb_scalars(run_dir: Path, tags: list[str]) -> dict:
    result: dict = {}
    if EventAccumulator is None:
        return result
    try:
        ev_files = list(run_dir.glob("events.*"))
        if not ev_files:
            return result
        ea = EventAccumulator(str(run_dir), size_guidance={'scalars': 0})
        ea.Reload()
        for tag in tags:
            try:
                s = ea.Scalars(tag)
                if s:
                    result[tag] = {"first": s[0].value, "last": s[-1].value, "step": s[-1].step}
            except KeyError:
                pass
    except Exception:
        pass
    return result


def decide_next_params(metrics: dict, params: dict) -> dict:
    """Heuristic adjustments based on last scalars.
    metrics: dict of tag -> {last, first, step}
    params: current run params to adjust
    """
    approx_kl = metrics.get('train/approx_kl', {}).get('last')
    clip_frac = metrics.get('train/clip_fraction', {}).get('last')
    curr_sr = metrics.get('custom/curr_success_rate', {}).get('last')
    avg_tile = metrics.get('custom/avg_max_tile_window', {}).get('last')
    ent_coef = metrics.get('custom/ent_coef', {}).get('last')

    new_params = dict(params)

    # Default gentle changes
    lr = float(new_params['lr'])
    n_epochs = int(new_params['n_epochs'])
    ent_start = float(new_params['entropy_start'])
    ent_end = float(new_params['entropy_end'])

    # If updates are too weak: raise lr and/or epochs
    if approx_kl is not None and clip_frac is not None and approx_kl < 0.002 and clip_frac < 0.02:
        lr = min(lr * 1.5, 1e-4)
        n_epochs = min(n_epochs + 5, 30)

    # If success rate is zero for the window: ease curriculum and keep exploration higher
    if curr_sr is not None and curr_sr <= 0.0:
        new_params['curr_target_start'] = max(128, int(new_params.get('curr_target_start', 256)))
        new_params['curr_promote'] = max(0.4, float(new_params.get('curr_promote', 0.5)))
        new_params['curr_window'] = min(256, int(new_params.get('curr_window', 256)))
        ent_start = max(ent_start, ent_end + 0.01)

    # If updates are too strong: back off
    if approx_kl is not None and approx_kl > 0.03:
        lr = max(lr * 0.7, 1e-5)
        n_epochs = max(n_epochs - 5, 10)

    new_params['lr'] = lr
    new_params['n_epochs'] = n_epochs
    new_params['entropy_start'] = ent_start
    new_params['entropy_end'] = ent_end
    return new_params


def run_once(args, base_params: dict, iter_idx: int) -> tuple[dict, dict, Path]:
    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.run_prefix}_it{iter_idx}_{ts}"
    tb_dir = Path(args.tensorboard_root) / run_name
    results_dir = Path(args.results_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{run_name}.json"

    # Build command
    cmd_parts = [
        sys.executable, "-m", "src.train_god_mode",
        "--subproc" if args.subproc else "",
        "--num-envs", str(args.num_envs),
        "--total-steps", str(args.total_steps),
        "--n-steps", str(args.n_steps),
        "--batch-size", str(args.batch_size),
        "--arch", base_params['arch'],
        "--lr", str(base_params['lr']),
        "--n-epochs", str(base_params['n_epochs']),
        "--entropy-start", str(base_params['entropy_start']),
        "--entropy-end", str(base_params['entropy_end']),
        "--clip-range", str(base_params['clip_range']),
        "--gamma", str(base_params['gamma']),
        "--episode-steps", str(args.episode_steps),
        "--tb", str(tb_dir),
        "--results", str(results_path),
        "--seed", str(args.seed),
    ]
    if args.vecnorm:
        cmd_parts.append("--vecnorm")
    if args.curriculum:
        cmd_parts.extend([
            "--curriculum",
            "--curr-target-start", str(base_params['curr_target_start']),
            "--curr-promote", str(base_params['curr_promote']),
            "--curr-window", str(base_params['curr_window']),
            "--curr-bonus", str(base_params['curr_bonus']),
        ])

    cmd = " ".join([p for p in cmd_parts if p])
    print(f"[AUTO] Launching: {cmd}")

    env = os.environ.copy()
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    proc = subprocess.run(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout
    print(out)

    # Launch tensorboard (point at root so it can see all runs)
    launch_tensorboard(args.tensorboard_root, port=args.tb_port)

    # Read results JSON
    metrics = {}
    try:
        with open(results_path) as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"[AUTO] Could not read results: {e}")

    # Read TB scalars for this run
    tags = [
        'train/approx_kl','train/clip_fraction','train/explained_variance',
        'custom/avg_max_tile_window','custom/positive_reward_rate','custom/curr_success_rate','custom/ent_coef'
    ]
    tb_scalars = read_tb_scalars(tb_dir / "PPO_1", tags)
    if not tb_scalars:
        # Try other trial indices if present
        for trial in ("PPO_2", "PPO_3", "PPO_4"):
            tb_scalars = read_tb_scalars(tb_dir / trial, tags)
            if tb_scalars:
                break

    return metrics, tb_scalars, tb_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=3)
    ap.add_argument('--run-prefix', type=str, default='auto_run')
    ap.add_argument('--tensorboard-root', type=str, default='runs')
    ap.add_argument('--results-root', type=str, default='results')
    ap.add_argument('--tb-port', type=int, default=6006)
    # Base training controls
    ap.add_argument('--subproc', action='store_true')
    ap.add_argument('--num-envs', type=int, default=64)
    ap.add_argument('--n-steps', type=int, default=4096)
    ap.add_argument('--batch-size', type=int, default=131072)
    ap.add_argument('--arch', type=str, default='1024,1024')
    ap.add_argument('--total-steps', type=int, default=8_000_000)
    ap.add_argument('--episode-steps', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--vecnorm', action='store_true')
    ap.add_argument('--curriculum', action='store_true')
    args = ap.parse_args()

    base_params = {
        'arch': args.arch,
        'lr': 5e-5,
        'n_epochs': 20,
        'entropy_start': 0.03,
        'entropy_end': 0.01,
        'clip_range': 0.1,
        'gamma': 0.995,
        'curr_target_start': 128,
        'curr_promote': 0.5,
        'curr_window': 256,
        'curr_bonus': 2.0,
    }

    last_metrics = {}
    last_tb = {}
    for i in range(1, args.iterations + 1):
        print(f"\n[AUTO] === Iteration {i}/{args.iterations} ===")
        metrics, tb_scalars, run_dir = run_once(args, base_params, i)
        last_metrics, last_tb = metrics, tb_scalars
        # Print brief summary
        try:
            sr = metrics.get('success_rates', {})
            p95 = metrics.get('p95_max_tile')
            print(f"[AUTO] Results: p95={p95} | success@256={sr.get('256')} @512={sr.get('512')} @1024={sr.get('1024')}")
        except Exception:
            pass
        # Adjust params for next run
        base_params = decide_next_params(tb_scalars, base_params)
        print(f"[AUTO] Next params: lr={base_params['lr']}, epochs={base_params['n_epochs']}, ent=({base_params['entropy_start']}->{base_params['entropy_end']}), curr_target_start={base_params['curr_target_start']} curr_promote={base_params['curr_promote']} curr_window={base_params['curr_window']}")

    # Final dump for convenience
    print("\n[AUTO] Finished. Open TensorBoard at http://localhost:%d/" % args.tb_port)


if __name__ == '__main__':
    main()


