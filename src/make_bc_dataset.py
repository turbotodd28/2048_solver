import os
import json
import numpy as np
from tqdm import trange

from src.game import Game2048
from src.expectimax import expectimax_best_action


def collect_examples(num_games: int = 2000, max_steps: int = 2000, depth: int = 3) -> dict:
    data = {"obs": [], "action": []}
    for g in trange(num_games):
        env = Game2048()
        steps = 0
        while steps < max_steps and not env.is_game_over():
            with np.errstate(divide='ignore'):
                obs = np.where(env.board > 0, np.log2(env.board) / 11, 0).astype(np.float32).flatten()
            a = expectimax_best_action(env.board, depth=depth)
            # try the move; if invalid, break
            try:
                env.move(['up','down','left','right'][a])
            except ValueError:
                break
            data["obs"].append(obs.tolist())
            data["action"].append(int(a))
            steps += 1
    return data


def main():
    out = os.environ.get('BC_DATA', 'bc_dataset.json')
    depth = int(os.environ.get('BC_DEPTH', '3'))
    games = int(os.environ.get('BC_GAMES', '2000'))
    data = collect_examples(num_games=games, depth=depth)
    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, out)
    print(f"Saved {len(data['action'])} examples to {out}")


if __name__ == '__main__':
    main()


