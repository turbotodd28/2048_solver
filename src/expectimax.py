import numpy as np
import math
from typing import Tuple, List

from src.game import Game2048


def get_valid_action_mask_for_board(board: np.ndarray) -> np.ndarray:
    g = Game2048()
    g.board = board.copy()
    return g.get_valid_action_mask()


def preview_after_move(board: np.ndarray, action_idx: int) -> np.ndarray:
    directions = ['up', 'down', 'left', 'right']
    g = Game2048()
    g.board = board.copy()
    # Use internal preview helper added earlier
    return g._preview_board_after_move(directions[action_idx])


def evaluate_board(board: np.ndarray) -> float:
    # Heuristic: empties, max tile in corner, monotonicity (rough), smoothness (rough)
    empties = np.count_nonzero(board == 0)
    max_tile = np.max(board)
    corner_bonus = 0.0
    if max_tile > 0 and (board[3, 0] == max_tile or board[3, 3] == max_tile or board[0, 0] == max_tile or board[0, 3] == max_tile):
        corner_bonus = 4.0

    # Monotonicity (rough): sum of signed diffs along rows and cols
    def mono_score(arr: np.ndarray) -> float:
        s = 0.0
        for r in range(4):
            s += float(np.sum(np.diff(arr[r, :])))
        for c in range(4):
            s += float(np.sum(np.diff(arr[:, c])))
        return abs(s) * 0.01

    # Smoothness (rough): penalize large adjacent differences in log2 space
    # Use exponent representation if values are powers of two; else fall back to log2
    try:
        # Fast path: if board entries are powers of two or zero
        logb = np.zeros_like(board, dtype=np.float32)
        nonzero = board > 0
        # np.log2 may be fast enough, but try integer log2 if possible
        logb[nonzero] = np.log2(board[nonzero])
    except Exception:
        with np.errstate(divide='ignore'):
            logb = np.where(board > 0, np.log2(board), 0.0)
    smooth = 0.0
    for r in range(4):
        for c in range(3):
            smooth -= abs(logb[r, c] - logb[r, c + 1]) * 0.05
    for c in range(4):
        for r in range(3):
            smooth -= abs(logb[r, c] - logb[r + 1, c]) * 0.05

    return empties * 1.0 + corner_bonus + mono_score(board) + smooth


def expectimax_best_action(board: np.ndarray, depth: int = 3, chance_sample_k: int = 6) -> int:
    mask = get_valid_action_mask_for_board(board)
    valid_actions = [i for i in range(4) if mask[i]]
    if not valid_actions:
        return 0

    def max_value(b: np.ndarray, d: int) -> float:
        if d == 0:
            return evaluate_board(b)
        m = get_valid_action_mask_for_board(b)
        if not np.any(m):
            return evaluate_board(b)
        best = -1e9
        for a in range(4):
            if not m[a]:
                continue
            nb = preview_after_move(b, a)
            if np.array_equal(nb, b):
                continue
            val = chance_value(nb, d - 1)
            if val > best:
                best = val
        return best

    def chance_value(b: np.ndarray, d: int) -> float:
        empties = np.argwhere(b == 0)
        if empties.size == 0:
            return max_value(b, d)
        # Sample at most K empty positions for speed
        if len(empties) > chance_sample_k:
            idx = np.random.choice(len(empties), size=chance_sample_k, replace=False)
            empties = empties[idx]
        total = 0.0
        total_p = 0.0
        for (r, c) in empties:
            for tile, p in [(2, 0.9), (4, 0.1)]:
                nb = b.copy()
                nb[r, c] = tile
                total += p * max_value(nb, d)
                total_p += p
        return total / max(total_p, 1e-8)

    best_action = valid_actions[0]
    best_val = -1e9
    for a in valid_actions:
        nb = preview_after_move(board, a)
        if np.array_equal(nb, board):
            continue
        val = chance_value(nb, depth - 1)
        if val > best_val:
            best_val = val
            best_action = a
    return best_action


