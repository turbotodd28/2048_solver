import numpy as np
import math
from typing import Tuple, List, Dict
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import time

from src.game import Game2048

# ===== CONFIGURABLE WEIGHTS =====
# Tune these values to optimize performance!

DEFAULT_WEIGHTS = {
    'empty_spaces': 2.0,      # How much to value empty spaces (higher = more conservative)
    'corner_bonus': 8.0,      # Bonus for having max tile in corner
    'snake_pattern': 2.0,     # Bonus for snake pattern (decreasing values)
    'monotonicity': 0.5,      # Bonus for monotonic rows/columns
    'smoothness': 0.1,        # Penalty for adjacent tile differences
    'merge_potential': 0.1,   # Bonus for immediate merge opportunities
    'max_tile_bonus': 0.0,    # Bonus for having high tiles (0 = disabled)
    'edge_bonus': 0.0,        # Bonus for tiles on edges (0 = disabled)
}

def get_valid_action_mask_for_board(board: np.ndarray) -> np.ndarray:
    """Convert board to valid action mask [up, down, left, right]"""
    g = Game2048()
    g.board = board.copy()
    valid_moves = g.get_valid_moves()
    mask = np.zeros(4, dtype=bool)
    directions = ['up', 'down', 'left', 'right']
    for i, direction in enumerate(directions):
        if direction in valid_moves:
            mask[i] = True
    return mask


def preview_after_move(board: np.ndarray, action_idx: int) -> np.ndarray:
    """Preview what the board would look like after a move"""
    directions = ['up', 'down', 'left', 'right']
    g = Game2048()
    g.board = board.copy()
    g.move(directions[action_idx])
    return g.board.copy()


def evaluate_board_tunable(board: np.ndarray, weights: Dict[str, float] = None) -> float:
    """Tunable evaluation function with configurable weights"""
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    # Basic metrics
    empties = np.count_nonzero(board == 0)
    max_tile = np.max(board)
    
    # Strategic positioning
    corner_bonus = 0.0
    if max_tile > 0:
        corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
        if max_tile in corners:
            corner_bonus = weights['corner_bonus']
    
    # Snake pattern bonus (classic 2048 strategy)
    snake_bonus = 0.0
    snake_pattern = [
        board[0, 0], board[0, 1], board[0, 2], board[0, 3],
        board[1, 3], board[1, 2], board[1, 1], board[1, 0],
        board[2, 0], board[2, 1], board[2, 2], board[2, 3],
        board[3, 3], board[3, 2], board[3, 1], board[3, 0]
    ]
    
    # Check if high tiles follow snake pattern
    non_zero = [x for x in snake_pattern if x > 0]
    if len(non_zero) >= 2:
        # Calculate how well tiles follow decreasing pattern
        for i in range(len(non_zero) - 1):
            if non_zero[i] >= non_zero[i + 1]:
                snake_bonus += weights['snake_pattern']
    
    # Monotonicity (improved)
    def mono_score(arr: np.ndarray) -> float:
        s = 0.0
        # Row monotonicity
        for r in range(4):
            row = arr[r, :]
            for i in range(3):
                if row[i] >= row[i + 1] and row[i] > 0:
                    s += 1.0
        # Column monotonicity
        for c in range(4):
            col = arr[:, c]
            for i in range(3):
                if col[i] >= col[i + 1] and col[i] > 0:
                    s += 1.0
        return s * weights['monotonicity']
    
    # Smoothness (improved)
    def smooth_score(arr: np.ndarray) -> float:
        smooth = 0.0
        
        # Horizontal smoothness
        for r in range(4):
            for c in range(3):
                if arr[r, c] > 0 and arr[r, c + 1] > 0:
                    diff = abs(np.log2(arr[r, c]) - np.log2(arr[r, c + 1]))
                    smooth -= diff * weights['smoothness']
        
        # Vertical smoothness
        for c in range(4):
            for r in range(3):
                if arr[r, c] > 0 and arr[r + 1, c] > 0:
                    diff = abs(np.log2(arr[r, c]) - np.log2(arr[r + 1, c]))
                    smooth -= diff * weights['smoothness']
        
        return smooth
    
    # Merge potential bonus
    merge_potential = 0.0
    for r in range(4):
        for c in range(3):
            if board[r, c] > 0 and board[r, c] == board[r, c + 1]:
                merge_potential += board[r, c] * weights['merge_potential']
    for c in range(4):
        for r in range(3):
            if board[r, c] > 0 and board[r, c] == board[r + 1, c]:
                merge_potential += board[r, c] * weights['merge_potential']
    
    # Max tile bonus (optional)
    max_tile_bonus = 0.0
    if weights['max_tile_bonus'] > 0:
        max_tile_bonus = max_tile * weights['max_tile_bonus']
    
    # Edge bonus (optional)
    edge_bonus = 0.0
    if weights['edge_bonus'] > 0:
        edge_tiles = 0
        for r in range(4):
            for c in range(4):
                if board[r, c] > 0 and (r == 0 or r == 3 or c == 0 or c == 3):
                    edge_tiles += board[r, c]
        edge_bonus = edge_tiles * weights['edge_bonus']
    
    # Weighted combination
    score = (
        empties * weights['empty_spaces'] +     # Empty spaces
        corner_bonus +                          # Corner positioning
        snake_bonus +                           # Snake pattern
        mono_score(board) +                     # Monotonicity
        smooth_score(board) +                   # Smoothness
        merge_potential +                       # Immediate merge opportunities
        max_tile_bonus +                        # Max tile bonus
        edge_bonus                              # Edge bonus
    )
    
    return score


def expectimax_best_action_tunable(board: np.ndarray, depth: int = 4, chance_sample_k: int = 8, 
                                  weights: Dict[str, float] = None) -> int:
    """Tunable expectimax with configurable evaluation weights"""
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    # Limit depth to reasonable values
    depth = min(depth, 6)  # Cap at depth 6 for practical performance
    
    mask = get_valid_action_mask_for_board(board)
    valid_actions = [i for i in range(4) if mask[i]]
    if not valid_actions:
        return 0

    best_action = valid_actions[0]
    best_val = -1e9
    
    for a in valid_actions:
        nb = preview_after_move(board, a)
        if np.array_equal(nb, board):
            continue
        
        val = chance_value_tunable(nb, depth - 1, chance_sample_k, weights)
        if val > best_val:
            best_val = val
            best_action = a
    
    return best_action


def chance_value_tunable(b: np.ndarray, d: int, chance_sample_k: int, weights: Dict[str, float]) -> float:
    """Tunable chance value calculation"""
    if d == 0:
        return evaluate_board_tunable(b, weights)
    
    empties = np.argwhere(b == 0)
    if empties.size == 0:
        return max_value_tunable(b, d, chance_sample_k, weights)
    
    # Sample more scenarios for better accuracy
    if len(empties) > chance_sample_k:
        idx = np.random.choice(len(empties), size=chance_sample_k, replace=False)
        empties = empties[idx]
    
    total = 0.0
    total_p = 0.0
    
    for (r, c) in empties:
        for tile, p in [(2, 0.9), (4, 0.1)]:
            nb = b.copy()
            nb[r, c] = tile
            total += p * max_value_tunable(nb, d, chance_sample_k, weights)
            total_p += p
    
    return total / max(total_p, 1e-8)


def max_value_tunable(b: np.ndarray, d: int, chance_sample_k: int, weights: Dict[str, float]) -> float:
    """Tunable max value calculation"""
    if d == 0:
        return evaluate_board_tunable(b, weights)
    
    m = get_valid_action_mask_for_board(b)
    if not np.any(m):
        return evaluate_board_tunable(b, weights)
    
    best = -1e9
    for a in range(4):
        if not m[a]:
            continue
        nb = preview_after_move(b, a)
        if np.array_equal(nb, b):
            continue
        val = chance_value_tunable(nb, d - 1, chance_sample_k, weights)
        if val > best:
            best = val
    
    return best


# Predefined weight configurations for different strategies
WEIGHT_PRESETS = {
    'conservative': {
        'empty_spaces': 3.0,      # Very high - prioritize keeping spaces open
        'corner_bonus': 10.0,     # High - always keep max tile in corner
        'snake_pattern': 3.0,     # High - maintain snake pattern
        'monotonicity': 1.0,      # High - keep rows/cols ordered
        'smoothness': 0.2,        # High - minimize adjacent differences
        'merge_potential': 0.05,  # Low - don't rush merges
        'max_tile_bonus': 0.0,    # None
        'edge_bonus': 0.0,        # None
    },
    'aggressive': {
        'empty_spaces': 1.5,      # Lower - willing to fill spaces
        'corner_bonus': 6.0,      # Medium - corner is good but not critical
        'snake_pattern': 1.0,     # Lower - flexible pattern
        'monotonicity': 0.3,      # Lower - less strict ordering
        'smoothness': 0.05,       # Lower - tolerate differences
        'merge_potential': 0.2,   # High - actively seek merges
        'max_tile_bonus': 0.01,   # Small bonus for high tiles
        'edge_bonus': 0.0,        # None
    },
    'balanced': {
        'empty_spaces': 2.0,      # Medium - balanced approach
        'corner_bonus': 8.0,      # High - corner positioning
        'snake_pattern': 2.0,     # Medium - maintain pattern
        'monotonicity': 0.5,      # Medium - some ordering
        'smoothness': 0.1,        # Medium - moderate smoothness
        'merge_potential': 0.1,   # Medium - opportunistic merges
        'max_tile_bonus': 0.0,    # None
        'edge_bonus': 0.0,        # None
    },
    'experimental': {
        'empty_spaces': 2.5,      # High - keep spaces open
        'corner_bonus': 12.0,     # Very high - corner critical
        'snake_pattern': 4.0,     # Very high - strict pattern
        'monotonicity': 1.5,      # Very high - strict ordering
        'smoothness': 0.3,        # Very high - very smooth
        'merge_potential': 0.05,  # Low - patient merging
        'max_tile_bonus': 0.005,  # Tiny bonus
        'edge_bonus': 0.001,      # Tiny edge bonus
    }
}


# Keep the original function for compatibility
def expectimax_best_action(board: np.ndarray, depth: int = 3, chance_sample_k: int = 6) -> int:
    """Original single-threaded version"""
    return expectimax_best_action_tunable(board, depth, chance_sample_k, DEFAULT_WEIGHTS)
