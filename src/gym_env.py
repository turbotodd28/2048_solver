import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.game import Game2048

class Game2048Env(gym.Env):
    """
    OpenAI Gymnasium-compatible 2048 environment.
    Observation: 4x4 board, log2 encoding, zeros for empty tiles.
    Action space: 0=up, 1=down, 2=left, 3=right
    Reward: score delta + nonlinear merge bonus (same as DQN agent)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        reward_variant='default',
        empty_tile_weight=0.1,
        monotonicity_weight=2.0,
        corner_bonus=10.0,
        merge_bonus_exp=1.5,
        milestone_1024=100.0,
        milestone_2048=500.0,
        milestone_4096=2000.0,
        move_penalty=0.1,
        lose_penalty=20.0
    ):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=16, shape=(16,), dtype=np.float32)
        self.game = Game2048()
        self.last_score = 0
        self.last_max_tile = 0
        self.reward_variant = reward_variant
        # Reward weights (can be swept)
        self.empty_tile_weight = empty_tile_weight
        self.monotonicity_weight = monotonicity_weight
        self.corner_bonus = corner_bonus
        self.merge_bonus_exp = merge_bonus_exp
        self.milestone_1024 = milestone_1024
        self.milestone_2048 = milestone_2048
        self.milestone_4096 = milestone_4096
        self.move_penalty = move_penalty
        self.lose_penalty = lose_penalty

    def set_reward_variant(self, variant):
        self.reward_variant = variant

    def set_reward_weights(self, **kwargs):
        # Dynamically update reward weights
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game2048()
        self.last_score = 0
        self.last_max_tile = np.max(self.game.board)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        move = ['up', 'down', 'left', 'right'][action]
        board_before = self.game.board.copy()
        score_before = self.game.score
        max_tile_before = np.max(board_before)

        try:
            self.game.move(move)
        except Exception:
            reward = -10.0
            done = True
            obs = self._get_obs()
            info = {'episode': {'r': reward, 'l': 1}, 'max_tile': int(np.max(self.game.board))}
            return obs, reward, done, False, info

        reward = self._compute_reward(board_before, score_before, max_tile_before)
        done = self.game.is_game_over()
        obs = self._get_obs()
        info = {'episode': {'r': reward, 'l': 1}, 'max_tile': int(np.max(self.game.board))}
        return obs, reward, done, False, info

    def _compute_reward(self, board_before, score_before, max_tile_before):
        score_after = self.game.score
        board_after = self.game.board
        max_tile_after = np.max(board_after)
        reward = score_after - score_before
        # Nonlinear merge bonus
        if max_tile_after > max_tile_before:
            reward += (max_tile_after ** self.merge_bonus_exp - max_tile_before ** self.merge_bonus_exp)
        # Bonus for new max tiles
        if max_tile_after >= 4096 and max_tile_before < 4096:
            reward += self.milestone_4096
        elif max_tile_after >= 2048 and max_tile_before < 2048:
            reward += self.milestone_2048
        elif max_tile_after >= 1024 and max_tile_before < 1024:
            reward += self.milestone_1024
        # Dynamic weights for empty tile and monotonicity rewards (can be swept or fixed)
        empty_tile_weight = self.empty_tile_weight
        monotonicity_weight = self.monotonicity_weight
        empty_tiles = np.count_nonzero(board_after == 0)
        reward += empty_tile_weight * empty_tiles
        # Cornerness bonus: reward if max tile is in a corner
        corners = [(0,0), (0,3), (3,0), (3,3)]
        if any(board_after[c] == max_tile_after for c in corners):
            reward += self.corner_bonus
        # Monotonicity bonus: reward if rows/cols are monotonic from a corner
        reward += monotonicity_weight * self._monotonicity_score(board_after)
        # Penalty per move
        reward -= self.move_penalty
        # Penalty for losing
        if self.game.is_game_over():
            reward -= self.lose_penalty
        return reward

    def _monotonicity_score(self, board):
        # Check monotonicity from each corner, take the best
        def mono_score(arr):
            return sum(arr[i] >= arr[i+1] for i in range(len(arr)-1))
        best = 0
        for row in [0, 3]:
            for col in [0, 3]:
                if row == 0:
                    row_seq = board[row, :] if col == 0 else board[row, ::-1]
                else:
                    row_seq = board[row, :] if col == 0 else board[row, ::-1]
                if col == 0:
                    col_seq = board[:, col] if row == 0 else board[::-1, col]
                else:
                    col_seq = board[:, col] if row == 0 else board[::-1, col]
                row_mono = mono_score(row_seq)
                col_mono = mono_score(col_seq)
                best = max(best, row_mono + col_mono)
        # Normalize: max possible is 6+6=12
        return best / 12.0

    def _get_obs(self):
        return np.where(self.game.board > 0, np.log2(self.game.board), 0).flatten().astype(np.float32)

    def render(self, mode="human"):
        print(self.game.board)

    def close(self):
        pass