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
        self.observation_space = spaces.Box(low=0, high=16, shape=(17,), dtype=np.float32)
        self.game = Game2048()
        self.last_score = 0
        self.last_max_tile = 0
        self.reward_variant = reward_variant
        # Reward weights (all in a dictionary for transparency)
        self.reward_weights = {
            'max_in_corner': 5.0,      # Reward for max in corner
            'not_in_corner': -2.0,     # Penalty for not in corner
            'moved_out_of_corner': -30.0, # Strong penalty for moving max out of corner
            'max_tile_increased': 15.0,       # Bonus for increasing max tile
            'combine_top_two': 10.0    # Bonus for combining the two highest tiles
        }
        # ...existing code for other weights...
        self.empty_tile_weight = empty_tile_weight
        self.monotonicity_weight = monotonicity_weight
        self.corner_bonus = corner_bonus
        self.merge_bonus_exp = merge_bonus_exp
        self.milestone_1024 = milestone_1024
        self.milestone_2048 = milestone_2048
        self.milestone_4096 = milestone_4096
        self.move_penalty = move_penalty
        self.lose_penalty = lose_penalty
        # --- Metrics tracking ---
        self.episode_metrics = None

    def set_reward_variant(self, variant):
        self.reward_variant = variant

    def set_reward_weights(self, **kwargs):
        # Dynamically update reward weights
        for k, v in kwargs.items():
            if k in self.reward_weights:
                self.reward_weights[k] = v
            elif hasattr(self, k):
                setattr(self, k, v)

    def _compute_reward(self, board_before, score_before, max_tile_before):
        board_after = self.game.board
        max_tile_after = np.max(board_after)
        bottom_left = (3, 0)
        max_in_corner = board_after[bottom_left] == max_tile_after
        reward = 0.0
        breakdown = {}
        # Max in corner
        if max_in_corner:
            reward += self.reward_weights['max_in_corner']
            breakdown['max_in_corner'] = self.reward_weights['max_in_corner']
        else:
            reward += self.reward_weights['not_in_corner']
            breakdown['not_in_corner'] = self.reward_weights['not_in_corner']
        # Only penalize if the max tile was in the bottom-left before, and is not anymore (and the value didn't increase)
        max_in_corner_before = board_before[bottom_left] == max_tile_before
        if max_in_corner_before and not max_in_corner and max_tile_after == max_tile_before:
            reward += self.reward_weights['moved_out_of_corner']
            breakdown['moved_out_of_corner'] = self.reward_weights['moved_out_of_corner']
        # Max tile increased
        if max_tile_after > max_tile_before:
            reward += self.reward_weights['max_tile_increased']
            breakdown['max_tile_increased'] = self.reward_weights['max_tile_increased']
        # Monotonicity reward: only for leftmost column, moving upward, if max is in bottom-left
        if max_in_corner:
            left_col = board_after[:, 0]
            mono_score = sum(left_col[i] >= left_col[i+1] for i in range(3)) / 3.0
            mono_reward = self.monotonicity_weight * mono_score
            reward += mono_reward
            breakdown['monotonicity'] = mono_reward
        # Reward for combining the two highest tiles possible
        before_flat = board_before.flatten()
        after_flat = board_after.flatten()
        before_sorted = np.sort(before_flat)[::-1]
        after_sorted = np.sort(after_flat)[::-1]
        if len(before_sorted) > 1 and len(after_sorted) > 1:
            before_second = before_sorted[1]
            after_second = after_sorted[1]
            if max_tile_after > max_tile_before and after_second < before_second:
                reward += self.reward_weights['combine_top_two']
                breakdown['combine_top_two'] = self.reward_weights['combine_top_two']
        return reward, breakdown

    def _monotonicity_score(self, arrs):
        # Accepts a list/array of 1D arrays (row and col), returns average normalized monotonicity
        def mono_score(arr):
            return sum(arr[i] >= arr[i+1] for i in range(len(arr)-1)) / (len(arr)-1)
        if isinstance(arrs, np.ndarray) and arrs.ndim == 2:
            scores = [mono_score(arr) for arr in arrs]
            return np.mean(scores)
        else:
            return mono_score(arrs)

    def get_reward_structure_str(self):
        rw = self.reward_weights
        return (
            f"Reward: +{rw['max_in_corner']} if max in bottom-left, "
            f"{rw['not_in_corner']} otherwise, "
            f"{rw['moved_out_of_corner']} if moved out, "
            f"+{rw['max_tile_increased']} max tile increased, "
            f"+{self.monotonicity_weight}*monotonicity if max in bottom-left"
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game2048()
        self.last_score = 0
        self.last_max_tile = np.max(self.game.board)
        obs = self._get_obs()
        # --- Initialize episode metrics ---
        self.episode_metrics = {
            'step_count': 0,
            'action_counts': np.zeros(4, dtype=int),
            'max_tile_history': [],
            'corner_occupancy': 0,
            'monotonicity_history': [],
            'empty_tiles_history': [],
            'score_history': [],
            'merge_events': 0,
            'top_tile_merges': 0,
            'reward_breakdown_history': [],
            'final_score': 0,
            'end_reason': None
        }
        return obs, {}

    def valid_moves(self):
        # Returns a boolean mask of valid moves [up, down, left, right]
        moves = ['up', 'down', 'left', 'right']
        valid = []
        for move in moves:
            try:
                test_game = Game2048()
                test_game.board = self.game.board.copy()
                test_game.score = self.game.score
                test_game.move(move)
                if not np.array_equal(test_game.board, self.game.board):
                    valid.append(True)
                else:
                    valid.append(False)
            except Exception:
                valid.append(False)
        return np.array(valid, dtype=bool)

    def step(self, action):
        valid_mask = self.valid_moves()
        if not valid_mask[action]:
            # Explicit penalty and termination for invalid move
            reward = -20.0  # Clearly negative penalty
            done = True
            obs = self._get_obs()
            self.episode_metrics['final_score'] = int(self.game.score)
            self.episode_metrics['end_reason'] = 'invalid_move'
            info = {
                'episode': {'r': reward, 'l': 1},
                'max_tile': int(np.max(self.game.board)),
                'end_reason': 'invalid_move',
                'valid_moves': valid_mask,
                'metrics': self.episode_metrics.copy()
            }
            return obs, reward, done, False, info

        move = ['up', 'down', 'left', 'right'][action]
        board_before = self.game.board.copy()
        score_before = self.game.score
        max_tile_before = np.max(board_before)

        before_flat = board_before.flatten()
        self.game.move(move)
        after_flat = self.game.board.flatten()

        merges = np.sum((after_flat > before_flat) & (after_flat > 2))
        self.episode_metrics['merge_events'] += merges

        max_tile_after = np.max(self.game.board)
        if max_tile_after > max_tile_before and np.count_nonzero(after_flat == max_tile_after) == 1:
            self.episode_metrics['top_tile_merges'] += 1

        reward, breakdown = self._compute_reward(board_before, score_before, max_tile_before)

        self.episode_metrics['step_count'] += 1
        self.episode_metrics['action_counts'][action] += 1
        self.episode_metrics['max_tile_history'].append(int(max_tile_after))

        bottom_left = (3, 0)
        if self.game.board[bottom_left] == max_tile_after:
            self.episode_metrics['corner_occupancy'] += 1

        left_col = self.game.board[:, 0]
        mono_score = sum(left_col[i] >= left_col[i+1] for i in range(3)) / 3.0
        self.episode_metrics['monotonicity_history'].append(mono_score)

        empty_tiles = np.count_nonzero(self.game.board == 0)
        self.episode_metrics['empty_tiles_history'].append(empty_tiles)

        self.episode_metrics['score_history'].append(int(self.game.score))
        self.episode_metrics['reward_breakdown_history'].append(breakdown)

        done = self.game.is_game_over() or (np.max(self.game.board) >= 2**20)
        obs = self._get_obs()

        end_reason = ('game_over' if self.game.is_game_over()
                    else 'max_tile_limit' if np.max(self.game.board) >= 2**20
                    else 'not_done')

        if done:
            self.episode_metrics['final_score'] = int(self.game.score)
            self.episode_metrics['end_reason'] = end_reason

        info = {
            'episode': {'r': reward, 'l': 1},
            'max_tile': int(np.max(self.game.board)),
            'end_reason': end_reason,
            'valid_moves': valid_mask,
            'metrics': self.episode_metrics.copy() if done else None
        }
        return obs, reward, done, False, info

    def _get_obs(self):
        obs = np.where(self.game.board > 0, np.log2(self.game.board), 0).flatten().astype(np.float32)
        # Add binary feature: 1 if max tile is in bottom-left, 0 otherwise
        board = self.game.board
        max_tile = np.max(board)
        bottom_left = (3, 0)
        max_in_corner = 1.0 if board[bottom_left] == max_tile else 0.0
        obs = np.concatenate([obs, [max_in_corner]]).astype(np.float32)
        return obs

    def render(self, mode="human"):
        print(self.game.board)

    def close(self):
        pass