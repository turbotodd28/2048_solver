import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.game import Game2048

class Game2048Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(17,), dtype=np.float32)
        self.game = Game2048()
        
        # Episode-level counters for diagnostics/analytics
        self.episode_moves = 0
        self.episode_invalid_moves = 0
        self.episode_valid_moves = 0


        # Reward weights (milestones disabled for now)
        self.reward_weights = {
            'empty_tile': 0.1,
            'corner_bonus': 10.0,
            'corner_penalty': -5.0,
            'invalid_penalty': -5.0,
            'merge_power': 1.5,
            'step_penalty': -1.0,
            'gameover_penalty': -100.0,
            'milestone_1024': 0,
            'milestone_2048': 0,
            'milestone_4096': 0,
        }

    def set_reward_weights(self, **kwargs):
        self.reward_weights.update(kwargs)

    def get_reward_structure_str(self):
        return ", ".join(f"{k}={v}" for k, v in self.reward_weights.items())
    


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game2048()
        # Reset episode counters
        self.episode_moves = 0
        self.episode_invalid_moves = 0
        self.episode_valid_moves = 0
        return self._get_obs(), {}

    def step(self, action):
        move = ['up', 'down', 'left', 'right'][action]
        
        # Capture state before move
        board_before = self.game.board.copy()
        score_before = self.game.score
        max_tile_before = np.max(board_before)
        
        # Try the move
        try:
            self.game.move(move)
            board_after = self.game.board
            score_after = self.game.score
            max_tile_after = np.max(board_after)
            empty_tiles_after = int(np.count_nonzero(board_after == 0))
            max_in_corner_after = bool(max_tile_after > 0 and board_after[3, 0] == max_tile_after)
            
            # Check if move was valid (board changed)
            if np.array_equal(board_before, board_after):
                # Invalid move - board didn't change
                reward = self.reward_weights['invalid_penalty']
                self.episode_moves += 1
                self.episode_invalid_moves += 1
                return self._get_obs(), reward, False, False, {
                    "invalid_move": True,
                    "episode_moves": self.episode_moves,
                    "episode_invalid_moves": self.episode_invalid_moves,
                    "episode_valid_moves": self.episode_valid_moves,
                    "empty_tiles": int(np.count_nonzero(board_before == 0)),
                    "max_in_corner": bool(np.max(board_before) > 0 and board_before[3, 0] == np.max(board_before)),
                    "action_mask": self.get_action_mask(),
                }
            
            # Valid move - board changed
            rw = self.reward_weights

            reward = score_after - score_before
            if max_tile_after > max_tile_before:
                reward += (max_tile_after ** rw['merge_power'] - max_tile_before ** rw['merge_power'])

            # (Temporarily disabled) milestone bonuses
            if max_tile_after >= 4096 and max_tile_before < 4096:
                reward += rw['milestone_4096']
            elif max_tile_after >= 2048 and max_tile_before < 2048:
                reward += rw['milestone_2048']
            elif max_tile_after >= 1024 and max_tile_before < 1024:
                reward += rw['milestone_1024']

            reward += rw['empty_tile'] * np.count_nonzero(board_after == 0)

            # Corner penalty/bonus logic (only penalize if moved out)
            was_in_corner = (max_tile_before > 0 and board_before[3, 0] == max_tile_before)
            now_in_corner = (max_tile_after > 0 and board_after[3, 0] == max_tile_after)
            if was_in_corner and not now_in_corner:
                reward += rw['corner_penalty']
            if now_in_corner:
                reward += rw['corner_bonus']

            reward += rw['step_penalty']

            # Book-keeping
            self.episode_moves += 1
            self.episode_valid_moves += 1

            done = self.game.is_game_over()
            if done:
                reward += rw['gameover_penalty']

            info = {
                "score": self.game.score,
                "max_tile": int(max_tile_after),
                "invalid_move": False,
                "episode_moves": self.episode_moves,
                "episode_invalid_moves": self.episode_invalid_moves,
                "episode_valid_moves": self.episode_valid_moves,
                "empty_tiles": empty_tiles_after,
                "max_in_corner": max_in_corner_after,
            }

            # When episode ends, include summary counters (same keys already present)
            # Also include action mask for mask-aware algorithms
            info["action_mask"] = self.get_action_mask()
            return self._get_obs(), reward, done, False, info
            
        except ValueError:
            # Invalid move - game threw exception
            reward = self.reward_weights['invalid_penalty']
            self.episode_moves += 1
            self.episode_invalid_moves += 1
            return self._get_obs(), reward, False, False, {
                "invalid_move": True,
                "episode_moves": self.episode_moves,
                "episode_invalid_moves": self.episode_invalid_moves,
                "episode_valid_moves": self.episode_valid_moves,
                "empty_tiles": int(np.count_nonzero(self.game.board == 0)),
                "max_in_corner": bool(np.max(self.game.board) > 0 and self.game.board[3, 0] == np.max(self.game.board)),
                "action_mask": self.get_action_mask(),
            }

    def _get_obs(self):
        with np.errstate(divide='ignore'):
            obs = np.where(self.game.board > 0, np.log2(self.game.board) / 11, 0).flatten().astype(np.float32)
        max_tile = np.max(self.game.board)
        max_in_corner = 1.0 if (max_tile > 0 and self.game.board[3, 0] == max_tile) else 0.0
        return np.concatenate([obs, [max_in_corner]]).astype(np.float32)

    def get_action_mask(self):
        # Boolean mask [up, down, left, right]
        mask_bool = self.game.get_valid_action_mask()
        # Convert to float mask in {0.0, 1.0} for sb3-contrib conventions
        return mask_bool.astype(np.float32)

    def render(self, mode="human"):
        print(self.game.board)

    def close(self):
        pass
