import random
import numpy as np
import curses
from typing import List, Optional, Tuple

class Game2048:
    """
    Clean, fast 2048 game implementation focused on core mechanics.
    No animation logic - designed for AI training and headless operation.
    """
    
    def __init__(self, size: int = 4):
        self.n = size
        self.board = np.zeros((self.n, self.n), dtype=int)
        self.score = 0
        self.move_count = 0
        # Start with two tiles as usual
        self._add_random_tile()
        self._add_random_tile()

    def _add_random_tile(self) -> Optional[Tuple[int, int, int]]:
        """Add a random tile and return (row, col, value) or None if no space."""
        empty = [(i, j) for i in range(self.n) for j in range(self.n) if self.board[i, j] == 0]
        if not empty:
            return None
        i, j = random.choice(empty)
        val = 2 if random.random() < 0.9 else 4
        self.board[i, j] = val
        return (i, j, val)

    def _transform(self, arr: np.ndarray, direction: str) -> Tuple[np.ndarray, bool]:
        """Return a transformed view for computing 'left' logic."""
        rotated = False
        out = arr
        if direction in ['up', 'down']:
            out = out.T
            rotated = True
        if direction in ['down', 'right']:
            out = np.flip(out, axis=1)
        return out, rotated

    def _inverse_transform(self, arr: np.ndarray, direction: str, rotated: bool) -> np.ndarray:
        out = arr
        if direction in ['down', 'right']:
            out = np.flip(out, axis=1)
        if rotated:
            out = out.T
        return out

    def _slide_and_merge_row(self, row: np.ndarray) -> Tuple[np.ndarray, int]:
        """Slide and merge a single row, return (new_row, score_gain)."""
        non_zero = row[row != 0]
        out = []
        score_gain = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_val = non_zero[i] * 2
                out.append(merged_val)
                score_gain += merged_val
                i += 2
            else:
                out.append(non_zero[i])
                i += 1
        # Pad with zeros
        while len(out) < len(row):
            out.append(0)
        return np.array(out, dtype=int), score_gain

    def move(self, direction: str) -> bool:
        """
        Perform a move in the given direction.
        Returns True if the move was valid and changed the board, False otherwise.
        """
        if direction not in ['up', 'down', 'left', 'right']:
            return False

        # Transform board for left-merge computation
        vboard, rotated = self._transform(self.board.copy(), direction)
        
        # Track if any changes occurred
        changed = False
        total_score_gain = 0

        # Process each row
        for r in range(self.n):
            old_row = vboard[r].copy()
            new_row, score_gain = self._slide_and_merge_row(old_row)
            vboard[r] = new_row
            total_score_gain += score_gain
            if not np.array_equal(old_row, new_row):
                changed = True

        if not changed:
            return False

        # Transform back to original orientation
        self.board = self._inverse_transform(vboard, direction, rotated)
        self.score += total_score_gain
        self.move_count += 1

        # Add new tile
        self._add_random_tile()
        return True

    def get_valid_moves(self) -> List[str]:
        """Return list of valid move directions."""
        valid_moves = []
        for direction in ['up', 'down', 'left', 'right']:
            if self._is_move_possible(direction):
                valid_moves.append(direction)
        return valid_moves

    def _is_move_possible(self, direction: str) -> bool:
        """Check if a move in the given direction is possible."""
        if direction not in ['up', 'down', 'left', 'right']:
            return False

        # Transform board
        vboard, rotated = self._transform(self.board.copy(), direction)
        
        # Check each row for possible moves
        for r in range(self.n):
            row = vboard[r]
            # Check for slides (zeros before non-zeros)
            non_zero_indices = np.where(row != 0)[0]
            if len(non_zero_indices) > 0:
                # Check if there are zeros before the first non-zero
                if non_zero_indices[0] > 0:
                    return True
                # Check for merges (adjacent equal values)
                for i in range(len(non_zero_indices) - 1):
                    if row[non_zero_indices[i]] == row[non_zero_indices[i + 1]]:
                        return True
        return False

    def is_game_over(self) -> bool:
        """Check if the game is over (no valid moves possible)."""
        return len(self.get_valid_moves()) == 0

    def get_state(self) -> np.ndarray:
        """Get current board state as numpy array."""
        return self.board.copy()

    def get_score(self) -> int:
        """Get current score."""
        return self.score

    def get_move_count(self) -> int:
        """Get number of moves made."""
        return self.move_count

    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((self.n, self.n), dtype=int)
        self.score = 0
        self.move_count = 0
        self._add_random_tile()
        self._add_random_tile()

    # ---------- Simple curses render (for debugging) ----------

    def render(self, stdscr, previous_board=None):
        stdscr.clear()
        stdscr.addstr(0, 0, f"Score: {self.score}    Moves: {self.move_count}\n")

        if previous_board is not None:
            stdscr.addstr(1, 0, "Previous Board:")
            for i, row in enumerate(previous_board):
                row_str = "|".join(f"{num:4}" if num > 0 else "    " for num in row)
                stdscr.addstr(i * 2 + 2, 0, f"|{row_str}|")
                if i < self.n - 1:
                    stdscr.addstr(i * 2 + 3, 0, "+----" * self.n + "+")

        stdscr.addstr(1, 25, "Current Board:")
        for i, row in enumerate(self.board):
            row_str = "|".join(f"{num:4}" if num > 0 else "    " for num in row)
            stdscr.addstr(i * 2 + 2, 25, f"|{row_str}|")
            if i < self.n - 1:
                stdscr.addstr(i * 2 + 3, 25, "+----" * self.n + "+")

        stdscr.refresh()


# ---------- CLI runner (for testing) ----------

def main(stdscr):
    curses.curs_set(0)  # Hide the cursor
    stdscr.clear()

    game = Game2048()
    key_mapping = {
        curses.KEY_UP: 'up',
        curses.KEY_DOWN: 'down',
        curses.KEY_LEFT: 'left',
        curses.KEY_RIGHT: 'right',
        ord('w'): 'up',
        ord('s'): 'down',
        ord('a'): 'left',
        ord('d'): 'right'
    }

    game.render(stdscr)
    previous_board = None
    while not game.is_game_over():
        previous_board = game.board.copy()
        key = stdscr.getch()
        if key in key_mapping:
            if game.move(key_mapping[key]):
                game.render(stdscr, previous_board=previous_board)
        else:
            stdscr.addstr(0, 0, "Invalid input. Use arrow keys or w, a, s, d.")

    # Show final state using existing render method with previous state
    stdscr.clear()
    stdscr.addstr(0, 0, "Game Over! Final State:")
    game.render(stdscr, previous_board=previous_board)
    stdscr.addstr(game.n * 2 + 3, 0, "\nPress any key to exit...")
    stdscr.refresh()
    stdscr.getch()  # Wait for user input before exiting

if __name__ == "__main__":
    curses.wrapper(main)
