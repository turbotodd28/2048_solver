import random
import numpy as np
import curses

class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.move_count = 0
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.board[i, j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i, j] = 2 if random.random() < 0.9 else 4

    def slide_and_merge(self, row):
        # Slide non-zero elements to the left
        non_zero = row[row != 0]
        new_row = []
        skip = False
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i < len(non_zero) - 1 and non_zero[i] == non_zero[i + 1]:
                new_row.append(non_zero[i] * 2)
                self.score += non_zero[i] * 2
                skip = True
            else:
                new_row.append(non_zero[i])
        # Fill the rest with zeros to ensure proper clearing
        while len(new_row) < len(row):
            new_row.append(0)
        return np.array(new_row)

    def move(self, direction):
        if direction not in ['up', 'down', 'left', 'right']:
            raise ValueError("Invalid move direction")

        original_board = self.board.copy()
        rotated = False
        
        # Strategy: Convert all moves to leftward operations by rotating/flipping the board
        # This allows us to use a single slide_and_merge function for all directions
        
        # For up/down moves: transpose the board so columns become rows
        if direction in ['up', 'down']:
            self.board = self.board.T
            rotated = True

        # For down/right moves: flip horizontally so we can slide leftward
        # This converts right->left and down->up (after transpose)
        if direction in ['down', 'right']:
            self.board = np.flip(self.board, axis=1)

        # Apply leftward slide and merge to all rows
        for i in range(4):
            self.board[i] = self.slide_and_merge(self.board[i])

        # Reverse the horizontal flip if it was applied
        if direction in ['down', 'right']:
            self.board = np.flip(self.board, axis=1)

        # Reverse the transpose if it was applied
        if rotated:
            self.board = self.board.T

        if not np.array_equal(self.board, original_board):
            self.add_random_tile()
            self.move_count += 1  # Increment move count
        else:
            raise ValueError("Invalid move: No tiles moved or combined.")

    def is_game_over(self):
        # Check if any cell is empty
        if np.any(self.board == 0):
            return False
        
        # Check if any move is possible by looking for adjacent equal tiles
        # Optimized: check each position only once for both horizontal and vertical matches
        for i in range(4):
            for j in range(4):
                current = self.board[i, j]
                # Check right neighbor (horizontal match)
                if j < 3 and current == self.board[i, j + 1]:
                    return False
                # Check bottom neighbor (vertical match)
                if i < 3 and current == self.board[i + 1, j]:
                    return False
        return True

    def render(self, stdscr, previous_board=None):
        stdscr.clear()
        stdscr.addstr(0, 0, f"Score: {self.score}    Moves: {self.move_count}\n")

        if previous_board is not None:
            stdscr.addstr(1, 0, "Previous Board:")
            for i, row in enumerate(previous_board):
                row_str = "|".join(f"{num:4}" if num > 0 else "    " for num in row)
                stdscr.addstr(i * 2 + 2, 0, f"|{row_str}|")
                if i < 3:
                    stdscr.addstr(i * 2 + 3, 0, "+----+----+----+----+")

        stdscr.addstr(1, 25, "Current Board:")
        for i, row in enumerate(self.board):
            row_str = "|".join(f"{num:4}" if num > 0 else "    " for num in row)
            stdscr.addstr(i * 2 + 2, 25, f"|{row_str}|")
            if i < 3:
                stdscr.addstr(i * 2 + 3, 25, "+----+----+----+----+")

        stdscr.refresh()

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
    while not game.is_game_over():
        previous_board = game.board.copy()
        key = stdscr.getch()
        if key in key_mapping:
            try:
                game.move(key_mapping[key])
                game.render(stdscr, previous_board=previous_board)
            except ValueError as e:
                stdscr.addstr(0, 0, str(e))
        else:
            stdscr.addstr(0, 0, "Invalid input. Use arrow keys or w, a, s, d.")

    stdscr.addstr(0, 0, "Game Over!")

if __name__ == "__main__":
    curses.wrapper(main)