#!/usr/bin/env python3
"""
2048 Game - Command Line Interface
Play 2048 using arrow keys or WASD
"""

import sys
import os
import termios
import tty
import time
from src.game import Game2048
import numpy as np

# Updated high-contrast ANSI colors
COLORS = {
    2: '\033[97m',    # white
    4: '\033[90m',    # bright black
    8: '\033[36m',    # cyan
    16: '\033[31m',   # red
    32: '\033[32m',   # green
    64: '\033[33m',   # yellow
    128: '\033[35m',  # magenta
    256: '\033[34m',  # blue
    512: '\033[91m',  # bright red
    1024: '\033[92m', # bright green
    2048: '\033[95m', # bright magenta
    4096: '\033[93m', # bright yellow
}
RESET = '\033[0m'
BOLD = '\033[1m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def get_color_for_tile(value):
    return COLORS.get(value, '\033[93m')  # fallback: bright yellow

def format_tile(value):
    if value == 0:
        return "     "
    else:
        color = get_color_for_tile(value)
        num_str = str(value)
        padding = " " * (5 - len(num_str))
        return f"{padding}{color}{BOLD}{num_str}{RESET}"

def draw_board(game):
    clear_screen()

    sys.stdout.write(f"{BOLD}2048 Game{RESET}\n")
    sys.stdout.write(f"Score: {GREEN}{game.score}{RESET} | Moves: {BLUE}{game.move_count}{RESET}\n")
    sys.stdout.write("Use arrow keys or WASD to move, 'q' to quit\n\n")

    sys.stdout.write("┌─────┬─────┬─────┬─────┐\n")
    for i in range(4):
        sys.stdout.write("│")
        for j in range(4):
            tile_str = format_tile(game.board[i, j])
            sys.stdout.write(tile_str + "│")
        sys.stdout.write("\n")
        if i < 3:
            sys.stdout.write("├─────┼─────┼─────┼─────┤\n")
    sys.stdout.write("└─────┴─────┴─────┴─────┘\n\n")
    sys.stdout.flush()

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def get_arrow_key():
    ch = get_key()
    if ch == '\x1b':
        next_ch = get_key()
        if next_ch == '[':
            third_ch = get_key()
            return {
                'A': 'up',
                'B': 'down',
                'C': 'right',
                'D': 'left'
            }.get(third_ch, None)
    elif ch.lower() in 'wasd':
        return {
            'w': 'up',
            'a': 'left',
            's': 'down',
            'd': 'right'
        }.get(ch.lower())
    elif ch.lower() == 'q':
        return 'quit'
    return None

def print_move_result(direction, was_valid):
    direction_upper = direction.upper()
    if was_valid:
        sys.stdout.write(f"{GREEN}{direction_upper}{RESET}\n")
    else:
        sys.stdout.write(f"{RED}{direction_upper}\nINVALID MOVE{RESET}\n")
    sys.stdout.flush()
    time.sleep(0.5)

def main():
    game = Game2048()

    sys.stdout.write(f"{BOLD}Welcome to 2048!{RESET}\n")
    sys.stdout.write("Use arrow keys or WASD to move tiles\n")
    sys.stdout.write("Press 'q' to quit\n")
    sys.stdout.write("Press any key to start...\n")
    sys.stdout.flush()
    get_key()

    while not game.is_game_over():
        draw_board(game)
        direction = get_arrow_key()
        if direction == 'quit':
            sys.stdout.write(f"\n{YELLOW}Game ended. Final score: {game.score}{RESET}\n")
            sys.stdout.flush()
            break
        if direction is None:
            continue
        board_before = game.board.copy()
        try:
            game.move(direction)
            was_valid = not np.array_equal(board_before, game.board)
            print_move_result(direction, was_valid)
        except ValueError:
            print_move_result(direction, False)

    draw_board(game)
    sys.stdout.write(f"\n{BOLD}GAME OVER!{RESET}\n")
    sys.stdout.write(f"Final Score: {GREEN}{game.score}{RESET}\n")
    sys.stdout.write(f"Total Moves: {BLUE}{game.move_count}{RESET}\n")
    sys.stdout.write(f"Highest Tile: {YELLOW}{np.max(game.board)}{RESET}\n")
    sys.stdout.flush()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.write(f"\n\n{YELLOW}Game interrupted. Thanks for playing!{RESET}\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(f"\n{RED}Error: {e}{RESET}\n")
        sys.stdout.write("Make sure you're running this in a terminal that supports colors and arrow keys.\n")
        sys.stdout.flush()
