#!/usr/bin/env python3
"""
Tunable expectimax demo with configurable evaluation weights
"""

import sys
import time
import argparse
import json
from src.game import Game2048
from src.expectimax import expectimax_best_action_tunable, WEIGHT_PRESETS, DEFAULT_WEIGHTS

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Tunable 2048 Expectimax AI Demo with Configurable Weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=4,
        help="Search depth (capped at 6 for practical performance)"
    )
    
    parser.add_argument(
        "--chance-samples", "-c",
        type=int,
        default=8,
        help="Number of random tile placements to consider"
    )
    
    parser.add_argument(
        "--delay", "-w",
        type=float,
        default=0.0,
        help="Delay between moves in seconds (0 = no delay)"
    )
    
    parser.add_argument(
        "--max-moves", "-m",
        type=int,
        default=2000,
        help="Maximum moves before stopping"
    )
    
    parser.add_argument(
        "--processes", "-p",
        type=int,
        default=None,
        help="Number of CPU processes to use (default: all available cores)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress board display (faster execution)"
    )
    
    parser.add_argument(
        "--preset", "-P",
        type=str,
        choices=['conservative', 'aggressive', 'balanced', 'experimental'],
        default='balanced',
        help="Predefined weight preset to use"
    )
    
    parser.add_argument(
        "--weights", "-W",
        type=str,
        help="Custom weights as JSON string (overrides preset)"
    )
    
    parser.add_argument(
        "--show-weights", "-S",
        action="store_true",
        help="Show current weights and exit"
    )
    
    return parser.parse_args()

def print_board(board):
    """Print the board in a nice format"""
    print("┌─────┬─────┬─────┬─────┐")
    for i, row in enumerate(board):
        print("│", end=" ")
        for j, val in enumerate(row):
            if val == 0:
                print("   ", end=" │ ")
            else:
                print(f"{val:3}", end=" │ ")
        print()
        if i < 3:
            print("├─────┼─────┼─────┼─────┤")
    print("└─────┴─────┴─────┴─────┘")

def print_weights(weights, name="Current"):
    """Print weights in a nice format"""
    print(f"\n{name} Weights:")
    print("=" * 50)
    for key, value in weights.items():
        print(f"{key:20}: {value:8.3f}")
    print("=" * 50)

def main():
    args = parse_arguments()
    
    # Determine which weights to use
    if args.weights:
        try:
            weights = json.loads(args.weights)
            print("✅ Using custom weights from command line")
        except json.JSONDecodeError:
            print("❌ Invalid JSON in --weights argument")
            return
    else:
        weights = WEIGHT_PRESETS[args.preset].copy()
        print(f"✅ Using '{args.preset}' preset")
    
    if args.show_weights:
        print_weights(weights, f"{args.preset.title()} Preset")
        return
    
    print("🚀 2048 Expectimax AI Demo (TUNABLE WEIGHTS)")
    print("=" * 65)
    print(f"🔍 Search Depth: {min(args.depth, 6)} (capped for performance)")
    print(f"🎲 Chance Samples: {args.chance_samples}")
    print(f"⏱️  Show Delay: {args.delay}s")
    print(f"🎯 Max Moves: {args.max_moves}")
    print(f"🖥️  CPU Cores: {args.processes or 'Auto-detect'}")
    print(f"🔇 Quiet Mode: {args.quiet}")
    print(f"⚖️  Weight Preset: {args.preset}")
    
    print_weights(weights)
    
    # Create a new game
    game = Game2048()
    
    # Direction mapping
    directions = ['up', 'down', 'left', 'right']
    direction_symbols = ['↑', '↓', '←', '→']
    
    move_count = 0
    
    if not args.quiet:
        print("\n🎯 Starting position:")
        print_board(game.board)
    
    start_time = time.time()
    
    while not game.is_game_over() and move_count < args.max_moves:
        print(f"\n📊 Score: {game.score} | Moves: {move_count}")
        print("🤖 AI is thinking (tunable weights)...")
        
        # Get the best move from tunable expectimax
        think_start = time.time()
        best_action = expectimax_best_action_tunable(
            game.board, 
            depth=args.depth, 
            chance_sample_k=args.chance_samples,
            weights=weights
        )
        think_time = time.time() - think_start
        
        direction = directions[best_action]
        symbol = direction_symbols[best_action]
        
        print(f"🎯 AI chooses: {direction} {symbol} (thought for {think_time:.2f}s)")
        
        # Make the move
        success = game.move(direction)
        if not success:
            print("❌ Invalid move!")
            break
            
        move_count += 1
        
        # Show the result
        print(f"✅ Move successful!")
        if not args.quiet:
            print_board(game.board)
        
        # Optional delay
        if args.delay > 0:
            time.sleep(args.delay)
    
    total_time = time.time() - start_time
    
    # Final results
    print("\n" + "=" * 65)
    print("🏁 Game Over!")
    print(f"📊 Final Score: {game.score}")
    print(f"🎯 Total Moves: {move_count}")
    print(f"🏆 Highest Tile: {game.board.max()}")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print(f"⚡ Average Time per Move: {total_time/move_count:.2f}s")
    
    # Performance analysis
    if game.board.max() >= 2048:
        print("🎉 AMAZING! Reached 2048+ tile!")
    elif game.board.max() >= 1024:
        print("🎉 Excellent! Reached 1024+ tile")
    elif game.board.max() >= 512:
        print("👍 Good! Reached 512+ tile")
    elif game.board.max() >= 256:
        print("👌 Decent! Reached 256+ tile")
    else:
        print("😐 Could be better...")
    
    if game.is_game_over():
        print("💀 No more valid moves possible")
    else:
        print("⏰ Reached maximum move limit")

if __name__ == "__main__":
    main()
