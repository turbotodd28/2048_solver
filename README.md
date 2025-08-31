# 2048 Solver

A machine learning-based solver for the 2048 game using expectimax algorithm with configurable evaluation weights.

## Features

- **Expectimax AI**: Advanced search algorithm for optimal move selection
- **Tunable Weights**: Configurable evaluation function with multiple weight parameters
- **Multiple Presets**: Predefined weight configurations (conservative, aggressive, balanced, experimental)
- **Custom Weights**: Support for custom weight configurations via JSON
- **Performance Optimization**: Efficient implementation with depth limiting

## Quick Start

### Basic Usage
```bash
# Run with default balanced weights
python demo_expectimax_tunable.py

# Run with conservative preset (prioritizes keeping spaces open)
python demo_expectimax_tunable.py -P conservative

# Run with custom weights
python demo_expectimax_tunable.py -W '{"empty_spaces": 3.0, "corner_bonus": 12.0, "snake_pattern": 4.0}'
```

### Command Line Options
- `-d, --depth`: Search depth (default: 4, max: 6)
- `-c, --chance-samples`: Number of random tile placements to consider (default: 8)
- `-P, --preset`: Weight preset (conservative, aggressive, balanced, experimental)
- `-W, --weights`: Custom weights as JSON string
- `-q, --quiet`: Suppress board display for faster execution
- `-w, --delay`: Delay between moves in seconds (default: 0)
- `-m, --max-moves`: Maximum moves before stopping (default: 2000)

### Weight Parameters

| Parameter | Description | Higher = | Lower = |
|-----------|-------------|----------|---------|
| `empty_spaces` | Values empty cells | More conservative | More aggressive |
| `corner_bonus` | Bonus for max tile in corner | Always keep in corner | Flexible positioning |
| `snake_pattern` | Bonus for decreasing pattern | Strict snake pattern | Flexible pattern |
| `monotonicity` | Bonus for ordered rows/columns | Strict ordering | Flexible ordering |
| `smoothness` | Penalty for adjacent differences | Very smooth gradients | Tolerate differences |
| `merge_potential` | Bonus for immediate merges | Actively seek merges | Patient merging |

## Project Structure

```
2048_solver/
├── src/
│   ├── game.py              # Core 2048 game logic
│   ├── expectimax_tunable.py # Tunable expectimax implementation
│   ├── gui_2048.py          # Pygame GUI (optional)
│   └── animation_layer.py   # Animation support for GUI
├── demo_expectimax_tunable.py # Main demo script
├── test_game.py             # Unit tests
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## Performance

The solver typically achieves:
- **1024+ tiles** with balanced weights
- **2048+ tiles** with conservative weights
- **512+ tiles** with aggressive weights

Performance depends on search depth, chance samples, and weight configuration.

## Examples

### Conservative Strategy (for 2048+ tiles)
```bash
python demo_expectimax_tunable.py -P conservative -d 4 -c 12 -q
```

### Aggressive Strategy (for fast play)
```bash
python demo_expectimax_tunable.py -P aggressive -d 4 -c 8 -q
```

### Custom Configuration
```bash
python demo_expectimax_tunable.py -W '{"empty_spaces": 3.5, "corner_bonus": 15.0, "snake_pattern": 5.0, "monotonicity": 2.0, "smoothness": 0.4, "merge_potential": 0.03}' -d 4 -c 8 -q
```