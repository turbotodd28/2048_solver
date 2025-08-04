#!/usr/bin/env python3
"""
Analysis script for 2048 reward sweep results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

def load_results(filename):
    """Load the reward sweep results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_sweep_results(data):
    """Analyze the reward sweep results"""
    
    print("=== 2048 REWARD SWEEP ANALYSIS ===\n")
    
    # Store summary statistics for each sweep
    sweep_stats = {}
    
    for sweep_name, sweep_data in data.items():
        print(f"--- {sweep_name.upper()} ---")
        print(f"Parameters: {sweep_data['params']}")
        
        rewards = np.array(sweep_data['eval_rewards'])
        max_tiles = np.array(sweep_data['eval_max_tiles'])
        
        # Calculate statistics
        stats = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'median_reward': np.median(rewards),
            'mean_max_tile': np.mean(max_tiles),
            'max_tile_achieved': np.max(max_tiles),
            'num_games': len(rewards),
            'success_rate_1024': np.sum(max_tiles >= 1024) / len(max_tiles),
            'success_rate_2048': np.sum(max_tiles >= 2048) / len(max_tiles),
            'success_rate_4096': np.sum(max_tiles >= 4096) / len(max_tiles),
            'success_rate_8192': np.sum(max_tiles >= 8192) / len(max_tiles),
            'games_with_negative_reward': np.sum(rewards < 0),
            'games_with_zero_reward': np.sum(rewards == 0),
            'merge_power': sweep_data['params']['merge_power']
        }
        
        sweep_stats[sweep_name] = stats
        
        print(f"Games played: {stats['num_games']}")
        print(f"Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Reward range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        print(f"Median reward: {stats['median_reward']:.2f}")
        print(f"Mean max tile: {stats['mean_max_tile']:.1f}")
        print(f"Highest tile achieved: {stats['max_tile_achieved']}")
        print(f"Success rates:")
        print(f"  - 1024 tile: {stats['success_rate_1024']:.1%}")
        print(f"  - 2048 tile: {stats['success_rate_2048']:.1%}")
        print(f"  - 4096 tile: {stats['success_rate_4096']:.1%}")
        print(f"  - 8192 tile: {stats['success_rate_8192']:.1%}")
        print(f"Games with negative reward: {stats['games_with_negative_reward']}")
        print(f"Games with zero reward: {stats['games_with_zero_reward']}")
        print()
    
    # Compare sweeps
    print("=== COMPARISON ACROSS SWEEPS ===")
    
    # Create comparison table
    comparison_data = []
    for sweep_name, stats in sweep_stats.items():
        comparison_data.append({
            'Sweep': sweep_name,
            'Merge Power': stats['merge_power'],
            'Mean Reward': stats['mean_reward'],
            'Std Reward': stats['std_reward'],
            'Max Reward': stats['max_reward'],
            'Mean Max Tile': stats['mean_max_tile'],
            'Max Tile Achieved': stats['max_tile_achieved'],
            '1024 Success Rate': stats['success_rate_1024'],
            '2048 Success Rate': stats['success_rate_2048'],
            '4096 Success Rate': stats['success_rate_4096'],
            '8192 Success Rate': stats['success_rate_8192']
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False, float_format='%.2f'))
    
    # Find best performing sweep
    best_sweep = max(sweep_stats.items(), key=lambda x: x[1]['mean_reward'])
    print(f"\nBest performing sweep: {best_sweep[0]} (mean reward: {best_sweep[1]['mean_reward']:.2f})")
    
    # Find sweep with highest max tile
    highest_tile_sweep = max(sweep_stats.items(), key=lambda x: x[1]['max_tile_achieved'])
    print(f"Sweep with highest tile: {highest_tile_sweep[0]} (max tile: {highest_tile_sweep[1]['max_tile_achieved']})")
    
    # Analyze merge power impact
    print("\n=== MERGE POWER IMPACT ANALYSIS ===")
    merge_powers = [stats['merge_power'] for stats in sweep_stats.values()]
    mean_rewards = [stats['mean_reward'] for stats in sweep_stats.values()]
    max_tiles = [stats['max_tile_achieved'] for stats in sweep_stats.values()]
    
    print("Merge Power vs Performance:")
    for i, (mp, mr, mt) in enumerate(zip(merge_powers, mean_rewards, max_tiles)):
        sweep_name = list(sweep_stats.keys())[i]
        print(f"  {sweep_name}: merge_power={mp}, mean_reward={mr:.2f}, max_tile={mt}")
    
    # Check for trends
    if len(merge_powers) > 1:
        reward_correlation = np.corrcoef(merge_powers, mean_rewards)[0, 1]
        print(f"\nCorrelation between merge power and mean reward: {reward_correlation:.3f}")
        
        if abs(reward_correlation) > 0.5:
            trend = "strong positive" if reward_correlation > 0 else "strong negative"
            print(f"Trend: {trend} correlation between merge power and performance")
        else:
            print("No strong correlation between merge power and performance")
    
    # Detailed analysis of each sweep
    print("\n=== DETAILED ANALYSIS ===")
    
    for sweep_name, sweep_data in data.items():
        print(f"\n{sweep_name.upper()} DETAILS:")
        rewards = np.array(sweep_data['eval_rewards'])
        max_tiles = np.array(sweep_data['eval_max_tiles'])
        
        # Reward distribution analysis
        print(f"  Reward distribution:")
        print(f"    - Very high (>10000): {np.sum(rewards > 10000)} games")
        print(f"    - High (1000-10000): {np.sum((rewards >= 1000) & (rewards <= 10000))} games")
        print(f"    - Medium (100-1000): {np.sum((rewards >= 100) & (rewards < 1000))} games")
        print(f"    - Low (0-100): {np.sum((rewards >= 0) & (rewards < 100))} games")
        print(f"    - Negative: {np.sum(rewards < 0)} games")
        
        # Max tile distribution
        print(f"  Max tile distribution:")
        unique_tiles, counts = np.unique(max_tiles, return_counts=True)
        for tile, count in zip(unique_tiles, counts):
            percentage = count / len(max_tiles) * 100
            print(f"    - {tile}: {count} games ({percentage:.1f}%)")
        
        # Find best and worst games
        best_game_idx = np.argmax(rewards)
        worst_game_idx = np.argmin(rewards)
        
        print(f"  Best game: reward={rewards[best_game_idx]:.2f}, max_tile={max_tiles[best_game_idx]}")
        print(f"  Worst game: reward={rewards[worst_game_idx]:.2f}, max_tile={max_tiles[worst_game_idx]}")
    
    return sweep_stats

def create_visualizations(data, sweep_stats):
    """Create visualizations of the results"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('2048 Reward Sweep Analysis', fontsize=16, fontweight='bold')
    
    # 1. Mean rewards by sweep
    sweep_names = list(sweep_stats.keys())
    mean_rewards = [sweep_stats[name]['mean_reward'] for name in sweep_names]
    std_rewards = [sweep_stats[name]['std_reward'] for name in sweep_names]
    
    axes[0, 0].bar(sweep_names, mean_rewards, yerr=std_rewards, capsize=5)
    axes[0, 0].set_title('Mean Rewards by Sweep')
    axes[0, 0].set_ylabel('Mean Reward')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Max tiles achieved by sweep
    max_tiles = [sweep_stats[name]['max_tile_achieved'] for name in sweep_names]
    axes[0, 1].bar(sweep_names, max_tiles)
    axes[0, 1].set_title('Highest Tile Achieved by Sweep')
    axes[0, 1].set_ylabel('Max Tile')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Success rates for different milestones
    milestones = ['1024', '2048', '4096', '8192']
    success_rates = []
    for milestone in milestones:
        rates = [sweep_stats[name][f'success_rate_{milestone}'] for name in sweep_names]
        success_rates.append(rates)
    
    x = np.arange(len(sweep_names))
    width = 0.2
    for i, (milestone, rates) in enumerate(zip(milestones, success_rates)):
        axes[0, 2].bar(x + i*width, rates, width, label=f'{milestone} tile')
    
    axes[0, 2].set_title('Success Rates by Milestone')
    axes[0, 2].set_ylabel('Success Rate')
    axes[0, 2].set_xticks(x + width * 1.5)
    axes[0, 2].set_xticklabels(sweep_names, rotation=45)
    axes[0, 2].legend()
    
    # 4. Reward distributions (box plots)
    reward_data = []
    labels = []
    for sweep_name, sweep_data in data.items():
        reward_data.append(sweep_data['eval_rewards'])
        labels.extend([sweep_name] * len(sweep_data['eval_rewards']))
    
    # Flatten data for box plot
    all_rewards = [r for rewards in reward_data for r in rewards]
    
    axes[1, 0].boxplot(reward_data, labels=sweep_names)
    axes[1, 0].set_title('Reward Distributions')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Max tile distributions
    max_tile_data = []
    for sweep_name, sweep_data in data.items():
        max_tile_data.append(sweep_data['eval_max_tiles'])
    
    axes[1, 1].boxplot(max_tile_data, labels=sweep_names)
    axes[1, 1].set_title('Max Tile Distributions')
    axes[1, 1].set_ylabel('Max Tile')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Merge power vs performance
    merge_powers = [sweep_stats[name]['merge_power'] for name in sweep_names]
    axes[1, 2].scatter(merge_powers, mean_rewards, s=100, alpha=0.7)
    axes[1, 2].set_title('Merge Power vs Mean Reward')
    axes[1, 2].set_xlabel('Merge Power')
    axes[1, 2].set_ylabel('Mean Reward')
    
    # Add trend line
    if len(merge_powers) > 1:
        z = np.polyfit(merge_powers, mean_rewards, 1)
        p = np.poly1d(z)
        axes[1, 2].plot(merge_powers, p(merge_powers), "r--", alpha=0.8)
    
    # Add sweep labels to scatter plot
    for i, sweep_name in enumerate(sweep_names):
        axes[1, 2].annotate(sweep_name, (merge_powers[i], mean_rewards[i]), 
                           xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('reward_sweep_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'reward_sweep_analysis.png'")
    
    return fig

def main():
    """Main analysis function"""
    
    # Load the data
    data = load_results('reward_sweep_full_results.json')
    
    # Analyze the results
    sweep_stats = analyze_sweep_results(data)
    
    # Create visualizations
    try:
        fig = create_visualizations(data, sweep_stats)
        print("\nAnalysis complete! Check the generated visualization for detailed insights.")
    except ImportError:
        print("\nMatplotlib/Seaborn not available. Skipping visualizations.")
        print("Install with: pip install matplotlib seaborn")

if __name__ == "__main__":
    main() 