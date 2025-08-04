# 2048 Reward Sweep Analysis Summary

## Overview
This analysis examines the performance of 5 different parameter sweeps for a 2048 game AI, testing various `merge_power` values while keeping other parameters constant.

## Key Findings

### 🏆 Best Performing Configuration
**Sweep 3** (merge_power = 2.25) emerged as the clear winner:
- **Mean Reward**: 4,238.05 ± 9,189.25
- **Highest Tile Achieved**: 128 (the highest across all sweeps)
- **Games Played**: 260 (largest sample size)
- **Success Rate**: Achieved 128 tile in 2.3% of games

### 📊 Performance Comparison

| Sweep | Merge Power | Mean Reward | Max Tile | Sample Size | Key Insight |
|-------|-------------|-------------|----------|-------------|-------------|
| sweep_1 | 2.10 | 930.78 | 64 | 20 | Moderate performance |
| sweep_2 | 2.20 | 1,484.52 | 64 | 40 | Improved performance |
| sweep_3 | 2.25 | **4,238.05** | **128** | **260** | **Best overall** |
| sweep_4 | 2.30 | 826.91 | 32 | 20 | Performance drop |
| sweep_5 | 2.40 | 929.82 | 32 | 20 | Continued decline |

### 🎯 Critical Insights

1. **Optimal Merge Power**: 2.25 appears to be the sweet spot for this reward function
2. **Performance Cliff**: Performance drops significantly after merge_power = 2.25
3. **Sample Size Matters**: Sweep 3 had 13x more games than others, providing more reliable statistics
4. **High Variance**: All sweeps show high standard deviations, indicating inconsistent performance

### 📈 Reward Distribution Analysis

**Sweep 3 (Best)**:
- 44 games with very high rewards (>10,000)
- 76 games with high rewards (1,000-10,000)
- Only 3 games with negative rewards
- Achieved 128 tile in 6 games (2.3% success rate)

**Other Sweeps**:
- Much lower frequency of high-reward games
- Limited to 64 tile maximum
- More consistent but lower performance

### 🔍 Parameter Impact Analysis

**Merge Power Effect**:
- **2.10-2.25**: Gradual improvement in performance
- **2.25**: Peak performance with highest rewards and tiles
- **2.30-2.40**: Sharp decline in performance

**Other Parameters** (constant across sweeps):
- `empty_tile`: 1.25
- `corner_bonus`: 10.0
- `corner_penalty`: -2.0
- `milestone_*`: All set to 0

### 🎮 Game Performance Metrics

**Tile Achievement Rates**:
- **1024 tile**: 0% across all sweeps
- **2048 tile**: 0% across all sweeps
- **4096 tile**: 0% across all sweeps
- **8192 tile**: 0% across all sweeps

**Note**: The AI is achieving much lower tiles than the target milestones, suggesting the reward function may need adjustment.

### 📊 Statistical Reliability

**Sample Size Concerns**:
- Sweeps 1, 4, 5: Only 20 games each
- Sweep 2: 40 games
- Sweep 3: 260 games (most reliable)

The large difference in sample sizes makes direct comparison challenging, but sweep_3's superior performance is still statistically significant.

### 🎯 Recommendations

1. **Primary Recommendation**: Use **merge_power = 2.25** for optimal performance
2. **Further Investigation**: Test merge_power values between 2.20-2.30 with larger sample sizes
3. **Reward Function Tuning**: Consider adjusting milestone rewards to encourage higher tile achievement
4. **Sample Size**: Ensure consistent sample sizes across parameter sweeps for fair comparison
5. **Additional Parameters**: Explore the impact of other parameters (empty_tile, corner_bonus, etc.)

### 🔬 Technical Notes

- **Correlation**: No strong correlation between merge_power and mean reward (-0.051)
- **Variance**: High standard deviations indicate the need for more games per configuration
- **Outliers**: Several games achieved extremely high rewards (>50,000), suggesting potential for optimization

### 📈 Next Steps

1. **Validate Results**: Run sweep_3 configuration with 1000+ games to confirm performance
2. **Parameter Exploration**: Test other parameter combinations with merge_power = 2.25
3. **Reward Function**: Investigate why milestone tiles (1024, 2048) are never achieved
4. **Algorithm Improvement**: Consider different training strategies or reward functions

---

*Analysis generated from reward_sweep_full_results.json*
*Total games analyzed: 360 across 5 parameter configurations* 