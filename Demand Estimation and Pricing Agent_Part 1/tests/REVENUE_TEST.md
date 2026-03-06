# Revenue Test Instructions

All test scripts were generated with the help of LLMs

```python
import time
import numpy as np
from settings import default_params_1
from make_env_2025 import make_env_agents

# Configuration
agent_name = "maria-laia-victoria-bhoomika"
n_simulations = 10  # Number of simulation runs
n_steps = 1000      # Number of customers (each step = 1 customer arriving)
seed = None         # Set to a number for reproducibility (e.g., 42)

print(f"Testing {agent_name} agent revenue performance...")
print(f"Running {n_simulations} simulations with {n_steps} steps each...\n")

# Track results
final_profits = []
buyer_utilities = []
start_time = time.perf_counter()

for sim_num in range(n_simulations):
    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed + sim_num)
    
    # Create environment and agent
    env, agent_list = make_env_agents([agent_name], project_part=1)
    agent = agent_list[0]
    
    # Reset environment
    env.reset()
    customer_covariates, last_sale, state, inventories, time_until_replenish = \
        env.get_current_state_customer_to_send_agents()
    
    # Run simulation
    for t in range(n_steps):
        # Agent makes decision
        action = agent.action((customer_covariates, last_sale, state, inventories, time_until_replenish))
        
        # Environment steps
        customer_covariates, last_sale, state, inventories, time_until_replenish = \
            env.step([action])
    
    # Get final metrics
    final_profit = env.agent_profits[0]
    cumulative_buyer_utility = env.cumulative_buyer_utility
    
    final_profits.append(final_profit)
    buyer_utilities.append(cumulative_buyer_utility)
    
    # Progress update
    if (sim_num + 1) % max(1, n_simulations // 10) == 0:
        print(f"Completed {sim_num + 1}/{n_simulations} simulations...")

elapsed_time = time.perf_counter() - start_time

# Calculate statistics
final_profits = np.array(final_profits)
buyer_utilities = np.array(buyer_utilities)

profit_stats = {
    'mean': np.mean(final_profits),
    'median': np.median(final_profits),
    'std': np.std(final_profits),
    'min': np.min(final_profits),
    'max': np.max(final_profits),
    'p25': np.percentile(final_profits, 25),
    'p75': np.percentile(final_profits, 75),
    'p95': np.percentile(final_profits, 95),
}

utility_stats = {
    'mean': np.mean(buyer_utilities),
    'median': np.median(buyer_utilities),
    'std': np.std(buyer_utilities),
}

# Print results
print(f"\n{'='*70}")
print(f"Revenue Test Results for {agent_name}")
print(f"{'='*70}")
print(f"Number of simulations: {n_simulations}")
print(f"Steps per simulation: {n_steps}")
print(f"Total elapsed time: {elapsed_time:.2f} seconds")
print(f"\nProfit Statistics:")
print(f"  Mean:        {profit_stats['mean']:.2f}")
print(f"  Median:      {profit_stats['median']:.2f}")
print(f"  Std Dev:     {profit_stats['std']:.2f}")
print(f"  Min:         {profit_stats['min']:.2f}")
print(f"  Max:         {profit_stats['max']:.2f}")
print(f"  25th percentile: {profit_stats['p25']:.2f}")
print(f"  75th percentile: {profit_stats['p75']:.2f}")
print(f"  95th percentile: {profit_stats['p95']:.2f}")
print(f"\nBuyer Utility Statistics:")
print(f"  Mean:        {utility_stats['mean']:.2f}")
print(f"  Median:      {utility_stats['median']:.2f}")
print(f"  Std Dev:     {utility_stats['std']:.2f}")
print(f"{'='*70}\n")
```

## Comparing Multiple Agents

To compare your agent with a baseline, run the test twice with different agent names:

```python
# Test your optimized agent
agent_name = "maria-laia-victoria-bhoomika"
# ... run test code above ...

optimized_profit = profit_stats['mean']

# Test baseline agent
agent_name = "dummy_fixed_prices"
# ... run test code above again ...

baseline_profit = profit_stats['mean']

# Calculate improvement
improvement = ((optimized_profit - baseline_profit) / baseline_profit) * 100
print(f"Improvement: {improvement:+.2f}%")
```

## What to Look For:

- **Mean Profit**: Average revenue across all simulations (higher is better)
- **Median Profit**: Middle value (less affected by outliers)
- **Std Dev**: Consistency of performance (lower is better for stability)
- **Min/Max**: Range of performance
- **Percentiles**: Distribution of results

## Tips:

1. **Use a seed** for reproducibility: Set `seed = 42` to get consistent results
2. **Increase simulations** for more reliable statistics: Try `n_simulations = 20` or more
3. **Change number of customers**: Modify `n_steps` - each step = 1 customer
   - `n_steps = 500` → 500 customers
   - `n_steps = 2000` → 2000 customers
   - `n_steps = 1000` is a good default
4. **Compare versions**: Run the test before and after optimizations to measure improvement

The optimizations we made (two-stage price optimization and improved theta calculation) should improve your mean profit!

