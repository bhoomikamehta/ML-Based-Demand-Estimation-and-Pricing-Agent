"""
REVENUE TEST - Copy this entire code block into a new cell in your notebook
This version imports make_env directly, so it works standalone
All test scripts were generated with the help of LLMs
"""

# Import if not already imported (safe to run even if already imported)
try:
    import make_env_2025 as make_env
except:
    pass  # Already imported

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
    
    # Create environment and agent (using the same setup as your notebook)
    env, agent_list = make_env.make_env_agents(
        agentnames=[agent_name], 
        project_part=1,
        first_file='data/datafile1_2025.csv', 
        second_file='data/datafile2_2025.csv'
    )
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

# Optional: Store results for later comparison
# results = {
#     'profits': final_profits,
#     'utilities': buyer_utilities,
#     'stats': profit_stats
# }

