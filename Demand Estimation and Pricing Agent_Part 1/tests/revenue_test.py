""" All test scripts were generated with the help of LLMs """

import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is not installed. Please install it with: pip install numpy")
    sys.exit(1)

try:
    from settings import default_params_1
    import agents
    from make_env_2025 import make_env_agents
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def run_single_simulation(agent_name, n_steps=2500, verbose=False, seed=None):

    if seed is not None:
        np.random.seed(seed)
    
    env, agent_list = make_env_agents([agent_name], project_part=1)
    agent = agent_list[0]
    
    env.reset()
    customer_covariates, last_sale, state, inventories, time_until_replenish = \
        env.get_current_state_customer_to_send_agents()
    
    profits_over_time = []
    inventories_over_time = []
    
    for t in range(n_steps):
        action = agent.action((customer_covariates, last_sale, state, inventories, time_until_replenish))
        
        customer_covariates, last_sale, state, inventories, time_until_replenish = \
            env.step([action])
        
        profits_over_time.append(state[0])
        inventories_over_time.append(inventories[0])
        
        if verbose and t % 100 == 0:
            print(f"Step {t}: Profit = {state[0]:.2f}, Inventory = {inventories[0]}")
    
    final_profit = env.agent_profits[0]
    cumulative_buyer_utility = env.cumulative_buyer_utility
    
    return {
        'final_profit': final_profit,
        'cumulative_buyer_utility': cumulative_buyer_utility,
        'profits_over_time': profits_over_time,
        'inventories_over_time': inventories_over_time,
        'n_steps': n_steps
    }


def test_agent_revenue(agent_name="maria-laia-victoria-bhoomika", n_simulations=10, n_steps=1000, seed=None):

    print(f"Testing {agent_name} agent revenue performance...")
    print(f"Running {n_simulations} simulations with {n_steps} steps each...\n")
    
    results = []
    start_time = time.perf_counter()
    
    for i in range(n_simulations):
        sim_seed = seed + i if seed is not None else None
        result = run_single_simulation(agent_name, n_steps=n_steps, seed=sim_seed)
        results.append(result)
        
        if (i + 1) % max(1, n_simulations // 10) == 0:
            print(f"Completed {i + 1}/{n_simulations} simulations...")
    
    elapsed_time = time.perf_counter() - start_time
    
    final_profits = [r['final_profit'] for r in results]
    buyer_utilities = [r['cumulative_buyer_utility'] for r in results]
    
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
    
    summary = {
        'agent_name': agent_name,
        'n_simulations': n_simulations,
        'n_steps': n_steps,
        'profit_stats': profit_stats,
        'utility_stats': utility_stats,
        'all_profits': final_profits,
        'all_utilities': buyer_utilities,
        'elapsed_time': elapsed_time,
        'results': results
    }
    
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
    
    return summary


def compare_agents(agent_names, n_simulations=10, n_steps=1000, seed=None):

    print(f"Comparing {len(agent_names)} agents: {', '.join(agent_names)}\n")
    
    all_results = {}
    for agent_name in agent_names:
        print(f"\n{'='*70}")
        print(f"Testing {agent_name}")
        print(f"{'='*70}")
        results = test_agent_revenue(agent_name, n_simulations, n_steps, seed)
        all_results[agent_name] = results
    
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Agent Name':<30} {'Mean Profit':<15} {'Std Dev':<15} {'Max Profit':<15}")
    print("-" * 70)
    
    for agent_name, results in all_results.items():
        stats = results['profit_stats']
        print(f"{agent_name:<30} {stats['mean']:>14.2f}  {stats['std']:>14.2f}  {stats['max']:>14.2f}")
    
    best_agent = max(all_results.items(), 
                     key=lambda x: x[1]['profit_stats']['mean'])
    print(f"\nBest performing agent (by mean profit): {best_agent[0]}")
    print(f"  Mean profit: {best_agent[1]['profit_stats']['mean']:.2f}")
    print(f"{'='*70}\n")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test agent revenue performance')
    parser.add_argument('--agent', type=str, default='maria-laia-victoria-bhoomika',
                        help='Agent name to test')
    parser.add_argument('--simulations', type=int, default=10,
                        help='Number of simulations to run')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of steps per simulation')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--compare', nargs='+', default=None,
                        help='List of agent names to compare')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_agents(args.compare, args.simulations, args.steps, args.seed)
    else:
        test_agent_revenue(args.agent, args.simulations, args.steps, args.seed)