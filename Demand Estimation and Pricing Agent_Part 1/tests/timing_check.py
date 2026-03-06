"""
Timing check for agent decision speed.
Ensures each decision takes ≤0.5s.
Used LLMs for this test file.
"""
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is not installed. Please install it with: pip install numpy")
    sys.exit(1)

try:
    from settings import default_params_1
    import agents
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def generate_sample_observation():
    """Generate a sample observation tuple for testing."""
    new_buyer_covariates = np.array([np.random.uniform(0, 1) for _ in range(3)])
    last_sale = (np.nan, np.array([50.0]))
    state = np.array([0.0])
    inventories = np.array([10])
    time_until_replenish = 15
    return (new_buyer_covariates, last_sale, state, inventories, time_until_replenish)


def test_agent_timing(agent_name="maria-laia-victoria-bhoomika", n_trials=100):
    """
    Test agent decision speed.
    
    Args:
        agent_name: Name of the agent module (without .py)
        n_trials: Number of decision trials to run
    
    Returns:
        dict with timing statistics
    """
    print(f"Testing {agent_name} agent timing...")
    

    agent_module = agents.load(agent_name + ".py")
    agent = agent_module.Agent(0, default_params_1)
    

    obs = generate_sample_observation()
    _ = agent.action(obs)
    

    times = []
    for i in range(n_trials):
        obs = generate_sample_observation()
        start = time.perf_counter()
        _ = agent.action(obs)
        end = time.perf_counter()
        times.append(end - start)
    

    times_ms = [t * 1000 for t in times]
    avg_time = np.mean(times_ms)
    max_time = np.max(times_ms)
    min_time = np.min(times_ms)
    median_time = np.median(times_ms)
    p95_time = np.percentile(times_ms, 95)
    p99_time = np.percentile(times_ms, 99)
    
    results = {
        'avg_time': avg_time,
        'max_time': max_time,
        'min_time': min_time,
        'median_time': median_time,
        'p95_time': p95_time,
        'p99_time': p99_time,
        'n_trials': n_trials,
        'passes': max_time <= 500.0
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Timing Results for {agent_name}")
    print(f"{'='*60}")
    print(f"Number of trials: {n_trials}")
    print(f"\nTiming Statistics:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Median:  {median_time:.2f} ms")
    print(f"  Min:     {min_time:.2f} ms")
    print(f"  Max:     {max_time:.2f} ms")
    print(f"  95th percentile: {p95_time:.2f} ms")
    print(f"  99th percentile: {p99_time:.2f} ms")
    print(f"\nRequirement: ≤500 ms per decision")
    print(f"Status: {'PASS' if results['passes'] else 'FAIL'}")
    if not results['passes']:
        print(f"  Max time ({max_time:.2f} ms) exceeds 500 ms limit")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    results = test_agent_timing()
    sys.exit(0 if results['passes'] else 1)

