"""
Timing test code - Copy this into a Jupyter notebook cell to test
This is a standalone version you can run in your notebook
All test scripts were generated with the help of LLMs
"""

import time
import numpy as np
from settings import default_params_1
import agents

def generate_sample_observation():
    """Generate a sample observation tuple for testing."""
    new_buyer_covariates = np.array([np.random.uniform(0, 1) for _ in range(3)])
    last_sale = (np.nan, np.array([50.0]))
    state = np.array([0.0])
    inventories = np.array([10])
    time_until_replenish = 15
    return (new_buyer_covariates, last_sale, state, inventories, time_until_replenish)

# Load agent
agent_name = "maria-laia-victoria-bhoomika"
agent_module = agents.load(agent_name + ".py")
agent = agent_module.Agent(0, default_params_1)

# Warm-up run
print("Running warm-up...")
obs = generate_sample_observation()
_ = agent.action(obs)
print("Warm-up complete\n")

# Time multiple decisions
n_trials = 100
print(f"Running {n_trials} timing trials...")
times = []
for i in range(n_trials):
    obs = generate_sample_observation()
    start = time.perf_counter()
    _ = agent.action(obs)
    end = time.perf_counter()
    times.append((end - start) * 1000)  # Convert to milliseconds

# Calculate statistics
times = np.array(times)
avg_time = np.mean(times)
max_time = np.max(times)
min_time = np.min(times)
median_time = np.median(times)
p95_time = np.percentile(times, 95)
p99_time = np.percentile(times, 99)

# Print results
print("="*60)
print("Timing Results")
print("="*60)
print(f"Number of trials: {n_trials}")
print(f"\nTiming Statistics (in milliseconds):")
print(f"  Average:     {avg_time:.2f} ms")
print(f"  Median:      {median_time:.2f} ms")
print(f"  Min:         {min_time:.2f} ms")
print(f"  Max:         {max_time:.2f} ms")
print(f"  95th percentile: {p95_time:.2f} ms")
print(f"  99th percentile: {p99_time:.2f} ms")
print(f"\nRequirement: ≤500 ms per decision")

passes = max_time <= 500.0
if passes:
    print(f"Status: PASS (all decisions under 500ms)")
else:
    print(f"Status: FAIL (max time {max_time:.2f} ms exceeds 500 ms limit)")

print("="*60)

