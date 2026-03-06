# Dynamic Pricing Tournament Agent

## Overview

An intelligent pricing agent that competes in a real-time marketplace tournament. The system uses machine learning to predict customer demand and game theory to respond to competitor behavior, balancing immediate revenue against long-term strategic positioning.

**Key Results:**
- **Part 1 Demand Model:** Gradient Boosting classifier achieving 0.9940 ROC-AUC
- **Part 2 Demand Model:** Neural Networks (3 hidden layers: 128, 64, 32) achieving 0.9953 ROC-AUC
- **Real-time Performance:** <30ms decision time per customer
- **Competitive Strategy:** Adaptive pricing with opponent modeling and tit-for-tat cooperation

---

## Project Structure

### Part 1: Demand Estimation & Price Optimization

**Goal:** Maximize revenue by predicting customer purchase probability and setting optimal prices under inventory constraints.

**Approach:**
- **Demand Estimation:** Trained Gradient Boosting model on 50,000 customer transactions with price and 3 covariates
- **Price Optimization:** Vectorized expected revenue calculation across price grid
- **Inventory Management:** Dynamic threshold system (θ) balancing immediate sales vs. future opportunities

**Key Features:**
- Fully vectorized implementation with pre-allocated arrays
- Opportunity cost framework: Accept customer only if EV ≥ θ(inventory, time)
- Average expected revenue: $63.62 per customer
- Zero timeouts across all leaderboard runs

### Part 2: Multi-Agent Competition

**Goal:** Maximize revenue in head-to-head competition where customers choose the lower price.

**Approach:**
- **Improved Demand Model:** Neural Networks (3 layers: 128, 64, 32) replaced Gradient Boosting for higher accuracy
- **Opponent Modeling:** Ridge regression predicts competitor prices using historical data (opponent prices, customer features, win/loss outcomes)
- **Adaptive Pricing with α ∈ [0.5, 1.0]:**
  - Opponent passive → Price near p* (optimal)
  - Moderate competition → Competitive undercutting
  - Aggressive opponent → Maintain floor (avoid race-to-bottom)
- **Tit-for-Tat Cooperation:** Dynamic cooperation score encourages mutual high pricing
- **Strategic Exploration:** 10% random probing in first 20 rounds to gather training data

**Competitive Features:**
- Inventory-aware threshold: Lower θ when opponent out of stock, raise when I have advantage
- Cooperation detection based on Prisoner's Dilemma framework
- Race-to-bottom protection: Price bounds (0.65× to 1.0× p*)
- Smoothing factor (α=0.25) prevents overreaction to price fluctuations

---

## Technical Implementation

### Model Performance

**Part 1:**
| Model | ROC-AUC | Training Time | Selected |
|-------|---------|---------------|----------|
| Logistic Regression | 0.9366 | ~1s | Baseline |
| Random Forest (n=200, d=15) | 0.9936 | ~30s | - |
| **Gradient Boosting (n=100, d=7)** | **0.9940** | ~45s | ✓ |

**Part 2:**
| Model | ROC-AUC | Notes |
|-------|---------|-------|
| Gradient Boosting (from Part 1) | 0.9940 | Initial baseline |
| XGBoost | - | Tested |
| **Neural Networks (128, 64, 32)** | **0.9953** | ✓ Selected |

### Price Grid Optimization

Testing showed optimal trade-offs:
- **Ngrid=200:** 0.5355% regret, 4.98s runtime (Part 1)
- **Ngrid=400:** 0.0395% improvement, 7.06s runtime (too slow)
- **Ngrid=100:** Selected for Part 2 (speed priority in competition)

### Performance Optimizations

**Part 1:**
- Pre-allocated feature arrays (`_base_features`) - avoid repeated memory allocation
- Vectorized probability computation - 10× faster than loops
- Only update covariate columns per customer

**Part 2:**
- Cached p* calculation - saves ~161 predictions per customer
- Reduced grid size: 200→100 points for faster real-time decisions
- Sparse model updates: Retrain Ridge every 10 customers
- Batch operations throughout

**Result:** Consistent <30ms per decision, zero timeouts

## Running the Agent
```python
# Initialize agent
from agents.models import Agent

agent = Agent(
    agent_number=0,
    params={
        'project_part': 2,
        'inventory_limit': 20,
        'inventory_replenish': 20
    }
)

# Get price for customer
obs = (new_buyer_covariates, last_sale, state, inventories, time_until_replenish)
price = agent.action(obs)
```

## Tools & Technologies

- **ML:** scikit-learn (Gradient Boosting, Ridge Regression), NumPy, pandas
- **Evaluation:** 5-fold cross-validation, ROC-AUC, calibration analysis
- **Optimization:** Vectorized operations, array pre-allocation, caching
- **Game Theory:** Tit-for-tat, Nash equilibrium analysis, Prisoner's Dilemma

## Final Report

📄 **[Full Project Report](#)** _(link coming soon)_