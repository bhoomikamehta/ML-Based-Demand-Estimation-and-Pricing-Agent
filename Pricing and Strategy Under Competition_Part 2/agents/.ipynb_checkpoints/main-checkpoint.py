import random
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


'''
This template serves as a starting point for your agent.
'''


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        
        self.project_part = params['project_part'] 

        ### starting remaining inventory and inventory replenish rate are provided
        ## every time the inventory is replenished, it is set to the inventory limit
        ## the inventory_replenish rate is how often the inventory is replenished
        ## for example, we will run with inventory_replenish = 20, with the limit of 11. Then, the inventory will be replenished every 20 time steps (time steps 0, 20, 40, ...) and the inventory will be set to 11 at those time steps. 
        self.remaining_inventory = params['inventory_limit']
        self.inventory_replenish = params['inventory_replenish']

        # Loading model & scalar and then storing as instance variables
        scaler_path = f"agents/models/scaler.pkl"
        model_path = f"agents/models/demand_model.pkl"

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        with open(model_path, "rb") as f:
            self.demand_model = pickle.load(f)

        # Very high price to reject customers
        self.reject_price = 452.339045127419 * 2

        #based on the calculation in the graph for revenue distribution in the nb
        # low_prices = np.linspace(0.01, 40.86, 20)  # Up to 25th percentile as per nb
        # mid_prices = np.linspace(40.86, 81.70, 100)  # 25th to 75th percentile as per nb
        # high_prices = np.linspace(81.70, 452.339045127419, 40)  #max as per nb

        # Reduce grid size for to meet time constraints
        if self.project_part == 2:
            prices = np.linspace(0.01, 452.339045127419, 100)
        else:
            prices = np.linspace(0.01, 452.339045127419, 200)
        self.price_grid = np.unique(np.concatenate([prices]))

        # self.price_grid = np.unique(np.concatenate([low_prices, mid_prices, high_prices]))
        self.price_grid = np.unique(np.append(self.price_grid, self.reject_price))

        self.n_prices = len(self.price_grid)

        self._base_features = np.zeros((self.n_prices, 4))
        self._base_features[:, 0] = self.price_grid

        # self.avg_expected_revenue = 33.40450734169378 # as calculated in the python nb
        # self.theta_scale = 0.0055 #as observed in testing lower base is better so to ground the avg value
        # self.avg_expected_revenue = 63.622046695420046 # as calculated in the python nb
        self.avg_expected_revenue = 59.59415735288544 # as calculated in the python nb
        self.theta_scale = 0.01 #as observed in testing lower base is better so to ground the avg value

        ### useful if you want to use a more complex price prediction model
        ### note that you will need to change the name of the path and this agent file when submitting
        ### complications: pickle works with any machine learning models defined in sklearn, xgboost, etc.
        ### however, this does not work with custom defined classes, due to the way pickle serializes objects
        ### refer to './yourteamname/create_model.ipynb' for a quick tutorial on how to use pickle
        # self.filename = './[yourteamname]/trained_model'
        # self.trained_model = pickle.load(open(self.filename, 'rb'))
        ### potentially useful for Part 2 -- When competition is between two agents
        ### and you want to keep track of the opponent's status
        self.opponent_number = 1 - agent_number  # index for opponent
        
        # PART 2: Opponent tracking and competitive pricing
        self.opponent_price_history = []
        self.opponent_behavior_window = 50
        self.last_customer_covariates = None
        
        # Simple opponent price predictor
        self.opponent_price_model = Ridge(alpha=1.0)
        self.opponent_model_trained = False
        
        # Cooperation tracking for tit-for-tat strategy
        self.cooperation_score = 0.5
        self.opponent_cooperation_score = 0.5
        
        #Cache last p_star to avoid recalculation
        self._last_p_star_cache = None
        
        # Track rounds
        self.rounds_played = 0

        self._opponent_model_update_counter = 0
        self._opponent_model_update_frequency = 10  # Retrain every 10 customers  

    def _process_last_sale(
            self, 
            last_sale,
            state,
            inventories,
            time_until_replenish
        ):
        '''
        This function updates your internal state based on the last sale that occurred.
        This template shows you several ways you can keep track of important metrics.
        '''
        ### keep track of who if customer bought
        did_customer_buy_from_me = (last_sale[0] == self.this_agent_number)
        
        ### keep track of the prices offered in the last sale
        my_last_prices = last_sale[1][self.this_agent_number]

        ### keep track of the profit after the last sale
        my_current_profit = state[self.this_agent_number]

        ### keep track of the inventory levels after the last sale
        self.remaining_inventory = inventories[self.this_agent_number]

        ### keep track of the time until the next replenishment
        time_until_replenish = time_until_replenish

        ### Track opponent pricing behavior and update models
        if self.project_part == 2 and self.last_customer_covariates is not None:
            did_customer_buy_from_opponent = (last_sale[0] == self.opponent_number)
            opponent_last_prices = last_sale[1][self.opponent_number]
            opponent_current_profit = state[self.opponent_number]
            opponent_inventory = inventories[self.opponent_number]
            # Store opponent pricing behavior with richer features
            round_in_cycle = 20 - (time_until_replenish % 20) if time_until_replenish > 0 else 0
            self.opponent_price_history.append({
                'covariates': self.last_customer_covariates.copy(),
                'opponent_price': opponent_last_prices,
                'won': did_customer_buy_from_opponent,
                'my_p_star': self._last_p_star_cache if self._last_p_star_cache is not None else 0,
                'time_until_replenish': time_until_replenish,
                'opponent_inventory': opponent_inventory,
                'round_in_cycle': round_in_cycle
            })
            
            if len(self.opponent_price_history) > self.opponent_behavior_window:
                self.opponent_price_history.pop(0)
            
            # save computation time
            if len(self.opponent_price_history) >= 10:
                self._opponent_model_update_counter += 1
                if self._opponent_model_update_counter >= self._opponent_model_update_frequency:
                    self._update_opponent_model()
                    self._opponent_model_update_counter = 0
            
            # Pass cached p_star to avoid recalculation
            self._update_cooperation_scores(last_sale, self._last_p_star_cache)
    
    def _get_best_price_and_expected_revenue(self, buyer_covariates):

        # Optimized code in the python nb
        x_raw = self._base_features.copy()
        x_raw[:, 1] = buyer_covariates[0]
        x_raw[:, 2] = buyer_covariates[1]
        x_raw[:, 3] = buyer_covariates[2]

        x_scaled = self.scaler.transform(x_raw)

        # Get purchase probabilities (matches nb: predict_proba[:, 1])
        purchase_probs = self.demand_model.predict_proba(x_scaled)[:, 1]

        # Compute expected revenue for each price
        expected_revenues = self.price_grid * purchase_probs

        # Find the price that maximizes expected revenue
        best_idx = np.argmax(expected_revenues)
        best_price = float(self.price_grid[best_idx])
        best_expected_revenue = float(expected_revenues[best_idx])

        return best_price, best_expected_revenue
    
    def _compute_theta(self, inventory, time_until_replenish, opponent_inventory=None):
        '''
        Compute threshold θ based on inventory and time until replenish.
        
        Key insight: θ represents the opportunity cost of selling now.
        - When inventory is low relative to remaining customers, θ should be HIGH (be selective)
        - When inventory is high or replenishment is soon, θ should be LOW (be less selective)
        '''

        remaining_customers = time_until_replenish
        base_theta = self.avg_expected_revenue * self.theta_scale

        # When no inventory setting a very high threshold to reject all customers
        if inventory <= 0:
            return base_theta * 10.0

        # Lower theta when inventory is high - be aggressive to sell
        if inventory >= 18:
            base_adjustment = 0.5
        else:
            t = remaining_customers
            T = 20.0
            time_factor = t / T

            scarcity_ratio = remaining_customers / inventory
            max_scarcity = 15.0
            normalized_scarcity = min(scarcity_ratio / max_scarcity, 1.0)

            alpha = 0.5
            adjustment_factor = 1.0 + alpha * normalized_scarcity * time_factor
            adjustment_factor = min(adjustment_factor, 2.5)
            base_adjustment = adjustment_factor

        #Inventory-aware competition
        if self.project_part == 2 and opponent_inventory is not None:
            if opponent_inventory <= 0:
                # Opponent can't compete best price
                return base_theta * 0.5
            elif opponent_inventory < inventory:
                # We have more inventory be picky
                inventory_advantage = (inventory - opponent_inventory) / 10.0
                return base_theta * base_adjustment * (1.0 + 0.2 * min(inventory_advantage, 1.0))

        return base_theta * base_adjustment

    def _update_opponent_model(self):
        if len(self.opponent_price_history) < 10:
            return
        
        # Build feature array more efficiently (avoid concatenate in loop)
        n_samples = len(self.opponent_price_history)
        n_features = 7  # 3 covariates + 4 additional features
        
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples)
        
        for i, h in enumerate(self.opponent_price_history):
            # Fill covariates (first 3 features)
            X[i, :3] = h['covariates']
            # Fill additional features
            X[i, 3] = h.get('my_p_star', 0)
            X[i, 4] = h.get('time_until_replenish', 0)
            X[i, 5] = h.get('opponent_inventory', 0)
            X[i, 6] = h.get('round_in_cycle', 0)
            y[i] = h['opponent_price']
        
        # Filter out NaN values
        valid_mask = ~np.isnan(y)
        if X.size > 0:
            valid_mask = valid_mask & ~np.isnan(X).any(axis=1)
        
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(y_valid) >= 5:
            self.opponent_price_model.fit(X_valid, y_valid)
            self.opponent_model_trained = True

    def _predict_opponent_price(self, buyer_covariates, p_star=None, time_until_replenish=0, opponent_inventory=0):
        if not self.opponent_model_trained or len(self.opponent_price_history) < 10:
            # Fallback: assume opponent prices 20% below our best
            if p_star is None:
                p_star, _ = self._get_best_price_and_expected_revenue(buyer_covariates)
            return p_star * 0.8
        
        round_in_cycle = 20 - (time_until_replenish % 20) if time_until_replenish > 0 else 0
        if p_star is None:
            p_star, _ = self._get_best_price_and_expected_revenue(buyer_covariates)
        
        features = np.concatenate([
            buyer_covariates,
            [p_star, time_until_replenish, opponent_inventory, round_in_cycle]
        ])
        
        #predicted = self.opponent_price_model.predict([features])[0]
        #return max(0.01, predicted)
        # instead will apply the following to smooth prediction and avoid overreacting
        predicted_raw = self.opponent_price_model.predict([features])[0]
        alpha = 0.25   # smoothing factor
        if hasattr(self, "_opp_pred_smooth"):
            predicted = alpha * predicted_raw + (1 - alpha) * self._opp_pred_smooth
        else:
            predicted = predicted_raw

        self._opp_pred_smooth = predicted
        return max(0.01, predicted)


    def _detect_alternation_pattern(self):
        """Check if opponent appears to be alternating high/low prices"""
        if len(self.opponent_price_history) < 10:
            return False
        
        # More efficient: use numpy array instead of list comprehension
        recent_prices = np.array([h['opponent_price'] for h in self.opponent_price_history[-10:]])
        if len(recent_prices) < 2:
            return False
        
        #check for high-low switching
        is_high = recent_prices > 100
        switches = np.sum(is_high[1:] != is_high[:-1])
        return switches >= 7

    def _update_cooperation_scores(self, last_sale, p_star=None):
        if self.last_customer_covariates is None:
            return
        
        opponent_price = last_sale[1][self.opponent_number]
        my_price = last_sale[1][self.this_agent_number]
        
        # Use provided p_star if available to avoid expensive recalculation
        if p_star is None:
            p_star, _ = self._get_best_price_and_expected_revenue(self.last_customer_covariates)
        
        #alternation pattern
        if self._detect_alternation_pattern():
            self.opponent_cooperation_score = min(1.0, self.opponent_cooperation_score + 0.15)
        
        # Opponent cooperation: to check reasonable prices?
        if opponent_price >= p_star * 0.8:
            self.opponent_cooperation_score = min(1.0, self.opponent_cooperation_score + 0.1)
        elif opponent_price < p_star * 0.5:
            self.opponent_cooperation_score = max(0.0, self.opponent_cooperation_score - 0.1)
        
        # cooperation adjustment
        if self.opponent_cooperation_score > 0.6:
            self.cooperation_score = min(1.0, self.cooperation_score + 0.05)
        elif self.opponent_cooperation_score < 0.4:
            self.cooperation_score = max(0.0, self.cooperation_score - 0.05)

    def _get_purchase_probability(self, buyer_covariates, price):
        # Pre-allocate array to save time
        x_raw = np.array([[price, buyer_covariates[0], buyer_covariates[1], buyer_covariates[2]]])
        x_scaled = self.scaler.transform(x_raw)
        return self.demand_model.predict_proba(x_scaled)[0, 1]
    
    def _get_purchase_probability_batch(self, buyer_covariates, prices):
        """Compute probabilities for multiple prices at once (performance optimization)"""
        n = len(prices)
        x_raw = np.zeros((n, 4))
        x_raw[:, 0] = prices
        x_raw[:, 1:] = buyer_covariates
        x_scaled = self.scaler.transform(x_raw)
        return self.demand_model.predict_proba(x_scaled)[:, 1]

    def _get_competitive_price(self, buyer_covariates, p_star, EV_star, time_until_replenish=0, opponent_inventory=0, my_inventory=0):
        predicted_opp = self._predict_opponent_price(buyer_covariates, p_star, time_until_replenish, opponent_inventory)
        
        # competitive pricing logic to avoid race-to-bottom
        # find the price that maximizes expected value
        
        if predicted_opp > p_star * 1.1:
            # Opponent pricing high best price
            return p_star
        elif predicted_opp > p_star * 0.8:
            # Moderate competition - price just below but not much
            # Use 95% of opponent price or 70% of optimal whichever is higher
            # try 0.98 and 0.75 instead
            competitive_price = max(predicted_opp * 0.98, p_star * 0.75)
        else:
            # Aggressive opponent let it goooo
            # more if inventory is low
            # avoid the chase to the bottom
            competitive_price = p_star * 0.85
        
        # Apply tit-for-tat adjustment
        cooperation_adj = (self.cooperation_score - 0.5) * 0.15
        competitive_price = competitive_price * (1.0 + cooperation_adj)
        competitive_price = np.clip(competitive_price, p_star * 0.65, p_star * 1.0)
        
        return competitive_price

    def action(self, obs):
        '''
        This function is called every time the agent needs to choose an action by the environment.

        The input 'obs' is a 5 tuple, containing the following information:
        -- new_buyer_covariates: a vector of length 3, containing the covariates of the new buyer.
        -- last_sale: a tuple of length 2. The first element is the index of the agent that made the last sale, if it is NaN, then the customer did not make a purchase. The second element is a numpy array of length n_agents, containing the prices that were offered by each agent in the last sale.
        -- state: a vector of length n_agents, containing the current profit of each agent.
        -- inventories: a vector of length n_agents, containing the current inventory level of each agent.
        -- time_until_replenish: an integer indicating the time until the next replenishment, by which time your (and your opponent's, in part 2) remaining inventory will be reset to the inventory limit.

        The expected output is a single number, indicating the price that you would post for the new buyer.
        '''

        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs
        
        self.rounds_played += 1
        
        # Store customer covariates for next round's opponent tracking
        self.last_customer_covariates = new_buyer_covariates.copy()
        
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)
        current_inventory = inventories[self.this_agent_number]
        opponent_inventory = inventories[self.opponent_number] if self.project_part == 2 else None

        if current_inventory <= 0:
            return self.reject_price
        
        # this is the most expensive operation so we do it only once
        p_star, EV_star = self._get_best_price_and_expected_revenue(new_buyer_covariates)
        
        # Cache p_star toavoids recalculation and save time
        self._last_p_star_cache = p_star

        if self.project_part == 2:
            # Early game: occasionally probe opponent's response to different prices
            # explore in first 20 rounds
            if self.rounds_played < 20 and random.random() < 0.1:
                # Probing to learn opponent behavior
                probe_price = p_star * random.uniform(0.7, 1.3)
                return probe_price
            
            p_star = self._get_competitive_price(new_buyer_covariates, p_star, EV_star, 
                                                  time_until_replenish, opponent_inventory, current_inventory)
            EV_star = p_star * self._get_purchase_probability(new_buyer_covariates, p_star)

        # Pass opponent_inventory to theta computation
        theta = self._compute_theta(current_inventory, time_until_replenish, opponent_inventory)

        # Threshold pricing logic: if EV* >= θ, return p* otherwise just  reject
        if EV_star >= theta:
            return p_star
        else:
            return self.reject_price
        ### currently output is just a deterministic price for the item
        ### but you are expected to use the new_buyer_covariates
        ### combined with models you come up with using the training data 
        ### and history of prices from each team to set a better price for the item
        #return 112.358

