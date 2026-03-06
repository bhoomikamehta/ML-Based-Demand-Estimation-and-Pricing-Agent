import random
import pickle
import os
import numpy as np
import pandas as pd


'''
This template serves as a starting point for your agent.
'''


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number

        self.project_part = params['project_part'] 
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
        low_prices = np.linspace(0.01, 37.83, 20)  # Up to 25th percentile as per nb
        mid_prices = np.linspace(37.83, 74.14, 100)  # 25th to 75th percentile as per nb
        high_prices = np.linspace(74.14, 452.339045127419, 40)  #max as per nb

        self.price_grid = np.unique(np.concatenate([low_prices, mid_prices, high_prices]))
        self.price_grid = np.unique(np.append(self.price_grid, self.reject_price))

        self.n_prices = len(self.price_grid)

        self._base_features = np.zeros((self.n_prices, 4))
        self._base_features[:, 0] = self.price_grid

        self.avg_expected_revenue = 63.622046695420046 # as calculated in the python nb
        self.theta_scale = 0.001 #as observed in testing lower base is better so to ground the avg value
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
        did_customer_buy_from_me = (last_sale[0] == self.this_agent_number)

        my_last_prices = last_sale[1][self.this_agent_number]

        my_current_profit = state[self.this_agent_number]
        ### keep track of the inventory levels after the last sale
        self.remaining_inventory = inventories[self.this_agent_number]

        time_until_replenish = time_until_replenish

        pass

    def _get_best_price_and_expected_revenue(self, buyer_covariates):

        # Optimized version Maria's code in the python nb
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

    def _compute_theta(self, inventory, time_until_replenish):
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
            return base_theta * 0.5


        t = remaining_customers
        T = 20.0
        time_factor = t / T

        scarcity_ratio = remaining_customers / inventory
        max_scarcity = 15.0
        normalized_scarcity = min(scarcity_ratio / max_scarcity, 1.0)


        alpha = 0.5
        adjustment_factor = 1.0 + alpha * normalized_scarcity * time_factor

        adjustment_factor = min(adjustment_factor, 2.5)

        return base_theta * adjustment_factor

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
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)


        current_inventory = inventories[self.this_agent_number]

        # If no inventory, reject by returning the high price as initialized in the init fn
        if current_inventory <= 0:
            return self.reject_price
        p_star, EV_star = self._get_best_price_and_expected_revenue(new_buyer_covariates)

        # Compute threshold theta(inventory, time_until_replenish)
        theta = self._compute_theta(current_inventory, time_until_replenish)

        # Threshold pricing logic: if EV* >= θ, return p* otherwise just  reject
        if EV_star >= theta:
            return p_star
        else:
            return self.reject_price

