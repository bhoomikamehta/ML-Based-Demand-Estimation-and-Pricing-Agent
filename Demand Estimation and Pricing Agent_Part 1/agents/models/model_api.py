import os
import pickle
import numpy as np
import pandas as pd

#this is the same logic and code that was just generated and run in the notebook file to generate the static_prices_submission.csv

scaler_path = f"../../agents/maria-laia-victoria-bhoomika/scaler.pkl"
model_path  = f"../../agents/maria-laia-victoria-bhoomika/scaler/demand_model.pkl"

with open(scaler_path, "rb") as f:
    loaded_scaler = pickle.load(f)

with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)

features = ["price_item", "Covariate1", "Covariate2", "Covariate3"]

price_grid = np.linspace(0.01, 452.339045127419, 200)  #plan to adjust these after tuning the n-grid number - using the numbers found from the notebook file work
training_feature_columns = ["price_item", "Covariate1", "Covariate2", "Covariate3"]


def best_price_and_expected_rev_for_one_user(
    loaded_scaler,
    loaded_model,
    user_covariate_values_no_price
):

    covariate_df = pd.DataFrame(
        [user_covariate_values_no_price.values] * len(price_grid),
        columns=user_covariate_values_no_price.index
    )
    covariate_df.insert(0, "price_item", price_grid)

    x_raw = covariate_df[training_feature_columns]
    x_scaled = loaded_scaler.transform(x_raw)
    purchase_prob = loaded_model.predict_proba(x_scaled)[:, 1]

    expected_revenue = price_grid * purchase_prob
    idx = int(np.argmax(expected_revenue))
    return float(price_grid[idx]), float(expected_revenue[idx])


def predict_prob(user_covariate_values_no_price, price):
    covariate_df = pd.DataFrame(
        [[price] + list(user_covariate_values_no_price)],
        columns=training_feature_columns
    )
    x_scaled = loaded_scaler.transform(covariate_df)
    return float(loaded_model.predict_proba(x_scaled)[0, 1])

def expected_rev_unconstrained(user_covariate_values_no_price, price):
    prob = predict_prob(user_covariate_values_no_price, price)
    return float(price * prob)

def price_unconstrained(user_covariate_values_no_price):
    best_price, best_ev = best_price_and_expected_rev_for_one_user(
        loaded_scaler=loaded_scaler,
        loaded_model=loaded_model,
        user_covariate_values_no_price=pd.Series(
            user_covariate_values_no_price,
            index=["Covariate1", "Covariate2", "Covariate3"]
        )
    )
    return best_price