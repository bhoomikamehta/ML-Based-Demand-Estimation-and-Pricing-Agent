"""
Additional models to test in create_model.ipynb
These models often outperform GradientBoosting and are fast at inference time.

Add this code to your notebook after the Gradient Boosting section.
"""

# Option 1: XGBoost - Often better than GradientBoosting, very fast inference
from xgboost import XGBClassifier

xgb_models = {
    'XGB_n100_lr0.1_d5': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1),
    'XGB_n200_lr0.05_d5': XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1),
    'XGB_n100_lr0.1_d7': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42, n_jobs=-1),
    'XGB_n150_lr0.1_d6': XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1),
}

print("\n" + "="*60)
print("Testing XGBoost models...")
print("="*60)

xgb_results = {}
for name, xgb_model in xgb_models.items():
    scores = cross_val_score(xgb_model, x_scaled, y, cv=5, scoring='roc_auc')
    xgb_results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'model': xgb_model
    }
    print(f"{name:20s} - ROC-AUC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

if xgb_results:
    best_xgb_name = max(xgb_results.keys(), key=lambda k: xgb_results[k]['mean'])
    print(f"\nBest XGBoost: {best_xgb_name} with ROC-AUC: {xgb_results[best_xgb_name]['mean']:.4f}")


# Option 2: LightGBM - Often fastest and best performance
from lightgbm import LGBMClassifier

lgbm_models = {
    'LGBM_n100_lr0.1_d5': LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1, verbose=-1),
    'LGBM_n200_lr0.05_d5': LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, verbose=-1),
    'LGBM_n100_lr0.1_d7': LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42, n_jobs=-1, verbose=-1),
    'LGBM_n150_lr0.1_d6': LGBMClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1, verbose=-1),
}

print("\n" + "="*60)
print("Testing LightGBM models...")
print("="*60)

lgbm_results = {}
for name, lgbm_model in lgbm_models.items():
    scores = cross_val_score(lgbm_model, x_scaled, y, cv=5, scoring='roc_auc')
    lgbm_results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'model': lgbm_model
    }
    print(f"{name:20s} - ROC-AUC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

if lgbm_results:
    best_lgbm_name = max(lgbm_results.keys(), key=lambda k: lgbm_results[k]['mean'])
    print(f"\nBest LightGBM: {best_lgbm_name} with ROC-AUC: {lgbm_results[best_lgbm_name]['mean']:.4f}")


# Option 3: CatBoost - Good for tabular data, handles features well
from catboost import CatBoostClassifier

catboost_models = {
    'CatBoost_n100_lr0.1_d5': CatBoostClassifier(iterations=100, learning_rate=0.1, depth=5, random_state=42, verbose=False),
    'CatBoost_n200_lr0.05_d5': CatBoostClassifier(iterations=200, learning_rate=0.05, depth=5, random_state=42, verbose=False),
    'CatBoost_n100_lr0.1_d7': CatBoostClassifier(iterations=100, learning_rate=0.1, depth=7, random_state=42, verbose=False),
}

print("\n" + "="*60)
print("Testing CatBoost models...")
print("="*60)

catboost_results = {}
for name, cb_model in catboost_models.items():
    scores = cross_val_score(cb_model, x_scaled, y, cv=5, scoring='roc_auc')
    catboost_results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'model': cb_model
    }
    print(f"{name:20s} - ROC-AUC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

if catboost_results:
    best_cb_name = max(catboost_results.keys(), key=lambda k: catboost_results[k]['mean'])
    print(f"\nBest CatBoost: {best_cb_name} with ROC-AUC: {catboost_results[best_cb_name]['mean']:.4f}")


# Option 4: Neural Network (MLPClassifier) - Can capture complex patterns
# NOTE: Slower inference, but might improve accuracy
from sklearn.neural_network import MLPClassifier

mlp_models = {
    'MLP_h100_100': MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500, random_state=42, early_stopping=True),
    'MLP_h200_100': MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=500, random_state=42, early_stopping=True),
    'MLP_h100_50_25': MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=500, random_state=42, early_stopping=True),
}

print("\n" + "="*60)
print("Testing Neural Network (MLP) models...")
print("="*60)

mlp_results = {}
for name, mlp_model in mlp_models.items():
    scores = cross_val_score(mlp_model, x_scaled, y, cv=5, scoring='roc_auc')
    mlp_results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'model': mlp_model
    }
    print(f"{name:20s} - ROC-AUC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

if mlp_results:
    best_mlp_name = max(mlp_results.keys(), key=lambda k: mlp_results[k]['mean'])
    print(f"\nBest MLP: {best_mlp_name} with ROC-AUC: {mlp_results[best_mlp_name]['mean']:.4f}")


# Update the final comparison to include all new models
print("\n" + "="*60)
print("UPDATED FINAL MODEL COMPARISON")
print("="*60)

all_results_updated = {}
all_results_updated.update(lr_results)
all_results_updated.update(rf_results)
all_results_updated.update(gb_results)
all_results_updated.update(xgb_results)
all_results_updated.update(lgbm_results)
all_results_updated.update(catboost_results)
all_results_updated.update(mlp_results)

sorted_results_updated = sorted(all_results_updated.items(), key=lambda x: x[1]['mean'], reverse=True)

print("\nAll models ranked by ROC-AUC:")
print("-" * 60)
for i, (name, result) in enumerate(sorted_results_updated, 1):
    print(f"{i:2d}. {name:25s} - ROC-AUC: {result['mean']:.4f} (+/- {result['std'] * 2:.4f})")

best_model_name_updated, best_model_result_updated = sorted_results_updated[0]
print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model_name_updated}")
print(f"ROC-AUC Score: {best_model_result_updated['mean']:.4f} (+/- {best_model_result_updated['std'] * 2:.4f})")
print(f"{'='*60}")

# Train the best model on full dataset
print(f"\nTraining {best_model_name_updated} on full training dataset...")
best_model_updated = best_model_result_updated['model']
best_model_updated.fit(x_scaled, y)
print("Training complete!")

# Save the new best model
with open("../../agents/maria-laia-victoria-bhoomika/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("../../agents/maria-laia-victoria-bhoomika/demand_model.pkl", "wb") as f:
    pickle.dump(best_model_updated, f)

print(f"Saved best model ({best_model_name_updated}) to demand_model.pkl")


