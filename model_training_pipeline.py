#!/usr/bin/env python3
"""
Combined Training Pipeline for:
  - Model A: Residual Life Regression
  - Model B: Chemical Stability Classification
"""
import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, roc_auc_score
)

# === 1. Load Extended Dataset ===
data_path = 'extended_battery_data.csv'
df = pd.read_csv(data_path, parse_dates=['timestamp'])

# Drop any rows missing key targets

# === 2. Shared Feature Engineering ===
# voltage_per_current, temp_delta, internal_resistance already present
# If not, compute them as:
# df['voltage_per_current'] = df['voltage_drop_V'] / (df['current_draw_A'] + 1e-6)
# df['temp_delta'] = df['temp_end_C'] - df['temp_start_C']
# Ensure no duplicates
# Encode chemistry for both models
chem_encoder = LabelEncoder()
df['chemistry_encoded'] = chem_encoder.fit_transform(df['chemistry'])
# Voltage/current ratio and temperature delta already present
# Ensure internal_resistance exists
df['internal_resistance_ohm'] = df['internal_resistance_ohm']

# Encode chemistry for both models
chem_encoder = LabelEncoder()
df['chemistry_encoded'] = chem_encoder.fit_transform(df['chemistry'])

# === 3. Model A: Residual Life Regression ===
# Define features and target
residual_features = [
    'cycle_count', 'voltage_drop_V', 'current_draw_A',
    'impedance_1Hz_ohm', 'impedance_10Hz_ohm', 'impedance_100Hz_ohm',
    'phase_1Hz_deg', 'phase_10Hz_deg',
    'temp_start_C', 'temp_end_C', 'internal_resistance_ohm',
    'total_run_hours', 'voltage_per_current', 'temp_delta',
    'chemistry_encoded'
]
residual_target = 'remaining_cycles'

# Split
df_res = df.dropna(subset=residual_features + [residual_target])
X_res = df_res[residual_features]
y_res = df_res[residual_target]
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Train LightGBM Regressor
res_model = lgb.LGBMRegressor(
    n_estimators=200, learning_rate=0.05, random_state=42
)
res_model.fit(X_train_res, y_train_res)

# Evaluate
y_pred_res = res_model.predict(X_test_res)
mae = mean_absolute_error(y_test_res, y_pred_res)
mse = mean_squared_error(y_test_res, y_pred_res)
r2 = r2_score(y_test_res, y_pred_res)
print("=== Residual Life Model Evaluation ===")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 : {r2:.4f}\n")

# Save Model A artifacts
os.makedirs('model_a_residual_life', exist_ok=True)
joblib.dump(res_model, 'model_a_residual_life/residual_life_model.lgb')
joblib.dump(chem_encoder, 'model_a_residual_life/chemistry_encoder.pkl')

# === 4. Model B: Chemical Stability Classification ===
# Define numeric features and target
dmg_cols = [
    'cycle_count', 'voltage_drop_V', 'current_draw_A',
    'impedance_1Hz_ohm', 'impedance_10Hz_ohm', 'impedance_100Hz_ohm',
    'phase_1Hz_deg', 'phase_10Hz_deg',
    'temp_start_C', 'temp_end_C', 'internal_resistance_ohm',
    'total_run_hours', 'voltage_per_current', 'temp_delta'
]
stability_target = 'stability_status'  # string labels

# Prepare DataFrame
# Encode stability status
target_encoder = LabelEncoder()
df['stability_label'] = target_encoder.fit_transform(df[stability_target])

# Scale numeric features
scaler = StandardScaler()
df[dmg_cols] = scaler.fit_transform(df[dmg_cols])

# Features include scaled numeric + chemistry_encoded
X_stab = df[dmg_cols + ['chemistry_encoded']]
y_stab = df['stability_label']
X_train_stab, X_test_stab, y_train_stab, y_test_stab = train_test_split(
    X_stab, y_stab, test_size=0.2, random_state=42, stratify=y_stab
)

# Train LightGBM Classifier
stab_model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    objective='binary', random_state=42
)
stab_model.fit(
    X_train_stab, y_train_stab,
    eval_set=[(X_test_stab, y_test_stab)], eval_metric='auc')

# Evaluate
y_pred_stab = stab_model.predict(X_test_stab)
y_prob_stab = stab_model.predict_proba(X_test_stab)[:,1]
acc = accuracy_score(y_test_stab, y_pred_stab)
roc = roc_auc_score(y_test_stab, y_prob_stab)
print("=== Chemical Stability Model Evaluation ===")
print(classification_report(y_test_stab, y_pred_stab, target_names=target_encoder.classes_))
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC : {roc:.4f}\n")

# Save Model B artifacts
os.makedirs('model_b_chemical_stability', exist_ok=True)
joblib.dump(stab_model, 'model_b_chemical_stability/stability_classifier.lgb')
joblib.dump(chem_encoder, 'model_b_chemical_stability/chemistry_encoder.pkl')
joblib.dump(target_encoder, 'model_b_chemical_stability/stability_encoder.pkl')
joblib.dump(scaler, 'model_b_chemical_stability/feature_scaler.pkl')

print("All models and artifacts saved.")
