import os
import sys
import json
import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import torch.nn as nn

# 1. Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.data_cleaner import clean_air_quality_data
from src.preprocessing.fuzzy_sets import apply_fuzzification

class HybridLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128, 1)
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

def get_aqi_category(value):
    """Standard Indian AQI categories for PM2.5"""
    if value <= 30: return 'Good'
    if value <= 60: return 'Satisfactory'
    if value <= 90: return 'Moderate'
    if value <= 120: return 'Poor'
    if value <= 250: return 'Very Poor'
    return 'Severe'

def run_transfer_test(city_name, lookback=14):
    print(f"\n--- 🌍 EVALUATING MODEL GENERALIZATION: {city_name.upper()} ---")
    
    # 2. Paths
    MODEL_PATH = 'models/hybrid_lstm_v1.pth'
    SCALER_PATH = 'models/scaler.gz'
    RULES_PATH = 'metadata/rules.json'
    
    # 3. Load Artifacts
    scaler = joblib.load(SCALER_PATH)
    with open(RULES_PATH, 'r') as f:
        rules = json.load(f)

    # 4. Process Target City
    clean_air_quality_data(city_name=city_name)
    cleaned_path = f"data/processed/cleaned_{city_name.lower().replace(' ', '_')}_data.csv"
    fuzzified_path = f"data/processed/fuzzified_{city_name.lower().replace(' ', '_')}_data.csv"
    
    apply_fuzzification(cleaned_path, output_path=fuzzified_path)
    city_df = pd.read_csv(fuzzified_path)

    # 5. Knowledge Injection
    for i, rule in enumerate(rules):
        condition_col = rule['rule'].split("IF ")[1].split(" THEN")[0]
        city_df[f'Rule_{i}'] = city_df[condition_col] * rule['confidence'] if condition_col in city_df.columns else 0.0

    feature_cols = ['PM2.5', 'PM2.5_Good', 'PM2.5_Moderate', 'PM2.5_Hazardous', 'Rule_0', 'Rule_1', 'Rule_2', 'Rule_3']
    for col in feature_cols:
        if col not in city_df.columns: city_df[col] = 0.0
    # THE CRITICAL FIX: Interpolate missing sensor values
    # 'linear' interpolation fills gaps, 'ffill'/'bfill' handles the edges
    city_df[feature_cols] = city_df[feature_cols].interpolate(method='linear').ffill().bfill()

    # Final NaN check - if still NaN, drop them
    if city_df[feature_cols].isnull().values.any():
        print(f"⚠️ Warning: Dropping remaining {city_df[feature_cols].isnull().any(axis=1).sum()} rows with NaNs")
        city_df = city_df.dropna(subset=feature_cols)

    data_scaled = scaler.transform(city_df[feature_cols])

    # 6. Prepare Sequences
    X, y_actual_scaled = [], []
    for i in range(len(data_scaled) - lookback):
        X.append(data_scaled[i:i+lookback])
        y_actual_scaled.append(data_scaled[i+lookback, 0])

    X_tensor = torch.FloatTensor(np.array(X))

    # 7. Inference
    model = HybridLSTM(input_size=len(feature_cols))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy()

    # 8. Un-scale
    def unscale(vals):
        dummy = np.zeros((len(vals), len(feature_cols)))
        dummy[:, 0] = vals.flatten()
        return scaler.inverse_transform(dummy)[:, 0]

    y_actual = unscale(np.array(y_actual_scaled))
    y_pred = unscale(y_pred_scaled)

    # 9. PERFORMANCE EVALUATION
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
    corr, _ = pearsonr(y_actual, y_pred)
    
    # Peak Error (How much we miss by during the worst pollution spikes)
    peak_actual = np.max(y_actual)
    peak_pred = y_pred[np.argmax(y_actual)]
    peak_error = abs(peak_actual - peak_pred)

    print(f"\n--- 📈 GLOBAL METRICS ---")
    print(f"MAE:                  {mae:.2f} µg/m³ (Avg deviation)")
    print(f"RMSE:                 {rmse:.2f} µg/m³ (Penalty for large misses)")
    print(f"R² Score:             {r2:.4f} ({'Excellent' if r2 > 0.8 else 'Good' if r2 > 0.5 else 'Poor'})")
    print(f"Pearson r:            {corr:.4f} ({'Strong' if corr > 0.7 else 'Moderate'})")
    print(f"Peak Error:           {peak_error:.2f} µg/m³ (Error at max spike)")

    # 10. CATEGORICAL ANALYSIS (The Hybrid Advantage)
    results_df = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred})
    results_df['Actual_Cat'] = results_df['Actual'].apply(get_aqi_category)
    results_df['Pred_Cat'] = results_df['Predicted'].apply(get_aqi_category)
    
    cat_acc = (results_df['Actual_Cat'] == results_df['Pred_Cat']).mean()
    print(f"Categorical Accuracy: {cat_acc:.2%}")

    bias = np.mean(y_pred - y_actual)
    print(f"Mean Forecast Bias:   {bias:.2f} µg/m³")

    # 11. VISUALIZATION
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 1, 1)
    plt.plot(y_actual[-120:], label='Actual', color='black', alpha=0.6)
    plt.plot(y_pred[-120:], label='Hybrid Prediction', color='crimson', linestyle='--')
    plt.title(f"Performance Analysis: Delhi-Hybrid Model on {city_name}")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.scatter(y_actual, y_pred, alpha=0.3, color='blue')
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Prediction Correlation (Closer to red line is better)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_transfer_test("Bengaluru")
