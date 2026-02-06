"""
MODULE: Fuzzy Inference Preprocessing (Uncertainty Layer)
PURPOSE: Implements the 'Fuzzification' process for raw pollutant concentrations.
METHODOLOGY:
    - Membership Functions: Utilizes Gaussian and Sigmoid curves to represent 
      linguistic categories (Good, Satisfactory, Moderate, Hazardous).
    - Differentiable Logic: Employs smooth, continuous functions to ensure 
      compatibility with Gradient Descent in the subsequent Deep Learning stage.
    - Uncertainty Handling: Addresses the 'boundary effect' where small sensor 
      fluctuations otherwise cause abrupt category changes.
"""

import numpy as np
import pandas as pd


class FuzzyInference:
    def __init__(self):
        # Thresholds (Mean, Sigma) based on NAQI guidelines
        # Format: { 'Pollutant': { 'Good': (mean, sigma), 'Moderate': (mean, sigma), 'Hazardous_Center': (k, L) } }
        self.thresholds = {
            'PM2.5': {'Good': (15, 12), 'Moderate': (75, 30), 'Hazard': (0.05, 150)},
            'PM10':  {'Good': (50, 30), 'Moderate': (175, 75), 'Hazard': (0.02, 350)},
            'NO2':   {'Good': (20, 15), 'Moderate': (100, 40), 'Hazard': (0.03, 180)},
            'CO':    {'Good': (0.5, 0.3), 'Moderate': (5, 3), 'Hazard': (0.5, 12)},
            'SO2':   {'Good': (20, 15), 'Moderate': (100, 50), 'Hazard': (0.02, 200)},
            'O3':    {'Good': (25, 15), 'Moderate': (100, 40), 'Hazard': (0.03, 180)}
        }

    def _gaussian(self, x, mean, sigma):
        return np.exp(-((x - mean)**2) / (2 * sigma**2))

    def _sigmoid(self, x, k, L):
        return 1 / (1 + np.exp(-k * (x - L)))

    def fuzzify(self, value, pollutant):
        if pollutant not in self.thresholds:
            return {}
        
        t = self.thresholds[pollutant]
        return {
            "Good": self._gaussian(value, *t['Good']),
            "Moderate": self._gaussian(value, *t['Moderate']),
            "Hazardous": self._sigmoid(value, *t['Hazard'])
        }


def apply_fuzzification(input_path, output_path):
    df = pd.read_csv(input_path)
    fi = FuzzyInference()
    
    pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
    print(f"Fuzzifying {len(pollutants)} pollutants...")
    
    final_df = df.copy()
    
    for p in pollutants:
        if p in df.columns:
            # Generate memberships for this specific pollutant
            fuzzy_data = df[p].apply(lambda x: fi.fuzzify(x, p))
            fuzzy_cols = pd.json_normalize(fuzzy_data)
            
            # Rename columns: e.g., 'NO2_Good', 'NO2_Moderate'
            fuzzy_cols.columns = [f"{p}_{col}" for col in fuzzy_cols.columns]
            final_df = pd.concat([final_df, fuzzy_cols], axis=1)
    
    final_df.to_csv(output_path, index=False)
    print(f"✓ Full fuzzified data saved to: {output_path}")


if __name__ == "__main__":
    in_file = "data/processed/cleaned_delhi_data.csv"
    out_file = "data/processed/fuzzified_delhi_data.csv"
    apply_fuzzification(in_file, out_file)
