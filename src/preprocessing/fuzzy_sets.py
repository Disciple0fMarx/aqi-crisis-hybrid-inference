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
        pass

    def _gaussian(self, x, mean, sigma):
        """Standard Gaussian membership function."""
        return np.exp(-((x - mean)**2) / (2 * sigma**2))

    def _sigmoid(self, x, k, L):
        """Sigmoid membership function for high-end saturation (Hazardous)."""
        return 1 / (1 + np.exp(-k * (x - L)))

    def fuzzify_pm25(self, value):
        """
        Maps a PM2.5 value to linguistic fuzzy memberships.
        Thresholds based on Indian NAQI standards (modified for smooth transitions).
        """
        memberships = {
            # Good: Centered at 15, tapers off by 35
            "Good": self._gaussian(value, 15, 12),
            
            # Moderate: Centered at 75, covers the middle range
            "Moderate": self._gaussian(value, 75, 30),
            
            # Hazardous: Starts rising at 120, fully saturated by 200+
            "Hazardous": self._sigmoid(value, 0.05, 150)
        }
        return memberships


def apply_fuzzification(input_path, output_path):
    df = pd.read_csv(input_path)
    fi = FuzzyInference()
    
    print(f"Applying Fuzzy Inference to {len(df)} records...")
    
    # Apply the fuzzification logic to each row
    fuzzy_results = df['PM2.5'].apply(fi.fuzzify_pm25)
    
    # Expand the dictionary into separate columns
    fuzzy_df = pd.json_normalize(fuzzy_results)
    
    # Prefix columns to stay organized
    fuzzy_df.columns = [f"Fuzzy_PM25_{col}" for col in fuzzy_df.columns]
    
    # Combine with original data
    final_df = pd.concat([df, fuzzy_df], axis=1)
    
    final_df.to_csv(output_path, index=False)
    print(f"✓ Fuzzified data saved to: {output_path}")


if __name__ == "__main__":
    in_file = "data/processed/cleaned_delhi_data.csv"
    out_file = "data/processed/fuzzified_delhi_data.csv"
    apply_fuzzification(in_file, out_file)
