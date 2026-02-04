"""
MODULE: Time-Series Preprocessing & Cleaning
PURPOSE: Transforms raw, irregular city-day pollution data into a continuous time-series.
METHODOLOGY:
    - Chronological Alignment: Converts timestamps and sorts records by city and date.
    - Gap Filling: Implements 'Time-Interpolation' for pollutants (PM2.5, NO2, etc.).
    - Spatial Filtering: Isolates specific metropolitan regions (e.g., Delhi) to 
      maintain environmental consistency for the mining engine.
"""

import pandas as pd
import numpy as np
import os


def clean_air_quality_data(input_filename="city_day.csv"):
    # Define paths using os.path for cross-platform compatibility
    raw_path = os.path.join("data", "raw", input_filename)
    processed_dir = os.path.join("data", "processed")
    output_path = os.path.join(processed_dir, "cleaned_delhi_data.csv")

    if not os.path.exists(raw_path):
        print(f"[ERROR] Raw data not found at {raw_path}. Run download_data.py first.")
        return

    # 1. Load Data
    print(f"Loading {input_filename}...")
    df = pd.read_csv(raw_path)

    # 2. Preprocessing & Filtering
    # We focus on Delhi because it provides the most 'Hazardous' events for our mining engine
    df['Date'] = pd.to_datetime(df['Date'])
    df_delhi = df[df['City'] == 'Delhi'].copy()
    df_delhi = df_delhi.sort_values('Date')

    # 3. Scientific Interpolation
    # Why 'time' interpolation? Because pollution levels at 2 PM are more 
    # likely related to 1 PM than a simple average of the whole day.
    pollutants = ['PM2.5', 'PM10', 'NO2', 'NH3', 'CO', 'SO2', 'O3']
    
    print(f"Interpolating missing values for {len(pollutants)} pollutants...")
    df_delhi = df_delhi.set_index('Date')
    # Time-based interpolation respects the actual time gaps in the index
    df_delhi[pollutants] = df_delhi[pollutants].interpolate(method='time')
    
    # 4. Final Cleanup
    # Remove rows where data is still missing (typically at the very beginning of 2015)
    df_delhi = df_delhi.dropna(subset=['PM2.5', 'AQI'])
    df_delhi = df_delhi.reset_index()

    # 5. Save Results
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    df_delhi.to_csv(output_path, index=False)
    print(f"✓ Cleaned data saved to: {output_path}")
    print(f"✓ Final Record Count: {len(df_delhi)}")


if __name__ == "__main__":
    clean_air_quality_data()
