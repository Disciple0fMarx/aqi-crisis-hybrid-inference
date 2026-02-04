"""
MODULE: Data Ingestion Engine
PURPOSE: Automates the retrieval of the 'Air Quality in India' dataset from Kaggle.
METHODOLOGY: 
    - Utilizes the Kaggle API to authenticate and download raw CSV files.
    - Manages local directory creation and ZIP extraction.
    - Supports both kaggle.json and environment variable authentication.
"""

import os
from kaggle.api.kaggle_api_extended import KaggleApi


def download_aqi_data():
    dataset_slug = "rohanrao/air-quality-data-in-india"
    raw_data_path = os.path.join("data", "raw")
    
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception:
        print("\n[!] Auth Error: If you used 'export', ensure it's active in this terminal session.")
        return

    print(f"Fetching {dataset_slug}...")
    api.dataset_download_files(dataset_slug, path=raw_data_path, unzip=True)
    print(f"✓ Data saved to {raw_data_path}")


if __name__ == "__main__":
    download_aqi_data()
