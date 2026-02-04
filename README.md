# 🌫️ Atmospheric Uncertainty Logic
### Hybrid KDD Pipeline: Fuzzy-Apriori & Deep Learning for AQI Crisis Prediction

## 🧪 Research Overview

This project investigates whether augmenting a Deep Learning model (LSTM) with logic-based rules mined via Fuzzy Association Rule Mining can improve the prediction of "Hazardous" Air Quality Index (AQI) spikes.

## 🔬 Methodology

Following the standard scientific method:
1. **Problem Formulation:** Addressing non-linear pollution spikes and sensor uncertainty.
2. **State of the Art:** Comparing pure LSTMs against Knowledge-Informed Neural Networks.
3. **Data Collection:** Using the [Air Quality in India (2015-2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) dataset.
4. **Analysis:** - **Uncertainty Layer:** Fuzzy Membership Functions for $PM_{2.5}$ and $NO_2$.
    - **Mining Layer:** Extracting "Hazardous Profiles" using F-Apriori.
    - **Inference Layer:** Rule-Augmented LSTM.
5. **Synthesis:** Evaluating model sensitivity to rare events.

## 🛠️ Installation & Reproducibility

1. Clone the repo.
2. Place your `kaggle.json` in `~/.kaggle/`.
3. Run `pip install -r requirements.txt`.
4. Execute `python src/download_data.py`.
