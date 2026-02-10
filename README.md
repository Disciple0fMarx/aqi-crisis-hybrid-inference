# Atmospheric Uncertainty Logic: Hybrid AQI Inference

An adaptive AI framework designed to predict PM2.5 crises by merging **Fuzzy Logic (Knowledge Engineering)** with **Long Short-Term Memory (LSTM)** neural networks. This project explores the "Delhi Lens"—evaluating how expert rules mined from one of the world's most polluted cities generalize to diverse urban topographies across India.



---

## The Hybrid Architecture
Unlike "black-box" models, this system uses a **Knowledge-Augmented** approach to handle atmospheric uncertainty:

1.  **Fuzzy Preprocessing:** Raw sensor data is transformed into linguistic states ($Good, Moderate, Hazardous$) using Gaussian and Sigmoid membership functions to normalize environmental variance.
2.  **Apriori Knowledge Mining:** Frequent patterns in Delhi's atmosphere were mined to create a Knowledge Base (e.g., *IF NO2 is Moderate AND SO2 is High THEN PM2.5 is Hazardous*).
3.  **Knowledge Injection:** These rules are mathematically "activated" (Membership $\times$ Confidence) and injected as features into the LSTM, providing a logical anchor.
4.  **The NaN Fortress:** A specialized preprocessing pipeline implemented to handle real-world "sensor silence" through linear interpolation and temporal padding ($ffill/bfill$).



---

## Cross-City Generalization (Zero-Shot)
The model was trained exclusively on **Delhi** data and tested across diverse Indian clusters without retraining.

| City | Topography | Pearson $r$ | $R^2$ Score | Cat. Accuracy | Bias ($\mu g/m^3$) | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Kolkata** | Industrial | **0.93** | **0.85** | 70.1% | -4.36 | **Elite Fit** |
| **Mumbai** | Coastal | 0.87 | 0.66 | 76.1% | -4.64 | **Strong** |
| **Bengaluru** | Plateau | 0.62 | 0.24 | 71.0% | -1.38 | **Robust** |
| **Vizag** | Hilly/Coast | 0.78 | 0.60 | 62.2% | +0.83 | **Volatile** |
| **Aizawl** | Clean/Hills | 0.62 | 0.08 | **91.8%** | +5.18 | **Stable** |



---

## Technical Details & Key Findings

### 1. The "Delhi Lens" Bias
In coastal regions (Mumbai/Kolkata), the model shows a systematic **negative bias**. This suggests the "Delhi Brain" underestimates peaks in humid environments where aerosol mass behavior differs from the dry Indo-Gangetic Plain.

### 2. The Accuracy Paradox (Aizawl)
In clean-air zones, the $R^2$ score collapses due to low variance, yet **Categorical Accuracy** remains above 90%. This proves the Fuzzy Logic layers act as a safety net, preventing the model from predicting non-existent crises.

### 3. Topographical Traps
The massive **Peak Error (185.28)** in Bengaluru and **83.19** in Vizag highlight that localized "super-spikes" (likely construction dust or topographical trapping) cannot be fully captured by inter-pollutant chemical rules alone.


---

## Usage
1. **Prepare Environment:** Ensure `scikit-learn`, `torch`, and `joblib` are installed.
2. **Run Generalization Test:**
   ```bash
   python src/models/transfer_test.py
   ```
3. **Change Target City:** Modify the `run_transfer_test("CityName")` call in `__main__`.

---

## Conclusion

The success of this project lies in its resilience. By forcing a Deep Learning model to respect Fuzzy Logic constraints, we created a system that doesn't just predict numbers, but understands the 'Health Risk' categories of a city it has never seen before.
