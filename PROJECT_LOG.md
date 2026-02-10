# Research Log: Atmospheric Uncertainty Logic

## Phase 1: Problem Identification (Done)
- Research Problem: Predicting AQI spikes using Hybrid Logic-DL.
- Dataset: Air Quality in India (2015-2020).

## Phase 2: Data & Uncertainty (Done)
- [x] Repository initialized with branching strategy.
- [x] Data ingestion script finalized.
- [x] Time-series interpolation logic.
- [x] Gaussian/Sigmoid membership function design.

## Phase 3: Knowledge Discovery (Done)
- [x] Implemented Fuzzy-Support and T-norm (min) logic.
- [x] Mined inter-pollutant rules for Delhi dataset.
- [x] Exported 'Knowledge Base' to `metadata/rules.json`.

## Phase 4: The Hybrid Model (Next)
- [x] Implement Rule-Activation logic for Knowledge Injection.
- [x] Optimized LSTM architecture with Knowledge-Augmented features.
- [x] Conducted Cross-City Generalization tests (Kolkata, Mumbai, Vizag, Aizawl).

## Phase 5: Evaluation & Performance Analytics (Next)
- [x] Generate Comparative Error Matrices (MAE vs. RMSE vs. Bias) across topographies.
- [x] Analyze "Delhi Lens" bias in coastal vs. landlocked cities.
- [x] Conduct Residual Analysis to identify systematic under-prediction in high-humidity zones.
