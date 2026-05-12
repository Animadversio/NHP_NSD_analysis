# Time-Resolved Neural Encoding Dynamics — Analysis Plan

**Date**: Apr 7, 2026
**Goal**: Track how neural encoding properties change across the 450ms response window using ResNet50 layer regression.

## Research Question
Do neurons shift from low-level (early layers) to high-level (late layers) visual encoding over time?
How does predictive accuracy evolve across the response window?

## Analysis Steps

### Step 1: Feature Extraction (one-time)
- Model: ResNet50 (torchvision pretrained)
- Images: 1072 NSD images from `NSD1000_LOC/`
- Layers: `relu` (stem), `layer1`, `layer2`, `layer3`, `layer4`, `avgpool`
- Reduction: `sp_avg` (spatial average per channel — fast, no fitting required per time step)
- Also try `pca200` for denser layers
- Save feature dict to disk

### Step 2: Neural Data Preparation
- Session: JianJian 240629 (456 units, 1072 images)
- Smooth `response_matrix_img` with Gaussian (σ=10ms) along time axis
- Shape: (456 units × 450 timepoints × 1072 images)
- For regression at time t: reshape to (1072 images × 456 units)

### Step 3: Time-Resolved Regression
- Stride: every 5ms (90 time points across -49 to 400ms)
- For each time t:
  - y = response_matrix_img[:, t, :].T  →  (1072, 456)
  - Fit RidgeCV from pre-computed features → test R² per layer
- Record: test R², train R², for each layer × time
- Use pretrained Xtransforms (fit once on features, reuse for all time steps)

### Step 4: Visualization
- Fig A: R²(t) curves per layer — shows when and how strongly each layer predicts
- Fig B: Optimal layer index over time — shows hierarchy progression
- Fig C: Peak R² time per layer — latency of best prediction
- Fig D: Response heatmap (units × time) sorted by preferred layer
- Optional: cross-session consistency check

## Key Questions to Answer
1. Does optimal layer shift from early → late over time (feedforward sweep)?
2. Is there a "late" period (>200ms) where predictions drop / change character (feedback)?
3. Do different monkeys / sessions show consistent dynamics?
