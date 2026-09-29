# A Self-Supervised Framework for False Data Injection (FDI) Attack Detection in Smart Grids

This repository contains the complete experimental pipeline and benchmark framework for detecting cyber-physical FDI attacks across multi-feeder power distribution telemetry using a hybrid 1D-CNN + BiLSTM architecture with dual-pretext self-supervised learning.

## Key Empirical Findings
- **Data Scarcity (10%–50% Labels):** Self-supervised pretraining provides critical inductive regularization, preventing majority-class collapse and achieving a +32.61% F1 advantage at 25% labels over supervised baselines.
- **Data Abundance (75%–100% Labels):** Scratch models outperform fine-tuned models due to cross-frequency domain shift between 30-minute residential profiles and sub-second 50 Hz feeder telemetry.
- **Physical Attribution (SHAP):** Line current ($I$) drives >64% of attack classifications, confirming physical alignment ($I = P/V$).

## Project Structure
- `notebooks/`: Full 21-phase experimental notebooks (pretraining, fine-tuning, ablation, XAI, generalization).
- `src/`: Reusable PyTorch encoders, channel projectors, and classifiers.
- `results/`: Metric tables, confusion matrices, and SHAP attribution plots.