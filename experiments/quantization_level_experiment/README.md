# Quantization-Level Experiment Materials

This folder contains materials for the Qwen2.5-14B quantization-level follow-up experiment.

The goal is to fix the 14B requested model configuration and compare behavior across different load/quantization settings.

The heatmap notebook computes 8 total `3 x 3` theoretical-limit grids: 4 decoding conditions times 2 naive estimators, with requested load mode on rows and actual load mode on columns.

See `experiment_plan.md` for the current quantization-level heatmap protocol.
Shared Python helpers and paper files remain at the repository root.
