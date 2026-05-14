"""
NHP NSD Analysis — core library
================================
Reusable data loaders, regression pipelines, and analysis utilities.

Submodules
----------
core.nsd_n3       — NSD_N3 session loader (LOC array, all monkeys)
core.tripleN      — Triple-N dataset loader + area extraction (V1/V2/V4/IT)
core.features     — DINOv2 / ResNet50 feature extraction & PCA
core.regression   — Time-resolved RidgeCV regression pipeline
core.clustering   — Pooled neuron clustering (feature build, PCA→UMAP→k-means)
core.weight_dyn      — Regression weight dynamics (cosine sim, subspace angles, LDS)
core.piecewise_lds   — Piecewise LDS + PELT changepoint detection for weight trajectories
"""
