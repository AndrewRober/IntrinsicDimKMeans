# Python & GPU Embedding Benchmark Suite (`IntrinsicDimKMeans/python`)

This directory contains the reference `SIDKMeans` estimator, SymPy symbolic proof suite, and reproducible high-dimensional & pretrained Transformer embedding benchmarks validated on an **NVIDIA GB10 (Grace Blackwell)** GPU.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Run SymPy symbolic proofs (Huygens decomposition, Voronoi -2/pi bias, d_crit collapse)
python verify_proofs.py

# 2. Run automated unit & invariance tests (scale, translation, orthogonal rotation, null K*=1)
python test_all.py

# 3. Reproduce Figures 1-4 & 60-corpus BGE-Base (768D) / BGE-Large (1024D) benchmarks
python run_full_benchmarks.py
```

## Using `SIDKMeans` in Python

```python
from sid_kmeans import SIDKMeans

# Zero-hyperparameter automatic cluster selection (K* in {1..15}) on 1024D embeddings:
model = SIDKMeans(k_max=15, random_state=42).fit(embeddings)

print("Selected K*:", model.k_hat_)
print("Satterthwaite Effective DoF (nu):", model.nu_)
print("Marchenko-Pastur Signal Rank:", model.kept_rank_)
labels = model.labels_
```
