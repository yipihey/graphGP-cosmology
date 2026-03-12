# GP Density Reconstruction from Quijote Halo Catalogs

A pilot implementation of the Gaussian process density field reconstruction
pipeline described in the collaboration proposal "GP Field-Level Inference
of the Galaxy Density from Point Distributions using Scalable Vecchia Methods."

## What This Does

Given a set of dark matter halo positions (from the Quijote simulations) with
mass and velocity labels, this script:

1. **Reconstructs the smooth density field** at each halo position using a
   Gaussian process with a Poisson likelihood (the halos are treated as a
   Poisson sampling of the underlying density).

2. **Learns the correlation structure** (the kernel / power spectrum) from
   the data, by optimizing the GP marginal likelihood.

3. **Computes the Hessian** of the density field at each halo position,
   classifying the local cosmic web environment (peak, filament, sheet, void).

4. **Tests whether halo properties depend on local environment** beyond
   their dependence on density alone — probing tidal bias empirically.

## Setup

```bash
# Install dependencies
pip install graphgp jax jaxlib optax matplotlib numpy scipy

# For GPU acceleration (recommended for N > 10^4):
pip install jax[cuda12]
```

## Data

The script expects the Quijote halo data from the
[point-cloud-galaxy-diffusion](https://github.com/smsharma/point-cloud-galaxy-diffusion)
project. Update `DATA_DIR` in the script to point to your local copy:

```python
DATA_DIR = "/Users/tabel/Research/data/quijote_halos_set_diffuser_data"
```

The dataset contains the 5000 heaviest halos per simulation from the Quijote
latin hypercube, in a periodic box of side length 1000 Mpc/h.

## Running

```bash
python gp_density_pipeline.py
```

Output goes to `./output/`:
- `gp_reconstruction_results.npz` — all per-halo quantities
- `gp_density_reconstruction.png` — summary figure

## Pipeline Structure

```
Step 0: Load data          → positions (N,3), mass (N,), velocity (N,3)
Step 1: Build graph        → GraphGP neighbor graph (Vecchia approximation)
Step 2: Set up kernel      → RBF kernel C(r) with trainable variance & scale
Step 3: MAP optimization   → Reconstruct δ(x_i) and learn C(r)
Step 4: Compute Hessian    → Local gradient, Hessian, eigenvalues, web class
Step 5: Correlations       → Do mass/velocity depend on tidal environment?
```

## Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K_NEIGHBORS` | 15 | Conditioning neighbors in Vecchia approximation |
| `N0` | 100 | Points at coarsest exact level |
| `INIT_SCALE` | 30.0 | Initial correlation length (Mpc/h) |
| `N_OPTIM_STEPS` | 200 | Gradient descent steps per field optimization |

Increase `K_NEIGHBORS` for better accuracy (cost scales as k³).
Decrease for faster experimentation.

## References

- Dodge, Frank & Clark (2024): [GraphGP](https://github.com/stanford-ism/graphgp)
- Edenhofer et al. (2022): Iterative Charted Refinement, arXiv:2206.10634
- Banerjee & Abel (2021): kNN-CDF statistics, MNRAS 500, 5479
- Cuesta-Lazaro & Mishra-Sharma (2023): Point cloud diffusion, arXiv:2311.17141
- Katzfuss & Guinness (2021): Vecchia approximations, Stat. Sci. 36, 124
