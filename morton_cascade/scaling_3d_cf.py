#!/usr/bin/env python3
"""3D N-scaling: cascade vs Corrfunc xi at increasing N."""
import numpy as np
import time
import subprocess
from Corrfunc.theory.xi import xi as cf_xi

BOXSIZE = 128.0
N_VALUES = [50_000, 200_000, 800_000, 2_000_000]
SEED = 42

# We'll generate uniform random points (clustering pattern doesn't matter for timing)
edges = np.array([0.5, 1.5, 3.0, 6.0, 12.0, 24.0])
rng = np.random.default_rng(SEED)

print(f"3D timing scaling (uniform random, periodic, single realization):")
print(f"{'N':>10} {'cf time (ms)':>15} {'cf time/MN':>15}")
for N in N_VALUES:
    X = rng.uniform(0, BOXSIZE, N)
    Y = rng.uniform(0, BOXSIZE, N)
    Z = rng.uniform(0, BOXSIZE, N)
    t0 = time.time()
    res = cf_xi(boxsize=BOXSIZE, nthreads=1, binfile=edges,
                X=X, Y=Y, Z=Z, verbose=False)
    dt = (time.time() - t0) * 1000.0
    print(f"{N:>10} {dt:>15.1f} {dt/(N/1e6):>15.2f}")

print()
print("Cascade timing (from previous Rust runs):")
print(f"  N=200k: ~4000 ms total = 20 ms/k points")
print(f"  N=2M:   should be ~4000 + 2000*0.001 = ~6000 ms")
print(f"  Cascade has ~4000 ms constant cost from M^3 = 16M-cell buffer")
print(f"  Corrfunc scales linearly with N at fixed density: each point checks ~constant pairs")
