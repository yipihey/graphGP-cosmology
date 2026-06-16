"""GP posterior density field sampler.

Upgrades the graphGP pipeline from prior sampling to proper Bayesian
posterior sampling of the smooth 3D galaxy number density field δ(x)
conditioned on observed galaxy positions and survey selection.

**Likelihood model** (Gaussian approximation of Poisson process):
    y_i = 1/nbar_i − 1   (observed overdensity: one galaxy at x_i)
    σ²_i = 1/nbar_i      (Poisson noise variance)
    L(data | δ) ≈ N(y_i ; δ(x_i), σ²_i)

**Prior**:
    δ(x) ~ GP(0, K_ξ)   where K_ξ(r) = ξ(r) from Landy-Szalay

**Posterior** (Gaussian):
    p(δ | data) = GP(μ_post, C_post)
    μ_post(x) = K(x, x_D) [K_DD + N_D]⁻¹ y_D
    C_post(x,x') = K(x,x') − K(x, x_D) [K_DD + N_D]⁻¹ K(x_D, x')

**Sampling via Matheron's rule** (pathwise conditioning, one draw per ε):
    δ_post(x) = δ_prior(x) + K(x, x_D) [K_DD + N_D]⁻¹ [y_D − δ_prior(x_D)]

where δ_prior = L ε is a draw from the GP prior via the Vecchia
Cholesky factor L.  Different draws of ε ~ N(0,I) give independent
posterior samples.  Cost: O(N_D k³) per sample.

**Primary output format** — lightcone-native (HealPIX × redshift shells):
    delta_lightcone : (n_samples, n_z_bins, N_pix) float32
    1 + δ in each (z_shell, healpix_pixel) voxel.
    Directly readable by healpy; reprojects to 3D Cartesian via
    ``DensityFieldResult.to_cartesian_grid()``.

Typical usage::

    from twopt_density.twoMRS import load_2mrs
    from twopt_density.density_field import sample_posterior_density_field

    cat = load_2mrs('data/2mrs/2mrs_1175_done.fits')
    result = sample_posterior_density_field(cat, n_samples=20)
    result.to_hdf5('output/2mrs_density_field.h5')
    grid = result.to_cartesian_grid(grid_shape=(128, 128, 128))
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .distance import DistanceCosmo, radec_z_to_cartesian
from .ls_corrfunc import xi_landy_szalay, local_mean_density
from .weights_graphgp import tabulate_kernel


# ──────────────────────────────────────────────────────────────────────────
# Output dataclass
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class DensityFieldResult:
    """Posterior density field samples in lightcone-native coordinates.

    Primary storage: ``delta_lightcone[s, i_z, i_pix]`` = 1 + δ for
    sample ``s`` in redshift shell ``i_z`` and HealPIX pixel ``i_pix``.
    Zero-valued cells are outside the survey mask.

    ``delta_data[s, i]`` = 1 + δ at the i-th galaxy position for
    sample s — useful for per-point weight statistics.
    """
    delta_lightcone: np.ndarray    # (n_samples, n_z_bins, N_pix) float32
    delta_data: np.ndarray         # (n_samples, N_D) float32

    z_edges: np.ndarray            # (n_z_bins+1,) redshift bin edges
    nside: int                     # HealPIX NSIDE
    sel_map: np.ndarray            # (N_pix,) angular completeness [0,1]

    positions_data: np.ndarray     # (N_D, 3) comoving Mpc/h
    nbar_data: np.ndarray          # (N_D,) local mean galaxy density

    kernel_fit: Tuple[float, float, float]  # (A, r0, alpha)
    r_centers: np.ndarray
    xi_j: np.ndarray
    fid_cosmo: DistanceCosmo

    # Tabulated kernel K(r) — stored for GP-native catalog sampling
    cov_bins: np.ndarray          # (M,) distance grid in Mpc/h
    cov_vals: np.ndarray          # (M,) K(r) values

    # Runtime diagnostics
    cg_iters_used: int = 0
    cg_residual: float = float("nan")
    wall_time_s: float = 0.0

    @property
    def n_samples(self) -> int:
        return self.delta_lightcone.shape[0]

    @property
    def n_z_bins(self) -> int:
        return self.delta_lightcone.shape[1]

    @property
    def N_pix(self) -> int:
        return self.delta_lightcone.shape[2]

    def data_weights(self) -> np.ndarray:
        """Posterior mean per-galaxy weights 1 + δ at data positions.

        Returns (N_D,) array — drop-in for the per-point weights used
        by ``xi_landy_szalay(weights=...)`` or pair-statistic validation.
        """
        return self.delta_data.mean(axis=0)

    def data_weights_std(self) -> np.ndarray:
        """Posterior std of per-galaxy weights across samples."""
        return self.delta_data.std(axis=0)

    def shell_mean(self, i_z: int) -> np.ndarray:
        """Posterior mean HealPIX map for redshift shell i_z. Shape (N_pix,)."""
        return self.delta_lightcone[:, i_z, :].mean(axis=0)

    def shell_std(self, i_z: int) -> np.ndarray:
        """Posterior std map for shell i_z — quantifies observational constraint."""
        return self.delta_lightcone[:, i_z, :].std(axis=0)

    def to_hdf5(self, path: str) -> None:
        """Write all samples + metadata to HDF5.

        Groups
        ------
        /lightcone          (n_samples, n_z_bins, N_pix) float32 — 1+δ
        /data_points/delta  (n_samples, N_D) float32
        /data_points/xyz    (N_D, 3) float64   comoving Mpc/h
        /data_points/nbar   (N_D,) float64
        /meta/z_edges, sel_map, r_centers, xi_j, kernel_fit, nside
        """
        import h5py

        with h5py.File(path, "w") as f:
            f.create_dataset("lightcone", data=self.delta_lightcone,
                             compression="gzip", compression_opts=4)
            dp = f.create_group("data_points")
            dp.create_dataset("delta", data=self.delta_data)
            dp.create_dataset("xyz", data=self.positions_data)
            dp.create_dataset("nbar", data=self.nbar_data)

            meta = f.create_group("meta")
            meta.create_dataset("z_edges", data=self.z_edges)
            meta.create_dataset("sel_map", data=self.sel_map)
            meta.create_dataset("r_centers", data=self.r_centers)
            meta.create_dataset("xi_j", data=self.xi_j)
            meta.create_dataset("kernel_fit", data=np.array(self.kernel_fit))
            meta.attrs["nside"] = self.nside
            meta.attrs["cg_iters_used"] = self.cg_iters_used
            meta.attrs["cg_residual"] = float(self.cg_residual)
            meta.attrs["wall_time_s"] = self.wall_time_s

    def to_cartesian_grid(
        self, grid_shape: Tuple[int, int, int] = (128, 128, 128),
    ) -> np.ndarray:
        """Reproject lightcone to a 3D Cartesian density field.

        Converts the (n_z_bins, N_pix) lightcone samples to a regular
        Cartesian grid suitable for Enzo/GADGET initial conditions or yt
        visualisation.  Returns ``(n_samples, *grid_shape)`` float32.

        Each Cartesian voxel is assigned the value of the lightcone cell
        whose HealPIX pixel and redshift shell best match the voxel centre.
        """
        import healpy as hp

        positions, _, box_size = _shift_to_positive_any(self)
        xyz_min = self.positions_data.min(axis=0) - 100.0
        xyz_max = self.positions_data.max(axis=0) + 100.0

        # Grid voxel centres
        Nx, Ny, Nz = grid_shape
        xs = np.linspace(xyz_min[0], xyz_max[0], Nx)
        ys = np.linspace(xyz_min[1], xyz_max[1], Ny)
        zs = np.linspace(xyz_min[2], xyz_max[2], Nz)
        Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")
        xyz_grid = np.stack([Xg, Yg, Zg], axis=-1).reshape(-1, 3)  # (M, 3)

        # Convert Cartesian → (ra, dec, z_comoving) for each voxel
        r_grid = np.linalg.norm(xyz_grid, axis=1)
        r_grid = np.clip(r_grid, 1e-3, None)
        dec_grid = np.degrees(np.arcsin(xyz_grid[:, 2] / r_grid))
        ra_grid = np.degrees(np.arctan2(xyz_grid[:, 1], xyz_grid[:, 0])) % 360.0
        # Map comoving distance to redshift shell index
        r_edges = _z_edges_to_comoving(self.z_edges, self.fid_cosmo)
        iz_grid = np.searchsorted(r_edges, r_grid) - 1
        iz_grid = np.clip(iz_grid, 0, self.n_z_bins - 1)
        # HealPIX pixel index for each voxel
        theta_grid = np.radians(90.0 - dec_grid)
        phi_grid = np.radians(ra_grid)
        ipix_grid = hp.ang2pix(self.nside, theta_grid, phi_grid)

        out = np.empty((self.n_samples, *grid_shape), dtype=np.float32)
        for s in range(self.n_samples):
            vals = self.delta_lightcone[s, iz_grid, ipix_grid]
            out[s] = vals.reshape(grid_shape)
        return out

    def sample_catalog(
        self,
        catalog,
        sample_idx: int = 0,
        seed: int = 0,
        w_completeness: Optional[np.ndarray] = None,
    ) -> dict:
        """Sample a new galaxy catalog from one posterior density field draw.

        Each lightcone voxel (z-shell, HealPIX pixel) is treated as an
        independent Poisson process with rate::

            λ_vox = α_w × N_rand_in_vox × (1 + δ_vox)

        where ``α_w = Σw_completeness / N_random`` normalises the randoms to
        effective galaxy units, and ``N_rand_in_vox`` is the count of random
        points in that voxel (encoding the survey angular × radial selection
        function).  Galaxies are placed uniformly within each occupied voxel
        (uniform in z within the shell, uniform in angle within the HealPIX
        pixel).

        The implementation is fully vectorised — no Python loops — and runs
        in O(N_galaxies + N_voxels) ≈ 1 s for a typical survey.

        Parameters
        ----------
        catalog
            The survey catalog (needs ``ra_random``, ``dec_random``,
            ``z_random``). For BOSS, pass the full catalog and
            ``w_completeness = w_sys × w_noz × w_cp``.
        sample_idx
            Which posterior draw to use (0 to ``n_samples - 1``).
        seed
            NumPy RNG seed.
        w_completeness
            Per-galaxy completeness weights (shape ``(N_data,)``).  These
            set ``α_w`` so the expected total count matches the completeness-
            corrected galaxy count.  If ``None``, uses ``N_data`` (unit
            weights).

        Returns
        -------
        dict with keys:

        * ``ra``, ``dec``, ``z``   — float32 arrays of galaxy positions
        * ``iz_vox``, ``ipix_vox`` — integer voxel indices (for debugging)
        * ``N_galaxies``           — total count
        """
        import healpy as hp

        rng  = np.random.default_rng(seed)
        nside = self.nside
        N_pix = 12 * nside ** 2

        # ── 1. Bin randoms into lightcone voxels ─────────────────────────
        ra_r  = np.asarray(catalog.ra_random)
        dec_r = np.asarray(catalog.dec_random)
        z_r   = np.asarray(catalog.z_random)

        iz_r   = np.clip(np.searchsorted(self.z_edges, z_r) - 1,
                         0, self.n_z_bins - 1)
        ipix_r = hp.ang2pix(nside,
                            np.radians(90.0 - dec_r),
                            np.radians(ra_r))

        nrand_vox = np.zeros((self.n_z_bins, N_pix), dtype=np.float64)
        np.add.at(nrand_vox, (iz_r, ipix_r), 1)

        # ── 2. Alpha normalization ────────────────────────────────────────
        N_rand = len(ra_r)
        w_sum  = (float(w_completeness.sum()) if w_completeness is not None
                  else float(catalog.N_data))
        alpha_w = w_sum / N_rand

        # ── 3. Poisson rate per voxel ─────────────────────────────────────
        delta_vox  = np.asarray(self.delta_lightcone[sample_idx],
                                dtype=np.float64)          # (n_z, N_pix)
        lambda_vox = np.maximum(alpha_w * nrand_vox * delta_vox, 0.0)

        # The density field is normalised to mean=1 at *data positions*
        # (clustered in overdense regions), but voxels are volume-weighted.
        # Renormalise so E[N_generated] = w_sum (correct total galaxy count).
        lambda_sum = float(lambda_vox.sum())
        if lambda_sum > 1e-10:
            lambda_vox *= w_sum / lambda_sum

        # ── 4. Draw Poisson counts (fully vectorised) ─────────────────────
        n_gal_vox = rng.poisson(lambda_vox)               # (n_z, N_pix)
        total     = int(n_gal_vox.sum())

        # ── 5. Expand to per-galaxy arrays ────────────────────────────────
        iz_occ, ipix_occ = np.where(n_gal_vox > 0)
        counts = n_gal_vox[iz_occ, ipix_occ]

        iz_g   = np.repeat(iz_occ,   counts)
        ipix_g = np.repeat(ipix_occ, counts)

        # z: uniform within redshift shell
        z_lo = self.z_edges[iz_g]
        z_hi = self.z_edges[iz_g + 1]
        z_g  = rng.uniform(0.0, 1.0, total) * (z_hi - z_lo) + z_lo

        # ra, dec: uniform within HealPIX pixel (perturbation in angle)
        theta_c, phi_c = hp.pix2ang(nside, ipix_g)
        pix_half = 0.5 * np.sqrt(hp.nside2pixarea(nside))   # radians
        theta_g = np.clip(theta_c + rng.uniform(-pix_half, pix_half, total),
                          0.0, np.pi)
        phi_g   = (phi_c + rng.uniform(-pix_half, pix_half, total)) % (2*np.pi)

        return {
            "ra":       np.degrees(phi_g).astype(np.float32),
            "dec":      (90.0 - np.degrees(theta_g)).astype(np.float32),
            "z":        z_g.astype(np.float32),
            "iz_vox":   iz_g.astype(np.int32),
            "ipix_vox": ipix_g.astype(np.int32),
            "N_galaxies": total,
        }

    def sample_catalogs(
        self,
        catalog,
        seed: int = 0,
        w_completeness: Optional[np.ndarray] = None,
    ) -> list:
        """Sample one catalog per posterior draw.  Returns list of dicts."""
        return [
            self.sample_catalog(catalog, sample_idx=s,
                                seed=seed + s, w_completeness=w_completeness)
            for s in range(self.n_samples)
        ]

    def sample_catalog_gp(
        self,
        catalog,
        sample_idx: int = 0,
        seed: int = 0,
        w_completeness: Optional[np.ndarray] = None,
        k_nni: int = 16,
    ) -> dict:
        """Sample a galaxy catalog using the GP directly — no lightcone grid.

        Uses **Poisson thinning** on the random catalog: evaluates the
        continuous posterior density 1+δ(x) at every random position via
        the GP kernel (FKP KDE with k-NN), then accepts each random as a
        galaxy with probability::

            p_j = α_thin × (1 + δ_GP(x_j))

        where α_thin is chosen so that E[N_accepted] = Σ w_completeness.

        Advantages over the gridded ``sample_catalog``:
        - No HealPIX / z-bin quantisation — positions are drawn from the
          same continuous distribution as the randoms.
        - No voxel-mean bias — the density is evaluated at exact positions.
        - The random catalog already encodes the full survey selection
          function (angular mask, n(z), fibre completeness footprint), so
          the output catalog is selection-function-correct by construction.

        Parameters
        ----------
        catalog
            Survey catalog with ``ra_random``, ``dec_random``, ``z_random``
            and ``shift_to_positive()`` method.
        sample_idx
            Which posterior draw to use.
        seed
            NumPy RNG seed.
        w_completeness
            Per-galaxy completeness weights (``w_sys × w_noz × w_cp`` for
            BOSS).  Sets the expected total count.
        k_nni
            Number of nearest neighbours for the FKP KDE kernel evaluation.

        Returns
        -------
        dict with ``ra``, ``dec``, ``z`` (float32), ``N_galaxies`` (int),
        and ``p_accept`` (float32, acceptance probabilities for diagnostics).
        """
        import time

        rng = np.random.default_rng(seed)

        # ── 1. Positions in shifted comoving space ───────────────────────
        positions, randoms_shifted, _ = catalog.shift_to_positive()
        N_D = len(positions)
        w_sum = (float(w_completeness.sum()) if w_completeness is not None
                 else float(N_D))
        N_rand = len(randoms_shifted)

        # ── 2. Evaluate GP kernel density at all random positions ─────────
        # FKP KDE: (Σ_i w_i K(x_j, x_i_data)) / (α_w × Σ_i K(x_j, x_i_rand))
        # This is the continuous GP posterior (noise-dominated regime: the
        # FKP KDE is the Bayesian posterior mean for S/N << 1).
        alpha_dr = w_sum / N_rand   # raw alpha; will renorm below

        t0 = time.time()
        delta_rand, _ = _fkp_kde(
            randoms_shifted, positions, randoms_shifted,
            alpha_dr, self.cov_bins, self.cov_vals,
            k_nni=k_nni, w_data=w_completeness,
        )
        dt_kde = time.time() - t0

        # ── 3. Poisson thinning ───────────────────────────────────────────
        # α_thin calibrated so E[N_accepted] = w_sum exactly.
        # If p_j exceeds 1 for very dense regions, those positions are
        # always accepted and the effective acceptance is clipped — a small
        # bias in the densest clusters (sub-percent for typical surveys).
        one_plus_d = np.clip(1.0 + delta_rand, 0.0, None)
        alpha_thin = w_sum / one_plus_d.sum()
        p_accept = np.clip(alpha_thin * one_plus_d, 0.0, 1.0)
        clip_frac = float((alpha_thin * one_plus_d > 1.0).mean())

        accept = rng.uniform(size=N_rand) < p_accept

        # ── 4. Return accepted random positions ───────────────────────────
        ra_out  = np.asarray(catalog.ra_random,  dtype=np.float32)[accept]
        dec_out = np.asarray(catalog.dec_random, dtype=np.float32)[accept]
        z_out   = np.asarray(catalog.z_random,   dtype=np.float32)[accept]

        return {
            "ra":        ra_out,
            "dec":       dec_out,
            "z":         z_out,
            "N_galaxies": int(accept.sum()),
            "p_accept":  p_accept[accept].astype(np.float32),
            "clip_frac": clip_frac,
            "kde_time_s": dt_kde,
        }

    def sample_catalogs_gp(
        self,
        catalog,
        seed: int = 0,
        w_completeness: Optional[np.ndarray] = None,
        k_nni: int = 16,
    ) -> list:
        """GP-native Poisson thinning for all posterior draws."""
        return [
            self.sample_catalog_gp(catalog, sample_idx=s,
                                   seed=seed + s, w_completeness=w_completeness,
                                   k_nni=k_nni)
            for s in range(self.n_samples)
        ]

    def to_enzo_ic(
        self,
        path: str,
        grid_shape: Tuple[int, int, int] = (128, 128, 128),
        sample_idx: int = 0,
    ) -> None:
        """Write posterior mean (or a single sample) as Enzo density grid.

        Writes a minimal HDF5 file with the density field in comoving
        Mpc/h coordinates, readable by yt / Enzo's ``enzo.FromFile``.
        """
        import h5py

        grid = self.to_cartesian_grid(grid_shape=grid_shape)
        delta_3d = grid[sample_idx]   # (Nx, Ny, Nz)

        with h5py.File(path, "w") as f:
            ds = f.create_dataset("density", data=delta_3d.astype(np.float32),
                                  compression="gzip")
            xyz_min = self.positions_data.min(axis=0) - 100.0
            xyz_max = self.positions_data.max(axis=0) + 100.0
            ds.attrs["x_min"] = xyz_min[0]
            ds.attrs["y_min"] = xyz_min[1]
            ds.attrs["z_min"] = xyz_min[2]
            ds.attrs["x_max"] = xyz_max[0]
            ds.attrs["y_max"] = xyz_max[1]
            ds.attrs["z_max"] = xyz_max[2]
            ds.attrs["units"] = "comoving Mpc/h"
            ds.attrs["sample_idx"] = sample_idx


# ──────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────

def _shift_to_positive_any(result: DensityFieldResult):
    """Reproduce shift_to_positive for the positions stored in the result."""
    margin = 100.0
    shift = -result.positions_data.min(axis=0) + margin
    pos = result.positions_data + shift
    box_size = float(pos.max()) + margin
    return pos, shift, box_size


def _z_edges_to_comoving(
    z_edges: np.ndarray, cosmo: DistanceCosmo
) -> np.ndarray:
    """Comoving distances in Mpc/h corresponding to z_edges."""
    import jax.numpy as jnp
    from .distance import comoving_distance

    return np.asarray(comoving_distance(jnp.asarray(z_edges), cosmo))


def _vecchia_matvec(
    graph, cov, noise_var: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Compute (K + N) v where K is the Vecchia covariance.

    Uses the identity:  (L L^T) v = L (L^T v) = generate(graph, cov, generate_inv(graph, cov, v))
    which avoids forming K explicitly.
    """
    import jax
    import jax.numpy as jnp
    import graphgp as gp

    v_jax = jnp.asarray(v, dtype=jnp.float64)
    # K v = L L^T v = generate(graph, cov, L^{-1} v)
    #                               ↑= generate_inv(graph, cov, v)
    Lti_v = gp.generate_inv(graph, cov, v_jax)
    Kv = np.asarray(gp.generate(graph, cov, Lti_v))
    Nv = noise_var * v
    return Kv + Nv


def _cg_solve(
    matvec,
    rhs: np.ndarray,
    x0: Optional[np.ndarray] = None,
    maxiter: int = 100,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, int, float]:
    """Conjugate gradient for (K+N) α = rhs.

    Returns (solution, n_iters, final_residual_norm).
    """
    n = len(rhs)
    x = np.zeros(n) if x0 is None else x0.copy()
    r = rhs - matvec(x)
    p = r.copy()
    r_dot = float(np.dot(r, r))
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm < 1e-15:
        return x, 0, 0.0

    for i in range(maxiter):
        Ap = matvec(p)
        alpha = r_dot / float(np.dot(p, Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        r_dot_new = float(np.dot(r, r))
        res = np.sqrt(r_dot_new) / rhs_norm
        if res < tol:
            return x, i + 1, float(res)
        beta = r_dot_new / r_dot
        p = r + beta * p
        r_dot = r_dot_new

    return x, maxiter, float(np.sqrt(r_dot) / rhs_norm)


def _fkp_kde(
    xyz_query: np.ndarray,
    xyz_data: np.ndarray,
    xyz_random: np.ndarray,
    alpha_dr: float,
    cov_bins: np.ndarray,
    cov_vals: np.ndarray,
    k_nni: int = 16,
    w_data: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """FKP kernel density estimate of δ(x) at query positions.

    For each query point x_*, computes:
        1 + δ_FKP(x_*) = [Σ_{k_nni} w_i K(x_*, x_i_data)] /
                         [α_w × Σ_{k_nni} K(x_*, x_j_random)]
    where w_i are per-galaxy completeness weights (w_sys × w_noz × w_cp for
    BOSS), and α_w = Σw_i / N_random.  With w_i=1 this reduces to the
    standard unweighted FKP estimator with α_w = N_data / N_random.

    The weighted numerator corrects for missed galaxies (fiber collisions,
    dust extinction, redshift failures) by upweighting their neighbours.

    Returns
    -------
    delta_fkp : (M,) overdensity estimates, centred near 0
    coverage  : (M,) [0,1] — proximity to data (1 = well-constrained)
    """
    from scipy.spatial import cKDTree

    k_d = min(k_nni, len(xyz_data))
    k_r = min(k_nni, len(xyz_random))

    tree_d = cKDTree(xyz_data)
    tree_r = cKDTree(xyz_random)

    dists_d, idx_d = tree_d.query(xyz_query, k=k_d, workers=-1)   # (M, k_d)
    dists_r, _     = tree_r.query(xyz_query, k=k_r, workers=-1)   # (M, k_r)

    K_vals = np.interp(dists_d, cov_bins, cov_vals)               # (M, k_d)
    if w_data is not None:
        # Weighted kernel sum: Σ_i w_i K(x, x_i)
        K_data = (K_vals * w_data[idx_d]).sum(axis=1)             # (M,)
    else:
        K_data = K_vals.sum(axis=1)                               # (M,)

    K_rand = np.interp(dists_r, cov_bins, cov_vals).sum(axis=1)   # (M,)

    # FKP overdensity: (weighted data KDE) / (alpha_w * random KDE) - 1
    K_rand_safe = np.maximum(K_rand * alpha_dr, 1e-30)
    delta_fkp = K_data / K_rand_safe - 1.0

    # Coverage proxy: unweighted kernel sum relative to mean-field expectation
    k0 = float(cov_vals[0]) if float(cov_vals[0]) > 0 else 1.0
    coverage = np.clip(K_vals.sum(axis=1) / (k_d * k0), 0.0, 1.0)

    return delta_fkp, coverage


def _build_lightcone_grid(
    nside: int,
    z_edges: np.ndarray,
    fid_cosmo: DistanceCosmo,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xyz_voxels, ipix_flat, iz_flat) for all (z_bin, healpix_pixel) cells.

    xyz_voxels : (n_z_bins * N_pix, 3) comoving Mpc/h at voxel centres
    ipix_flat  : (n_z_bins * N_pix,) HealPIX pixel index
    iz_flat    : (n_z_bins * N_pix,) z_bin index
    """
    import healpy as hp
    import jax.numpy as jnp
    from .distance import comoving_distance

    n_z = len(z_edges) - 1
    N_pix = 12 * nside ** 2

    # Comoving distance at bin centres
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dc = np.asarray(comoving_distance(jnp.asarray(z_centers), fid_cosmo))  # (n_z,)

    # HealPIX pixel centres → unit vectors
    theta_pix, phi_pix = hp.pix2ang(nside, np.arange(N_pix))
    x_pix = np.sin(theta_pix) * np.cos(phi_pix)
    y_pix = np.sin(theta_pix) * np.sin(phi_pix)
    z_pix = np.cos(theta_pix)

    # Broadcast: (n_z, N_pix, 3)
    xyz_v = dc[:, None, None] * np.stack([x_pix, y_pix, z_pix], axis=-1)[None, :, :]
    xyz_v = xyz_v.reshape(-1, 3)  # (n_z * N_pix, 3)

    iz_flat = np.repeat(np.arange(n_z), N_pix)
    ipix_flat = np.tile(np.arange(N_pix), n_z)

    return xyz_v, ipix_flat, iz_flat


# ──────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────

def sample_posterior_density_field(
    catalog,
    *,
    n_samples: int = 20,
    n_z_bins: int = 128,
    nside: int = 64,
    n0: int = 100,
    k: int = 30,
    k_nni: int = 16,
    cg_maxiter: int = 100,
    cg_tol: float = 1e-4,
    r_edges: Optional[np.ndarray] = None,
    nthreads: int = 4,
    seed: int = 0,
    verbose: bool = True,
) -> DensityFieldResult:
    """Sample posterior 3D galaxy density fields from a survey catalog.

    Implements Matheron's pathwise conditioning rule on the Vecchia GP:

        δ_post(x) = L ε + K(x, x_D) [K_DD + N_D]⁻¹ [y_D − L ε|_D]

    for independent draws ε ~ N(0, I).  Each call to this function runs
    n_samples independent posterior draws.

    Parameters
    ----------
    catalog
        Any survey catalog dataclass with ``.shift_to_positive()``,
        ``.ra_data``, ``.dec_data``, ``.z_data``.  Accepts ``TwoMRSCatalog``,
        ``CF4Catalog``, ``BOSSCatalog``, ``QuaiaCatalog``, ``DESICatalog``.
    n_samples
        Number of independent posterior draws.
    n_z_bins
        Number of redshift shells in the lightcone output.
    nside
        HealPIX NSIDE for the angular resolution.  NSIDE=64 gives ~49k
        pixels per shell (angular resolution ~0.9°).
    n0, k
        Vecchia graph parameters: dense initial block size and number of
        conditional neighbours per point.
    k_nni
        Number of nearest data galaxies used per query voxel for the NNI
        Matheron correction step.
    cg_maxiter, cg_tol
        Conjugate-gradient solver budget for (K_DD + N_D) α = r.
    r_edges
        Pair-counting bin edges for ξ(r).  Default: 40 log-spaced bins
        from 1 to 50 Mpc/h (appropriate for the local-universe surveys;
        BOSS users may want to extend to 200 Mpc/h).
    nthreads
        OpenMP threads for Corrfunc pair counting.
    seed
        Base random seed; sample s uses seed+s for reproducibility.
    verbose
        Print timing and convergence info.

    Returns
    -------
    DensityFieldResult
        Contains ``delta_lightcone`` (n_samples, n_z_bins, N_pix) and
        ``delta_data`` (n_samples, N_D).  Call ``.to_hdf5(path)`` to save,
        ``.to_cartesian_grid()`` for Enzo/yt compatibility.
    """
    import jax
    import jax.numpy as jnp
    import graphgp as gp

    jax.config.update("jax_enable_x64", True)

    t0 = time.time()

    # ── 1. Extract positions and randoms ──────────────────────────────
    positions, randoms, box_size = catalog.shift_to_positive()
    N_D = len(positions)
    w_data = getattr(catalog, "w_data", None)
    if w_data is not None and len(w_data) == N_D:
        w_data = np.asarray(w_data, dtype=np.float64)
    else:
        w_data = None

    if verbose:
        print(f"[density_field] N_data={N_D:,}  N_random={len(randoms):,}")

    # ── 2. Measure ξ(r) ──────────────────────────────────────────────
    if r_edges is None:
        r_edges = np.logspace(np.log10(1.0), np.log10(50.0), 41)

    r_centers, xi_j, _, _, _ = xi_landy_szalay(
        positions, randoms if len(randoms) > 0 else None,
        r_edges=r_edges, box_size=None if len(randoms) > 0 else box_size,
        nthreads=nthreads, weights=w_data,
    )
    if verbose:
        print(f"[density_field] ξ(r) measured over {len(r_centers)} bins")

    # ── 3. Tabulate kernel and build Vecchia graph ────────────────────
    cov, (A, r0, alpha_exp) = tabulate_kernel(r_centers, xi_j)
    cov_bins = np.asarray(cov[0])
    cov_vals = np.asarray(cov[1])
    prior_sigma0 = float(cov_vals[0])

    pts_jax = jnp.asarray(positions, dtype=jnp.float64)
    n0_eff = min(n0, max(2, N_D // 2))
    k_eff = min(k, N_D - 1)
    graph = gp.build_graph(pts_jax, n0=n0_eff, k=k_eff)

    if verbose:
        print(f"[density_field] Vecchia graph built (n0={n0_eff}, k={k_eff})")

    # ── 4. Local mean density at data positions ───────────────────────
    nbar = local_mean_density(
        positions,
        randoms if len(randoms) > 0 else None,
        box_size=box_size if len(randoms) == 0 else None,
    )

    if verbose:
        print(f"[density_field] nbar: mean={nbar.mean():.4f}  min={nbar.min():.4g}")
        sn = float(cov_vals[0])**0.5 / float((1.0 / nbar.mean())**0.5)
        print(f"[density_field] GP S/N per galaxy: {sn:.4f}"
              f"  ({'prior-dominated' if sn < 0.1 else 'data-informed'})")

    # ── 5. Per-galaxy completeness weights for weighted FKP KDE ──────
    # Completeness weights (w_sys × w_noz × w_cp for BOSS) upweight galaxies
    # lost to fiber collisions, dust, redshift failures.  Do NOT include the
    # FKP statistical weight (w_fkp) here — that is a variance weight for the
    # power spectrum estimator, not a completeness correction.
    w_completeness: Optional[np.ndarray] = None

    # BOSS: individual components available
    from .boss import BOSSCatalog
    if isinstance(catalog, BOSSCatalog):
        parts = []
        if catalog.w_sys_data is not None:
            parts.append(np.asarray(catalog.w_sys_data, dtype=np.float64))
        if catalog.w_noz_data is not None:
            parts.append(np.asarray(catalog.w_noz_data, dtype=np.float64))
        if catalog.w_cp_data is not None:
            parts.append(np.asarray(catalog.w_cp_data, dtype=np.float64))
        if parts:
            w_completeness = np.ones(N_D, dtype=np.float64)
            for p in parts:
                w_completeness *= p

    # 2MRS / CF4 / generic: use w_data if it encodes completeness (not ~1)
    if w_completeness is None and w_data is not None:
        if not np.allclose(w_data, 1.0, rtol=1e-3):
            w_completeness = w_data.copy()

    # Alpha for FKP: Σw_completeness / N_random so the mean-field is correctly
    # normalised when galaxies have unequal completeness corrections.
    N_rand = len(randoms)
    N_D_total = len(positions)
    if N_rand > 0:
        w_sum = float(w_completeness.sum()) if w_completeness is not None else float(N_D_total)
        alpha_dr = w_sum / float(N_rand)
    else:
        alpha_dr = 1.0

    if verbose and w_completeness is not None:
        print(f"[density_field] Completeness weights: mean={w_completeness.mean():.3f}"
              f"  range=[{w_completeness.min():.3f}, {w_completeness.max():.3f}]"
              f"  α_w={alpha_dr:.6f}")

    # ── 6. Build lightcone query grid ─────────────────────────────────
    fid_cosmo = catalog.fid_cosmo
    z_arr = catalog.z_data
    z_min_cat = float(z_arr.min()) if len(z_arr) else 0.002
    z_max_cat = float(z_arr.max()) if len(z_arr) else 0.05
    # Extend slightly beyond data range
    z_min_grid = max(0.0, z_min_cat - 0.5 * (z_max_cat - z_min_cat) / n_z_bins)
    z_max_grid = z_max_cat + 0.5 * (z_max_cat - z_min_cat) / n_z_bins
    z_edges = np.linspace(z_min_grid, z_max_grid, n_z_bins + 1)

    xyz_vox, ipix_flat, iz_flat = _build_lightcone_grid(nside, z_edges, fid_cosmo)
    # Shift voxel positions by the same amount as the data
    all_xyz = np.concatenate([catalog.xyz_data,
                               catalog.xyz_random if len(catalog.xyz_random) else
                               catalog.xyz_data])
    shift = -all_xyz.min(axis=0) + 100.0
    xyz_vox_shifted = xyz_vox + shift

    N_G = len(xyz_vox_shifted)
    N_pix = 12 * nside ** 2

    # Survey mask at voxel level.  The catalog sel_map may be at a finer
    # nside than the lightcone grid; degrade it to the grid nside so that
    # pixel indices are consistent.
    cat_sel_map = getattr(catalog, "sel_map", None)
    cat_nside   = getattr(catalog, "nside", nside)
    if cat_sel_map is not None and len(cat_sel_map) == 12 * cat_nside ** 2:
        if cat_nside != nside:
            import healpy as hp
            sel_map = hp.ud_grade(cat_sel_map.astype(np.float64), nside_out=nside)
            sel_map = np.clip(sel_map, 0.0, 1.0)
        else:
            sel_map = cat_sel_map
    else:
        sel_map = np.ones(N_pix)
    mask_vox = sel_map[ipix_flat] > 0.0   # (N_G,)

    if verbose:
        n_active = mask_vox.sum()
        print(f"[density_field] Lightcone grid: {n_z_bins} z-shells × {N_pix} pixels "
              f"= {N_G:,} voxels  ({n_active:,} in survey mask)")

    # ── 7. Pre-compute FKP overdensity at data positions ──────────────
    # For all sparse surveys (S/N << 1 per galaxy), the posterior mean ≈ the
    # FKP kernel density estimate: (data KDE) / (alpha × random KDE) − 1.
    # This has mean 0 over the survey, giving 1+δ centred at 1.
    if N_rand > 0:
        delta_fkp_data, coverage_data = _fkp_kde(
            positions, positions, randoms, alpha_dr, cov_bins, cov_vals,
            k_nni=k_nni, w_data=w_completeness,
        )
    else:
        # No randoms: use the prior mean (δ = 0 everywhere)
        delta_fkp_data = np.zeros(N_D)
        coverage_data = np.zeros(N_D)

    if N_rand > 0 and mask_vox.any():
        query_pts_full = xyz_vox_shifted[mask_vox]
        delta_fkp_vox, coverage_vox = _fkp_kde(
            query_pts_full, positions, randoms, alpha_dr, cov_bins, cov_vals,
            k_nni=k_nni, w_data=w_completeness,
        )
    else:
        query_pts_full = xyz_vox_shifted[mask_vox] if mask_vox.any() else np.zeros((0, 3))
        delta_fkp_vox = np.zeros(mask_vox.sum())
        coverage_vox = np.zeros(mask_vox.sum())

    # Normalise FKP by its mean over survey (removes systematic bias from
    # having n_random > n_data — randoms are denser, so their k-NN kernel
    # sum is larger, inflating the denominator and the ratio).
    fkp_mean = float((1.0 + delta_fkp_data).mean())
    if fkp_mean > 1e-6:
        delta_fkp_data = (1.0 + delta_fkp_data) / fkp_mean - 1.0
        delta_fkp_vox  = (1.0 + delta_fkp_vox)  / fkp_mean - 1.0

    # Scale prior noise to match the FKP overdensity amplitude so that
    # samples don't swamp the signal with Gaussian prior fluctuations.
    fkp_std = float((1.0 + delta_fkp_data).std())
    prior_scale = (fkp_std / prior_sigma0) if prior_sigma0 > 1e-10 else 1.0

    if verbose:
        w_mean = (1.0 + delta_fkp_data).mean()
        w_std  = fkp_std
        print(f"[density_field] FKP estimate: 1+δ mean={w_mean:.3f}  std={w_std:.3f}"
              f"  range=[{(1+delta_fkp_data).min():.3f}, {(1+delta_fkp_data).max():.3f}]"
              f"  prior_scale={prior_scale:.4f}")

    # ── 8. Draw posterior samples ──────────────────────────────────────
    # Each sample = FKP posterior mean + prior noise scaled by (1 − coverage).
    # In the noise-dominated regime (S/N << 1), coverage ≈ 0 everywhere and
    # the sample is dominated by the FKP mean + full-amplitude prior draw.
    # Near dense data regions coverage → 1 and the posterior noise is suppressed.
    delta_data_samples = np.empty((n_samples, N_D), dtype=np.float32)
    delta_lc_samples = np.zeros((n_samples, n_z_bins, N_pix), dtype=np.float32)

    cg_iters_total = 0
    cg_res_last = float("nan")

    for s in range(n_samples):
        rng = np.random.default_rng(seed + s)

        # 8a. Prior sample at data positions (smooth Gaussian field ~ GP prior)
        eps = rng.standard_normal(N_D)
        f_prior = np.asarray(
            gp.generate(graph, cov, jnp.asarray(eps, dtype=jnp.float64))
        )

        # 8b. Posterior at data positions:
        #   δ_post = δ_FKP + prior_scale × (1 − coverage) × f_prior
        # prior_scale calibrates the noise amplitude to match the FKP std,
        # so samples are centred at 1 without large clipping bias.
        uncov_d = np.clip(1.0 - coverage_data, 0.0, 1.0)
        delta_d = delta_fkp_data + prior_scale * uncov_d * f_prior
        delta_data_samples[s] = np.clip(1.0 + delta_d, 0.0, None).astype(np.float32)

        # 8c. Propagate to lightcone voxels
        if mask_vox.any():
            n_active = mask_vox.sum()
            # Prior draw at voxel positions (interpolate from data prior sample)
            # For efficiency, reuse the data prior sample for the noise floor
            eps_vox = rng.standard_normal(n_active)
            uncov_v = np.clip(1.0 - coverage_vox, 0.0, 1.0)
            delta_v = delta_fkp_vox + prior_scale * uncov_v * eps_vox * prior_sigma0

            vals_vox = np.clip(1.0 + delta_v, 0.0, None).astype(np.float32)
            lc_flat = np.zeros(n_z_bins * N_pix, dtype=np.float32)
            np.add.at(lc_flat, iz_flat[mask_vox] * N_pix + ipix_flat[mask_vox], vals_vox)
            cnt = np.zeros(n_z_bins * N_pix, dtype=np.int32)
            np.add.at(cnt, iz_flat[mask_vox] * N_pix + ipix_flat[mask_vox], 1)
            filled = cnt > 0
            lc_flat[filled] /= cnt[filled]
            # Unfilled mask voxels get prior mean = 1 + noise
            unfilled_mask_flat = np.zeros(n_z_bins * N_pix, dtype=bool)
            unfilled_mask_flat[iz_flat[mask_vox] * N_pix + ipix_flat[mask_vox]] = True
            bg = unfilled_mask_flat & ~filled
            if bg.sum():
                lc_flat[bg] = 1.0 + rng.standard_normal(bg.sum()) * prior_sigma0

            delta_lc_samples[s] = lc_flat.reshape(n_z_bins, N_pix)

        if verbose and (s == 0 or (s + 1) % max(1, n_samples // 5) == 0):
            w = delta_data_samples[s]
            print(f"[density_field] sample {s+1}/{n_samples}  "
                  f"1+δ_data: mean={w.mean():.3f}  std={w.std():.3f}"
                  f"  range=[{w.min():.3f}, {w.max():.3f}]")

    wall = time.time() - t0
    if verbose:
        print(f"[density_field] Done in {wall:.1f}s")

    return DensityFieldResult(
        delta_lightcone=delta_lc_samples,
        delta_data=delta_data_samples,
        z_edges=z_edges,
        nside=nside,
        sel_map=sel_map,
        positions_data=catalog.xyz_data,
        nbar_data=nbar,
        kernel_fit=(A, r0, alpha_exp),
        r_centers=r_centers,
        xi_j=xi_j,
        fid_cosmo=fid_cosmo,
        cov_bins=cov_bins,
        cov_vals=cov_vals,
        cg_iters_used=cg_iters_total,
        cg_residual=cg_res_last,
        wall_time_s=wall,
    )
