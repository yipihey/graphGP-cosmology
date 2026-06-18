"""Fast, parallel anisotropic clustering via Corrfunc (the standard statistics).

The completion produces cosmology-free catalogues (RA, Dec, z). To MEASURE
clustering at scale (millions of randoms) and to compare against the community's
official BOSS analyses, we use Corrfunc's OpenMP-parallel mock pair counters with
``is_comoving_dist=True``: we convert z -> comoving distance with a chosen
**fiducial** cosmology (astropy) and hand Corrfunc the distances directly, so it
counts (rp, pi) / (s, mu) pairs across all cores. The fiducial cosmology is the
analyst's measurement-time choice (exactly as in real BOSS clustering papers) —
the catalogues themselves remain cosmology-free.

This is far faster than a hand-rolled scipy pair count (e.g. ~0.3 s for 300k on
32 threads) and gives the *standard* projected wp(rp), ξ(rp,π) and ξ(s,μ)
multipoles a referee expects. ``DDtheta_mocks`` (also Corrfunc, OpenMP, with
``link_in_ra``/``link_in_dec`` gridding) remains the tool for the cosmology-free
angular w(θ).
"""

from __future__ import annotations

import numpy as np


def comoving_mpc_h(z, cosmo=None):
    """Comoving distance [Mpc/h] for the fiducial cosmology (default Planck18)."""
    from astropy.cosmology import Planck18
    cosmo = cosmo or Planck18
    return cosmo.comoving_distance(np.asarray(z, float)).value * cosmo.h


def _f8(a):
    return np.ascontiguousarray(a, dtype="f8")


def _ddrppi(autocorr, nthreads, pimax, rp_edges, ra, dec, d, w,
            ra2=None, dec2=None, d2=None, w2=None):
    from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
    kw = dict(weights1=_f8(w), weight_type="pair_product", is_comoving_dist=True)
    if ra2 is not None:
        kw.update(RA2=_f8(ra2), DEC2=_f8(dec2), CZ2=_f8(d2), weights2=_f8(w2))
    # cosmology arg (2nd) is ignored when is_comoving_dist=True; pass a dummy 1
    return DDrppi_mocks(autocorr, 1, nthreads, pimax, rp_edges,
                        _f8(ra), _f8(dec), _f8(d), **kw)


def wp_rp(ra, dec, z, rar, decr, zr, *, rp_edges, pimax=40.0, w=None, wr=None,
          cosmo=None, nthreads=32, precomp_RR=None, return_RR=False):
    """Projected correlation wp(rp) via Landy-Szalay (Corrfunc, parallel).

    Randoms are fixed across many calls, so pass ``precomp_RR`` (the RR struct
    from a previous call, returned with ``return_RR=True``) to skip the dominant
    random-random count. ``w``/``wr`` default to equal weights (the completion
    catalogues are equal-weight)."""
    from Corrfunc.utils import convert_rp_pi_counts_to_wp
    d = comoving_mpc_h(z, cosmo); dr = comoving_mpc_h(zr, cosmo)
    w = np.ones(len(ra)) if w is None else w
    wr = np.ones(len(rar)) if wr is None else wr
    ND, NR = len(ra), len(rar)
    DD = _ddrppi(1, nthreads, pimax, rp_edges, ra, dec, d, w)
    DR = _ddrppi(0, nthreads, pimax, rp_edges, ra, dec, d, w, rar, decr, dr, wr)
    RR = precomp_RR if precomp_RR is not None else _ddrppi(1, nthreads, pimax, rp_edges, rar, decr, dr, wr)
    nrp = len(rp_edges) - 1
    wp = convert_rp_pi_counts_to_wp(ND, ND, NR, NR, DD, DR, DR, RR, nrp, pimax)
    return (wp, RR) if return_RR else wp


def xi_smu_multipoles(ra, dec, z, rar, decr, zr, *, s_edges, nmu=20, w=None, wr=None,
                      cosmo=None, nthreads=32, precomp_RR=None, return_RR=False):
    """Redshift-space ξ(s,μ) -> monopole ξ0 and quadrupole ξ2 (Corrfunc, parallel).

    The standard BAO/RSD statistic. Uses ``DDsmu_mocks`` with ``is_comoving_dist``;
    multipoles via Gauss-Legendre-free midpoint μ integration of the LS ξ(s,μ)."""
    from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
    from Corrfunc.utils import convert_3d_counts_to_cf
    d = comoving_mpc_h(z, cosmo); dr = comoving_mpc_h(zr, cosmo)
    w = np.ones(len(ra)) if w is None else w
    wr = np.ones(len(rar)) if wr is None else wr
    ND, NR = len(ra), len(rar)

    def smu(autocorr, A_ra, A_dec, A_d, A_w, B_ra=None, B_dec=None, B_d=None, B_w=None):
        kw = dict(weights1=_f8(A_w), weight_type="pair_product", is_comoving_dist=True)
        if B_ra is not None:
            kw.update(RA2=_f8(B_ra), DEC2=_f8(B_dec), CZ2=_f8(B_d), weights2=_f8(B_w))
        return DDsmu_mocks(autocorr, 1, nthreads, 1.0, nmu, s_edges,
                           _f8(A_ra), _f8(A_dec), _f8(A_d), **kw)

    DD = smu(1, ra, dec, d, w)
    DR = smu(0, ra, dec, d, w, rar, decr, dr, wr)
    RR = precomp_RR if precomp_RR is not None else smu(1, rar, decr, dr, wr)
    ns = len(s_edges) - 1
    cf = convert_3d_counts_to_cf(ND, ND, NR, NR, DD, DR, DR, RR).reshape(ns, nmu)
    mu = (np.arange(nmu) + 0.5) / nmu
    dmu = 1.0 / nmu
    xi0 = 2.0 * (cf * dmu).sum(axis=1)                       # ∫ L0 dμ (×2 for μ in [0,1] sym)
    L2 = 0.5 * (3 * mu ** 2 - 1)
    xi2 = 2.0 * 5.0 * (cf * L2[None, :] * dmu).sum(axis=1)
    scen = np.sqrt(s_edges[1:] * s_edges[:-1])
    return scen, xi0, xi2, (RR if return_RR else None)


def xi_smu_ell024(ra, dec, z, rar, decr, zr, *, s_edges, nmu=20, w=None, wr=None,
                  cosmo=None, nthreads=32, precomp_RR=None, return_RR=False):
    """Like :func:`xi_smu_multipoles` but also returns the hexadecapole ξ4.

    Returns ``(scen, xi0, xi2, xi4, RR)``. Used for the full multipole drop-in
    comparison (ℓ = 0, 2, 4)."""
    from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
    from Corrfunc.utils import convert_3d_counts_to_cf
    d = comoving_mpc_h(z, cosmo); dr = comoving_mpc_h(zr, cosmo)
    w = np.ones(len(ra)) if w is None else w
    wr = np.ones(len(rar)) if wr is None else wr
    ND, NR = len(ra), len(rar)

    def smu(autocorr, A_ra, A_dec, A_d, A_w, B_ra=None, B_dec=None, B_d=None, B_w=None):
        kw = dict(weights1=_f8(A_w), weight_type="pair_product", is_comoving_dist=True)
        if B_ra is not None:
            kw.update(RA2=_f8(B_ra), DEC2=_f8(B_dec), CZ2=_f8(B_d), weights2=_f8(B_w))
        return DDsmu_mocks(autocorr, 1, nthreads, 1.0, nmu, s_edges,
                           _f8(A_ra), _f8(A_dec), _f8(A_d), **kw)

    DD = smu(1, ra, dec, d, w)
    DR = smu(0, ra, dec, d, w, rar, decr, dr, wr)
    RR = precomp_RR if precomp_RR is not None else smu(1, rar, decr, dr, wr)
    ns = len(s_edges) - 1
    cf = convert_3d_counts_to_cf(ND, ND, NR, NR, DD, DR, DR, RR).reshape(ns, nmu)
    mu = (np.arange(nmu) + 0.5) / nmu
    dmu = 1.0 / nmu
    L2 = 0.5 * (3 * mu ** 2 - 1)
    L4 = (35 * mu ** 4 - 30 * mu ** 2 + 3) / 8.0
    xi0 = 2.0 * (cf * dmu).sum(axis=1)
    xi2 = 2.0 * 5.0 * (cf * L2[None, :] * dmu).sum(axis=1)
    xi4 = 2.0 * 9.0 * (cf * L4[None, :] * dmu).sum(axis=1)
    scen = np.sqrt(s_edges[1:] * s_edges[:-1])
    return scen, xi0, xi2, xi4, (RR if return_RR else None)
