"""Tests for the JAX port of syren-halofit and syren-new.

Reference values come from the upstream ``symbolic-pofk`` package; we
require ~1e-5 relative agreement on P(k) values across a typical k
range. Gradients are checked against centered finite differences.
"""

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
spkg = pytest.importorskip("symbolic_pofk")

from symbolic_pofk import linear as sp_lin
from symbolic_pofk import linear_new as sp_linn
from symbolic_pofk import syrenhalofit as sp_sh
from symbolic_pofk import syren_new as sp_sn

from twopt_density import cosmology as cj


@pytest.fixture
def k_grid():
    return np.logspace(-2, 1, 60)


@pytest.fixture
def lcdm():
    return dict(sigma8=0.8, Om=0.31, Ob=0.049, h=0.68, ns=0.965, a=1.0)


@pytest.fixture
def extended():
    return dict(
        As=2.1, Om=0.31, Ob=0.049, h=0.68, ns=0.965,
        mnu=0.06, w0=-1.0, wa=0.0, a=1.0,
    )


def test_pk_eh_zero_baryon_matches_numpy(k_grid, lcdm):
    """EH no-wiggle backbone with sigma8 normalisation matches scipy.simpson to 1e-5."""
    p_ref = sp_lin.pk_EisensteinHu_zb(
        k_grid, integral_norm=True, **{k: lcdm[k] for k in ("sigma8", "Om", "Ob", "h", "ns")},
    )
    p_jax = np.asarray(cj.pk_EisensteinHu_zb(jnp.asarray(k_grid),
        lcdm["sigma8"], lcdm["Om"], lcdm["Ob"], lcdm["h"], lcdm["ns"],
    ))
    rel = np.max(np.abs(p_ref - p_jax) / np.abs(p_ref))
    assert rel < 1e-5, f"EH-ZB max rel err {rel:.3e}"


def test_plin_emulated_matches_numpy(k_grid, lcdm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_ref = sp_lin.plin_emulated(k_grid, extrapolate=True, **lcdm)
    p_jax = np.asarray(cj.plin_emulated(jnp.asarray(k_grid), **lcdm))
    rel = np.max(np.abs(p_ref - p_jax) / np.abs(p_ref))
    assert rel < 1e-5, f"P_lin max rel err {rel:.3e}"


def test_run_halofit_matches_numpy(k_grid, lcdm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_ref = sp_sh.run_halofit(k_grid, extrapolate=True, **lcdm)
    p_jax = np.asarray(cj.run_halofit(jnp.asarray(k_grid), **lcdm))
    rel = np.max(np.abs(p_ref - p_jax) / np.abs(p_ref))
    assert rel < 1e-5, f"halofit max rel err {rel:.3e}"


def test_plin_new_matches_numpy(k_grid, extended):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_ref = sp_linn.plin_new_emulated(k_grid, **extended)
    p_jax = np.asarray(cj.plin_new_emulated(jnp.asarray(k_grid), **extended))
    rel = np.max(np.abs(p_ref - p_jax) / np.abs(p_ref))
    assert rel < 1e-5, f"plin_new max rel err {rel:.3e}"


def test_pnl_new_matches_numpy(k_grid, extended):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_ref = sp_sn.pnl_new_emulated(k_grid, **extended)
    p_jax = np.asarray(cj.pnl_new_emulated(jnp.asarray(k_grid), **extended))
    rel = np.max(np.abs(p_ref - p_jax) / np.abs(p_ref))
    assert rel < 1e-5, f"pnl_new max rel err {rel:.3e}"


def test_run_halofit_grad_matches_finite_difference(k_grid, lcdm):
    k = jnp.asarray(k_grid)
    fixed = {key: lcdm[key] for key in ("sigma8", "Ob", "h", "ns", "a")}

    def loss(Om):
        return jnp.sum(jnp.log(cj.run_halofit(k, Om=Om, **fixed)))

    g = float(jax.grad(loss)(lcdm["Om"]))
    eps = 1e-4
    fd = (float(loss(lcdm["Om"] + eps)) - float(loss(lcdm["Om"] - eps))) / (2 * eps)
    rel = abs(g - fd) / (abs(fd) + 1e-12)
    assert rel < 1e-4, f"d/dOm rel diff {rel:.3e} (jax={g:.4f}, fd={fd:.4f})"


def test_pnl_new_grad_w0_finite_difference(k_grid, extended):
    k = jnp.asarray(k_grid)
    fixed = {key: extended[key] for key in ("As", "Om", "Ob", "h", "ns", "mnu", "wa", "a")}

    def loss(w0):
        return jnp.sum(jnp.log(cj.pnl_new_emulated(k, w0=w0, **fixed)))

    g = float(jax.grad(loss)(extended["w0"]))
    eps = 1e-4
    fd = (float(loss(extended["w0"] + eps)) - float(loss(extended["w0"] - eps))) / (2 * eps)
    rel = abs(g - fd) / (abs(fd) + 1e-12)
    assert rel < 1e-4, f"d/dw0 rel diff {rel:.3e}"


def test_pnl_top_level_dispatches(k_grid, lcdm, extended):
    k = jnp.asarray(k_grid)
    p_h = np.asarray(cj.pnl(k, which="halofit", **lcdm))
    p_n = np.asarray(cj.pnl(k, which="new", **extended))
    assert p_h.shape == k.shape and np.all(np.isfinite(p_h))
    assert p_n.shape == k.shape and np.all(np.isfinite(p_n))
    with pytest.raises(ValueError):
        cj.pnl(k, which="halofit", As=2.0, Om=0.3, Ob=0.05, h=0.7, ns=0.96)
    with pytest.raises(ValueError):
        cj.pnl(k, which="new", sigma8=0.8, Om=0.3, Ob=0.05, h=0.7, ns=0.96,
               mnu=0.0, w0=-1.0, wa=0.0)
