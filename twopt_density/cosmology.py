"""JAX port of Bartlett+ symbolic-regression cosmology emulators.

Two parallel paths:

* **syren-halofit** (Bartlett et al. 2024, ``which='halofit'``)
    - ΛCDM only; parameterised by sigma8.
    - Eisenstein-Hu zero-baryon linear pivot + Bartlett ``logF`` correction +
      halofit + Bartlett ``A`` correction.

* **syren-new** (Bartlett et al. 2025, ``which='new'``)
    - Extended cosmology (mnu, w0, wa); parameterised by As.
    - Eisenstein-Hu no-wiggle linear pivot + approximate D(z, k) +
      same ``logF`` + ``S`` correction + ``R`` growth correction +
      analytic non-linear correction (no halofit machinery).

Ported function-by-function from the upstream ``symbolic-pofk`` package
(``symbolic_pofk.linear``, ``.linear_new``, ``.syrenhalofit``,
``.syren_new``). Public reference values come from those modules; tests
verify ~1e-10 agreement.

All functions are pure JAX arithmetic -- ``jax.grad`` works wrt every
cosmological parameter for free.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


# ----- Eisenstein & Hu zero-baryon linear pivot ------------------------

def _eh_zb_unnorm(k, Om, Ob, h, ns):
    """EH98 zero-baryon transfer-squared * k^ns (no normalisation)."""
    ombom0 = Ob / Om
    om0h2 = Om * h ** 2
    ombh2 = Ob * h ** 2
    theta2p7 = 2.7255 / 2.7
    s = 44.5 * jnp.log(9.83 / om0h2) / jnp.sqrt(1.0 + 10.0 * ombh2 ** 0.75)
    alphaGamma = (
        1.0 - 0.328 * jnp.log(431.0 * om0h2) * ombom0
        + 0.38 * jnp.log(22.3 * om0h2) * ombom0 ** 2
    )
    Gamma = Om * h * (
        alphaGamma + (1.0 - alphaGamma) / (1.0 + (0.43 * k * h * s) ** 4)
    )
    q = k * theta2p7 ** 2 / Gamma
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    L0 = jnp.log(2.0 * jnp.exp(1.0) + 1.8 * q)
    tk_eh = L0 / (L0 + C0 * q ** 2)
    return tk_eh ** 2 * k ** ns


def _sigma8_norm_factor(Om, Ob, h, ns, n_grid=1000):
    """Anorm such that sigma_8 of EH-ZB equals sigma8 (= 1).

    Returns the ratio (sigma8_target = 1) / sigma8_unnorm, squared.
    """
    log_k = jnp.linspace(jnp.log(1e-7), jnp.log(1e5), n_grid)
    k = jnp.exp(log_k)
    R = 8.0
    x = k * R
    W = jnp.where(
        x < 1e-3,
        1.0,
        3.0 / (x ** 3 + 1e-300) * (jnp.sin(x) - x * jnp.cos(x)),
    )
    pk_un = _eh_zb_unnorm(k, Om, Ob, h, ns)
    integrand = pk_un * W ** 2 * k ** 3
    sigma2_un = jnp.trapezoid(integrand, x=jnp.log(x))
    sigma_un = jnp.sqrt(sigma2_un / (2.0 * jnp.pi ** 2))
    return 1.0 / sigma_un ** 2


def pk_EisensteinHu_zb(k, sigma8, Om, Ob, h, ns):
    """EH98 zero-baryon P(k) at z=0 with integral sigma8 normalisation."""
    A1 = _sigma8_norm_factor(Om, Ob, h, ns)
    Anorm = sigma8 ** 2 * A1
    return Anorm * _eh_zb_unnorm(k, Om, Ob, h, ns)


# ----- Bartlett+ linear correction logF (shared by halofit + new) ------

_LOGF_B = jnp.array([
    0.05448654, 0.00379, 0.0396711937097927, 0.127733431568858, 1.35,
    4.053543862744234, 0.0008084539054750851, 1.8852431049189666,
    0.11418372931475675, 3.798, 14.909, 5.56, 15.8274343004709,
    0.0230755621512691, 0.86531976, 0.8425442636372944, 4.553956000000005,
    5.116999999999995, 70.0234239999998, 0.01107, 5.35, 6.421, 134.309,
    5.324, 21.532, 4.741999999999985, 16.68722499999999, 3.078, 16.987,
    0.05881491, 0.0006864690561825617, 195.498, 0.0038454457516892,
    0.276696018851544, 7.385, 12.3960625361899, 0.0134114370723638,
])


def logF_fiducial(k, Om, Ob, h):
    """Bartlett 2023 fiducial logF correction. Extrapolates outside k=[9e-3, 9]."""
    b = _LOGF_B
    line1 = b[0] * h - b[1]
    line2 = (
        ((Ob * b[2]) / jnp.sqrt(h ** 2 + b[3])) ** (b[4] * Om)
        * (
            (b[5] * k - Ob) / jnp.sqrt(b[6] + (Ob - b[7] * k) ** 2)
            * b[8] * (b[9] * k) ** (-b[10] * k)
            * jnp.cos(Om * b[11] - (b[12] * k) / jnp.sqrt(b[13] + Ob ** 2))
            - b[14] * ((b[15] * k) / jnp.sqrt(1 + b[16] * k ** 2) - Om)
            * jnp.cos(b[17] * h / jnp.sqrt(1 + b[18] * k ** 2))
        )
    )
    line3 = (
        b[19] * (b[20] * Om + b[21] * h - jnp.log(b[22] * k)
                 + (b[23] * k) ** (-b[24] * k))
        * jnp.cos(b[25] / jnp.sqrt(1 + b[26] * k ** 2))
    )
    line4 = (
        (b[27] * k) ** (-b[28] * k)
        * (b[29] * k - (b[30] * jnp.log(b[31] * k))
           / jnp.sqrt(b[32] + (Om - b[33] * h) ** 2))
        * jnp.cos(Om * b[34] - (b[35] * k) / jnp.sqrt(Ob ** 2 + b[36]))
    )
    return line1 + line2 + line3 + line4


# ----- syren-halofit (LCDM, sigma8 parameterised) ---------------------

def plin_emulated(k, sigma8, Om, Ob, h, ns, a=1.0):
    """syren-halofit linear P(k) at scale-factor a (no growth correction
    here; growth applied externally via a^2 * D(z)^2 if needed; identity at a=1)."""
    p_eh = pk_EisensteinHu_zb(k, sigma8, Om, Ob, h, ns)
    p_lin = p_eh * jnp.exp(logF_fiducial(k, Om, Ob, h))
    return p_lin


def ksigma_emulated(sigma8, Om, Ob, h, ns, a):
    c = jnp.array([
        4.35761588, 0.83576576, 0.43023897, 20.107738, 0.259285, 0.573205,
        1.680897, 20.043272, 0.425699, 0.39078063,
    ])
    out = (
        c[0] * (a * c[1] * (c[2] - sigma8)
                + (c[3] * a) ** (-c[4] * a - c[5] * ns)
                * (c[6] * Ob + (c[7] * Om) ** (-c[8] * h)))
        / (sigma8 * (a + c[9] * ns))
    )
    return jnp.exp(out)


def neff_emulated(sigma8, Om, Ob, h, ns, a):
    t = jnp.array([
        1.65139294, 4.88150280, 0.512499, 0.148848,
        15.6499400, 0.239307, 0.134631,
    ])
    return (
        (t[0] * ns - t[1])
        * (t[2] * Ob - t[3] * h
           + (t[4] * a) ** (-t[5] * Om - t[6] * sigma8))
    )


def C_emulated(sigma8, Om, Ob, h, ns, a):
    b = jnp.array([
        0.335853, 1.42946178682748, 0.115256188211481, 0.057211, 48.072159,
        0.194058, 1.176006, 1.015136, 0.235398, 0.359587, 2.389843,
        0.356875, 0.443138,
    ])
    return (
        b[0] * sigma8
        - b[1] * jnp.sqrt(b[2] * ns + sigma8 * (b[3] * h
                                                + (b[4] * Om) ** (b[5] * a)
                                                - b[6]))
        * (b[7] * Ob + b[8] * a + b[9] * sigma8 - (b[10] * h) ** (b[11] * Om))
        - b[12]
    )


def A_emulated(k, sigma8, Om, Ob, h, ns, a):
    ksigma = ksigma_emulated(sigma8, Om, Ob, h, ns, a)
    neff = neff_emulated(sigma8, Om, Ob, h, ns, a)
    C = C_emulated(sigma8, Om, Ob, h, ns, a)
    y = k / ksigma
    d = jnp.array([
        0.0, 0.2011, 1.2983, 16.8733, 3.6428, 1.0622, 0.1023, 2.2204,
        0.0105, 0.487, 0.6151, 0.3377, 3.315, 3.9819, 1.3572, 3.3259,
        0.3872, 4.1175, 2.6795, 5.3394, 0.0338,
    ])
    A = (
        d[0] - d[1] / jnp.sqrt(1 + (d[2] * y) ** (-d[3] * C))
        * (
            y - d[4] * (y - d[5] * ns)
            / jnp.sqrt((y - d[6] * jnp.log(d[7] * C)) ** 2 + d[8])
            + d[9] * neff / jnp.sqrt(d[10] + sigma8 ** 2)
            / jnp.sqrt((d[11] * y - jnp.cos(d[12] * neff)) ** 2 + 1)
            + (d[13] + d[14] * neff - d[15] * C - d[16] * y)
            * (d[17] * neff + d[18] * y + jnp.cos(d[19] * neff))
            / jnp.sqrt(y ** 2 + d[20])
        )
    )
    return A


_HALOFIT_PARS_BARTLETT = jnp.array([
    1.5358,  2.8533,  2.3692,  0.9916,  0.2244,  0.5862, -0.565,  0.5871,
    0.5757, -1.505,   0.3913,  2.0252,  0.7971,  0.5989,  0.2216, -0.001,
    1.1771,  5.2082,  3.7324, -0.0158, -0.0972,  0.155,   6.1043,  1.3408,
    -0.2138, -5.325,   1.9967, -0.7176,  0.3108,  1.2477,  0.4018, -0.3837,
])


def run_halofit(k, sigma8, Om, Ob, h, ns, a=1.0, add_correction=True):
    """syren-halofit nonlinear P(k). Bartlett 2024 model is the default."""
    pars = _HALOFIT_PARS_BARTLETT
    ksigma = ksigma_emulated(sigma8, Om, Ob, h, ns, a)
    neff = neff_emulated(sigma8, Om, Ob, h, ns, a)
    C = C_emulated(sigma8, Om, Ob, h, ns, a)
    y = k / ksigma
    plin = plin_emulated(k, sigma8, Om, Ob, h, ns, a=a)

    an = (pars[0] + pars[1] * neff + pars[2] * neff ** 2 + pars[3] * neff ** 3
          + pars[4] * neff ** 4 - pars[5] * C)
    an = 10.0 ** an
    bn = pars[6] + pars[7] * neff + pars[8] * neff ** 2 + pars[9] * C
    bn = 10.0 ** bn
    cn = pars[10] + pars[11] * neff + pars[12] * neff ** 2 + pars[13] * C
    cn = 10.0 ** cn
    gamma = pars[14] + pars[15] * neff + pars[16] * C
    nu = 10.0 ** (pars[17] + pars[18] * neff)
    Omz = Om / a ** 3 / (Om / a ** 3 + 1.0 - Om)
    f1 = Omz ** pars[19]
    f2 = Omz ** pars[20]
    f3 = Omz ** pars[21]

    alpha = jnp.abs(pars[22] + pars[23] * neff
                    + pars[24] * neff ** 2 + pars[25] * C)
    beta = (pars[26] + pars[27] * neff + pars[28] * neff ** 2
            + pars[29] * neff ** 3 + pars[30] * neff ** 4 + pars[31] * C)

    deltaH2 = an * y ** (3 * f1) / (
        1 + bn * y ** f2 + (cn * f3 * y) ** (3 - gamma)
    )
    deltaH2 = deltaH2 / (1 + nu / y ** 2)
    ph = deltaH2 * (2 * jnp.pi ** 2) / k ** 3

    deltaL2 = k ** 3 * plin / (2 * jnp.pi ** 2)
    pq = (
        plin * (1 + deltaL2) ** beta
        / (1 + alpha * deltaL2)
        * jnp.exp(-y / 4 - y ** 2 / 8)
    )

    p_nl = ph + pq
    if add_correction:
        p_nl = p_nl * (1 + A_emulated(k, sigma8, Om, Ob, h, ns, a))
    return p_nl


# ----- syren-new (extended cosmology, As parameterised) ---------------

def _get_eisensteinhu_nw(k, As, Om, Ob, h, ns):
    """EH no-wiggle P(k) at z=0, As-parameterised. (linear_new helper)"""
    ombom0 = Ob / Om
    om0h2 = Om * h ** 2
    ombh2 = Ob * h ** 2
    theta2p7 = 2.7255 / 2.7
    s = 44.5 * jnp.log(9.83 / om0h2) / jnp.sqrt(1.0 + 10.0 * ombh2 ** 0.75)
    alphaGamma = (
        1.0 - 0.328 * jnp.log(431.0 * om0h2) * ombom0
        + 0.38 * jnp.log(22.3 * om0h2) * ombom0 ** 2
    )
    Gamma = Om * h * (
        alphaGamma + (1.0 - alphaGamma) / (1.0 + (0.43 * k * h * s) ** 4)
    )
    q = k * theta2p7 ** 2 / Gamma
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    L0 = jnp.log(2.0 * jnp.exp(1.0) + 1.8 * q)
    tk_eh = L0 / (L0 + C0 * q ** 2)
    kpivot = 0.05
    return (
        2 * jnp.pi ** 2 / k ** 3
        * (As * 1e-9) * (k * h / kpivot) ** (ns - 1)
        * (2 * k ** 2 * 2998 ** 2 / 5 / Om) ** 2
        * tk_eh ** 2
    )


def _get_approximate_D(k, Om, Ob, h, mnu, w0, wa, a):
    """Bond-Lahav-Carroll-EH growth factor with massive neutrinos + (w0, wa)."""
    mnu = mnu + 1e-10
    z = 1.0 / a - 1.0
    theta2p7 = 2.7255 / 2.7
    zeq = 2.5e4 * Om * h ** 2 / theta2p7 ** 4
    Omega = Om * a ** (-3)
    OL = (1 - Om) * a ** (-3 * (1 + w0 + wa)) * jnp.exp(-3 * wa * (1 - a))
    g = jnp.sqrt(Omega + OL)
    Omega = Omega / g ** 2
    OL = OL / g ** 2
    D1 = (
        (1 + zeq) / (1 + z) * 5 * Omega / 2
        / (Omega ** (4 / 7) - OL + (1 + Omega / 2) * (1 + OL / 70))
    )
    Onu = mnu / 93.14 / h ** 2
    Oc = Om - Ob - Onu
    fc = Oc / Om
    fb = Ob / Om
    fnu = Onu / Om
    fcb = fc + fb
    pcb = 1.0 / 4.0 * (5.0 - jnp.sqrt(1.0 + 24.0 * fcb))
    Nnu = 3.0  # mnu>0 always after the +1e-10 above
    q = k * h * theta2p7 ** 2 / (Om * h ** 2)
    yfs = 17.2 * fnu * (1 + 0.488 / fnu ** (7.0 / 6.0)) * (Nnu * q / fnu) ** 2
    Dcbnu = (
        (fcb ** (0.7 / pcb) + (D1 / (1 + yfs)) ** 0.7) ** (pcb / 0.7)
        * D1 ** (1 - pcb)
    )
    return Dcbnu / (1 + zeq)


def _growth_correction_R(Om, w0, wa, a):
    d = jnp.array([
        0.8545, 0.394, 0.7294, 0.5347, 0.4662, 4.6669,
        0.4136, 1.4769, 0.5959, 0.4553, 0.0799, 5.8311,
        5.8014, 6.7085, 0.3445, 1.2498, 0.3756, 0.2136,
    ])
    part1 = d[0]
    den1 = a * d[1] + d[2] + (Om * d[3] - a * d[4]) * jnp.log(-d[5] * w0 - d[6] * wa)
    part2 = -1.0 / den1
    num2 = Om * d[7] - a * d[8] + jnp.log(-d[9] * w0 - d[10] * wa)
    den2 = (
        -a * d[11] + d[12]
        + d[13] * (Om * d[14] + a * d[15] - 1)
        * (d[16] * w0 + d[17] * wa + 1)
    )
    part3 = -num2 / den2
    return 1 + (1 - a) * (part1 + part2 + part3)


def _log10_S(k, Om, Ob, h, mnu, w0, wa):
    e = jnp.array([
        0.2841, 0.1679, 0.0534, 0.0024, 0.1183, 0.3971,
        0.0985, 0.0009, 0.1258, 0.2476, 0.1841, 0.0316,
        0.1385, 0.2825, 0.8098, 0.019, 0.1376, 0.3733,
    ])
    part1 = -e[0] * h
    part2 = -e[1] * w0
    part3 = -e[2] * mnu / jnp.sqrt(e[3] + k ** 2)
    part4 = -(e[4] * h) / (e[5] * h + mnu)
    part5 = e[6] * mnu / (h * jnp.sqrt(e[7] + (Om * e[8] + k) ** 2))
    num = (
        e[9] * Ob - e[10] * w0 - e[11] * wa
        + (e[12] * w0 + e[13]) / (e[14] * wa + w0)
    )
    den = jnp.sqrt(e[15] + (Om + e[16] * jnp.log(-e[17] * w0)) ** 2)
    part6 = num / den
    return (part1 + part2 + part3 + part4 + part5 + part6) / 10


def plin_new_emulated(k, As, Om, Ob, h, ns, mnu, w0, wa, a=1.0):
    """syren-new linear P(k, z) for extended cosmology."""
    eh = _get_eisensteinhu_nw(k, As, Om, Ob, h, ns)
    D = _get_approximate_D(k, Om, Ob, h, mnu, w0, wa, a)
    F = jnp.exp(logF_fiducial(k, Om, Ob, h))
    R = _growth_correction_R(Om, w0, wa, a)
    S = jnp.power(10.0, _log10_S(k, Om, Ob, h, mnu, w0, wa))
    return eh * D ** 2 * F * R * S


_PNL_NEW_G = jnp.array([
    0.2107, 0.0035, 0.0667, 0.0442, 1.2809, 0.2287, 0.1122, 4.3318, 1.1857,
    3.3117, 14.2829, 0.9039, 0.0749, 0.0741, 0.1277, 27.6818, 24.8736,
    0.6264, 0.3035, 0.6069, 0.7882, 0.4811, 1.4326, 1.8971, 0.0271, 0.9635,
    0.0264, 22.9213, 71.1658, 0.0371, 0.0099, 210.3925, 0.2555,
])


def _pnl_bias(k):
    h_ = jnp.array([
        0.5787, 2.3485, 27.3829, 16.4236, 97.3766, 90.9764,
        11.2046, 2447.2, 11376.93,
    ])
    term1 = ((h_[1] * k) - jnp.cos(h_[3] * jnp.cos(h_[2] * k))) * jnp.cos(h_[4] * k)
    den = -h_[7] * jnp.log(h_[6] * k) + (h_[8] * k)
    return ((h_[0] + term1 + jnp.cos(h_[5] * k))) / den


def pnl_new_emulated(k, As, Om, Ob, h, ns, mnu, w0, wa, a=1.0):
    """syren-new nonlinear P(k, z) for extended cosmology."""
    g = _PNL_NEW_G
    P_lin = jnp.log10(plin_new_emulated(k, As, Om, Ob, h, ns, mnu, w0, wa, a))

    term1 = P_lin
    num1 = g[0] * k * (g[1] * k) ** (g[2] * Om - g[3] * As)
    den1a = (g[4] * k ** (-g[5]) - g[6] * P_lin) ** (
        g[7] * P_lin + g[8] * wa + g[9] * w0 - g[10]
    )
    den1b = (g[11] * k ** g[12] + g[13] * P_lin - g[14] * Om) ** (
        g[15] * a - g[16] * ns
    )
    term2 = num1 / (den1a + den1b)

    num2 = (g[17] * a - g[18] * P_lin + g[19]) * k
    den2 = (
        g[20] * Om + g[21] * k + g[22] * ns - g[23]
        + (g[24] * P_lin + g[25] * k ** g[26]) ** (g[27] * a - g[28] * ns)
    )
    term3 = num2 / den2

    term4 = g[29] * k
    term5 = (g[30] * k) ** ((g[31] * k) ** (-a * g[32]))

    pk_nl = term1 + term2 + term3 - term4 - term5 - _pnl_bias(k)
    return jnp.power(10.0, pk_nl)


# ----- top-level convenience ------------------------------------------

def pnl(
    k,
    *,
    which: str = "halofit",
    Om: float = 0.3, Ob: float = 0.05, h: float = 0.7, ns: float = 0.96,
    sigma8: float | None = None, As: float | None = None,
    mnu: float = 0.0, w0: float = -1.0, wa: float = 0.0,
    a: float = 1.0,
) -> jnp.ndarray:
    """Top-level switch between syren-halofit and syren-new.

    ``which='halofit'`` uses ``sigma8``; ``which='new'`` uses ``As``.
    """
    if which == "halofit":
        if sigma8 is None:
            raise ValueError("syren-halofit requires sigma8")
        return run_halofit(k, sigma8, Om, Ob, h, ns, a=a)
    if which == "new":
        if As is None:
            raise ValueError("syren-new requires As")
        return pnl_new_emulated(k, As, Om, Ob, h, ns, mnu, w0, wa, a=a)
    raise ValueError(f"unknown which={which!r}")
