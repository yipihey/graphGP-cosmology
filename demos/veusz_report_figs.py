"""Build the report's figures as browser-editable Veusz ``.vsz`` embeds.

Each function takes the cached presentation data (the ``D``/``Dm``/``Dc`` dicts
already computed by ``build_completion_presentation``), writes a self-contained
``.vsz`` into ``figs_dir`` via :mod:`tools.veusz_vsz`, and returns the
``<veusz-figure>`` HTML tag. The browser renders/edits them via the hosted WASM
embed — the reader can adjust ranges, colours, markers and re-export.

Conventions (per the request): every RA axis is **wrapped** to (-180,180] and
drawn RA-increasing-leftwards so the CMASS-South cap is contiguous and centred
near 0; inpainted / augmented galaxies are drawn at **alpha 0.7** so overlaps
are visible.
"""

from __future__ import annotations

import os
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools import veusz_vsz as V
from tools.veusz_vsz import Series, Panel, wrap_ra

# observed / completed / z-fail / neutral palette (matches the matplotlib report)
C_OBS = "#e8853a"; C_NEW = "#3a6ea8"; C_ZF = "#7b3ff2"; C_NEUTRAL = "#888888"
RA_LABEL = "RA [deg]  (wrapped, centred near 0)"


def _sub(n, cap=14000, seed=1):
    """Index array subsampling ``n`` points to at most ``cap`` (stable seed)."""
    if n <= cap:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, cap, replace=False)


def _steps(edges, counts):
    """Step-plot (x,y) from bin edges + per-bin counts (for histograms-as-lines)."""
    edges = np.asarray(edges, float); counts = np.asarray(counts, float)
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(counts, 2)
    return x, y


def _ra_range(*ras):
    r = np.concatenate([np.asarray(a, float) for a in ras])
    lo, hi = np.nanpercentile(r, 0.5), np.nanpercentile(r, 99.5)
    pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad)


# ---------------------------------------------------------------------------
# the figures
# ---------------------------------------------------------------------------
def footprint(D, figs_dir):
    ra = wrap_ra(D["sky_ra"]); dec = np.asarray(D["sky_dec"])
    j = _sub(len(ra)); ra, dec = ra[j], dec[j]
    p_sky = Panel([Series(ra, dec, color=C_NEUTRAL, marker="dot", size="1.5pt", alpha=0.5)],
                  xlabel=RA_LABEL, ylabel="Dec [deg]", title="CMASS-South footprint",
                  xrange=_ra_range(ra), invert_x=True)
    zb = np.asarray(D["z_all"]); hb = np.linspace(zb.min(), zb.max(), 41)
    h, _ = np.histogram(zb, hb); hx, hy = _steps(hb, h)
    p_nz = Panel([Series(hx, hy, color=C_OBS, line_only=True)],
                 xlabel="spectroscopic redshift z", ylabel="galaxies / bin", title="redshift distribution n(z)")
    path = os.path.join(figs_dir, "footprint.vsz")
    V.grid(path, [p_sky, p_nz], rows=1, cols=2, width="30cm", height="12cm")
    return V.embed_tag("figs/footprint.vsz", width=900, height=380)


def missing(D, figs_dir):
    # zoom box matching the matplotlib fig (RA 12-22, Dec -3..3) -> wrapped (same, <180)
    ra = wrap_ra(D["sky_ra"]); dec = np.asarray(D["sky_dec"])
    box = (ra > 12) & (ra < 22) & (dec > -3) & (dec < 3)
    tra = wrap_ra(D["tgt_ra"]); tdec = np.asarray(D["tgt_dec"]); kind = np.asarray(D["tgt_kind"])
    tb = (tra > 12) & (tra < 22) & (tdec > -3) & (tdec < 3)
    cc = tb & (kind == "collided"); zz = tb & (kind == "zfail")
    s = [Series(ra[box], dec[box], color=C_NEUTRAL, marker="dot", size="2pt", alpha=0.5, label="observed (spec-z)"),
         Series(tra[cc], tdec[cc], color=C_NEW, marker="cross", size="4pt", alpha=0.7, label="missing: fiber-collided"),
         Series(tra[zz], tdec[zz], color=C_ZF, marker="plus", size="4pt", alpha=0.7, label="missing: redshift-failure")]
    p = Panel(s, xlabel=RA_LABEL, ylabel="Dec [deg]", title="observed + missing targets (zoom)",
              invert_x=True)
    path = os.path.join(figs_dir, "missing.vsz")
    V.scatter(path, p, width="22cm", height="13cm")
    return V.embed_tag("figs/missing.vsz", width=820, height=480)


def wtheta(D, figs_dir):
    tc = np.asarray(D["wt_tc"]); wd = np.asarray(D["wt_data"])
    ens = np.asarray(D["wt_ens_data"]); m = ens.mean(0); sd = ens.std(0)
    s = [Series(tc, wd, color="#000000", line_only=True, label="weighted observed"),
         Series(tc, m, yerr=sd, color=C_NEW, marker="circle", size="3pt", alpha=0.85,
                label="equal-weight completed (mean +/- realization sigma)")]
    p = Panel(s, xlabel="theta [deg]", ylabel="w(theta)", title="angular two-point function",
              xlog=True, ylog=True)
    path = os.path.join(figs_dir, "wtheta.vsz")
    V.scatter(path, p, width="20cm", height="14cm")
    return V.embed_tag("figs/wtheta.vsz", width=760, height=520)


def xi2d(D, figs_dir):
    grid = np.asarray(D["xi2d_c"]).T          # (nz, ntheta): rows=dz (y), cols=dtheta (x)
    tcen = np.asarray(D["k2d_tcen"]); zcen = np.asarray(D["k2d_zcen"])
    path = os.path.join(figs_dir, "xi2d.vsz")
    V.image(path, grid, xrange=(float(tcen[0]), float(tcen[-1])),
            yrange=(float(zcen[0]), float(zcen[-1])),
            xlabel="Delta theta [deg]", ylabel="Delta z", title="completed xi(Delta theta, Delta z)",
            colormap="viridis")
    return V.embed_tag("figs/xi2d.vsz", width=720, height=560)


def mask(Dm, figs_dir):
    ra = wrap_ra(Dm["sky_ra"]); dec = np.asarray(Dm["sky_dec"])
    j = _sub(len(ra)); ra, dec = ra[j], dec[j]
    hra = wrap_ra(Dm["hole_ra"]); hdec = np.asarray(Dm["hole_dec"])
    p_sky = Panel([Series(ra, dec, color=C_NEUTRAL, marker="dot", size="1.5pt", alpha=0.4),
                   Series(hra, hdec, color="#c0392b", marker="circle", size="4pt", alpha=0.9,
                          label="interior mask holes")],
                  xlabel=RA_LABEL, ylabel="Dec [deg]",
                  title=f"{int(Dm['n_holes'])} interior mask holes", invert_x=True)
    rad = np.asarray(Dm["hole_rad"]) * 60.0
    hb = np.linspace(0, 30, 31); h, _ = np.histogram(rad, hb); hx, hy = _steps(hb, h)
    p_h = Panel([Series(hx, hy, color="#c0392b", line_only=True)],
                xlabel="hole radius [arcmin]", ylabel="number of holes",
                title=f"masked interior area {float(Dm['hole_area_tot']):.1f} deg^2", ylog=False)
    path = os.path.join(figs_dir, "mask.vsz")
    V.grid(path, [p_sky, p_h], rows=1, cols=2, width="30cm", height="12cm")
    return V.embed_tag("figs/mask.vsz", width=900, height=380)


def inpaint_gallery(Dm, figs_dir):
    """The showcase: many holes, before (observed) | after (observed+inpainted)."""
    n = int(Dm["n_gallery"])
    g_ra, g_dec, g_hid = np.asarray(Dm["gal_ra"]), np.asarray(Dm["gal_dec"]), np.asarray(Dm["gal_hid"])
    i_ra, i_dec, i_hid = np.asarray(Dm["inp_ra"]), np.asarray(Dm["inp_dec"]), np.asarray(Dm["inp_hid"])
    cra, cdec = np.asarray(Dm["gcen_ra"]), np.asarray(Dm["gcen_dec"])
    rad = np.asarray(Dm["grad"]); box = np.asarray(Dm["gbox"])
    reason = Dm["greason"] if "greason" in Dm else np.array(["hole"] * n, dtype=object)
    panels = []
    for k in range(n):
        go = g_hid == k; io = i_hid == k
        cosd = np.cos(np.radians(cdec[k]))
        xr = (cra[k] - box[k] / cosd, cra[k] + box[k] / cosd)
        yr = (cdec[k] - box[k], cdec[k] + box[k])
        why = str(reason[k])
        # note: equal_aspect collapses panels inside a grid (zero plot area) — omit it
        before = Panel([Series(g_ra[go], g_dec[go], color=C_NEUTRAL, marker="dot", size="3pt", alpha=0.7)],
                       xlabel="", ylabel="Dec", title=f"{why} - observed",
                       xrange=xr, yrange=yr, invert_x=True)
        after = Panel([Series(g_ra[go], g_dec[go], color=C_NEUTRAL, marker="dot", size="3pt", alpha=0.7),
                       Series(i_ra[io], i_dec[io], color=C_NEW, marker="circle", size="3.5pt", alpha=0.7)],
                      xlabel="", ylabel="", title=f"{why} - inpainted",
                      xrange=xr, yrange=yr, invert_x=True)
        panels += [before, after]
    path = os.path.join(figs_dir, "inpaint_gallery.vsz")
    V.grid(path, panels, rows=n, cols=2, width="24cm", height=f"{max(10, 7*n)}cm")
    return V.embed_tag("figs/inpaint_gallery.vsz", width=820, height=min(2400, 240 * n))


def inpaint_closure(Dm, figs_dir):
    tc = np.asarray(Dm["wt_tc"]); wm = np.asarray(Dm["wt_masked"]); wi = np.asarray(Dm["wt_inp"])
    p1 = Panel([Series(tc, wm, color=C_OBS, line_only=True, line_style="dashed", label="masked + masked randoms"),
                Series(tc, wi, color=C_NEW, marker="circle", size="3pt", alpha=0.85, label="inpainted + hole-filled randoms")],
               xlabel="theta [deg]", ylabel="w(theta)", title="clustering closure", xlog=True, ylog=True)
    p2 = Panel([Series(tc, wi / wm, color="#333333", marker="circle", size="3pt", line=True)],
               xlabel="theta [deg]", ylabel="inpainted / masked", title="closure ratio",
               xlog=True, yrange=(0.85, 1.15))
    path = os.path.join(figs_dir, "inpaint_closure.vsz")
    V.grid(path, [p1, p2], rows=1, cols=2, width="28cm", height="12cm")
    return V.embed_tag("figs/inpaint_closure.vsz", width=900, height=400)


def coupling(Dc, figs_dir):
    s = []
    for kind, col, lab in [("zfail", C_ZF, "redshift failures"), ("collided", C_OBS, "fiber collisions")]:
        s.append(Series(np.asarray(Dc[f"{kind}_dc"]), np.asarray(Dc[f"{kind}_S"]),
                        yerr=np.asarray(Dc[f"{kind}_Se"]), color=col, marker="circle", size="4pt",
                        line=True, label=f"{lab}: h={float(Dc[f'{kind}_h']):+.2f} (z={float(Dc[f'{kind}_z']):+.1f})"))
    p1 = Panel(s, xlabel="local success overdensity delta", ylabel="redshift-success fraction S(delta)",
               title="density coupling of selection")
    tc = np.asarray(Dc["sp_tc"])
    p2 = Panel([Series(tc, np.asarray(Dc["sp_wgt"]), color="#000000", line_only=True, label="weighted baseline"),
                Series(tc, np.asarray(Dc["sp_real"]), color=C_NEW, marker="circle", size="3pt", alpha=0.85, label="completion (real positions)"),
                Series(tc, np.asarray(Dc["sp_blind"]), color="#c0392b", marker="square", size="3pt", alpha=0.85,
                       line=True, line_style="dashed", label="density-blind null")],
               xlabel="theta [deg]", ylabel="w(theta)", title="spurious large-scale power (MegaZ test)",
               xlog=True, ylog=True)
    path = os.path.join(figs_dir, "coupling.vsz")
    V.grid(path, [p1, p2], rows=1, cols=2, width="30cm", height="13cm")
    return V.embed_tag("figs/coupling.vsz", width=900, height=420)


def trust(Dc, figs_dir):
    dt = np.asarray(Dc["amp_dt"]); dc = np.asarray(Dc["amp_dc"])
    lim = (0.0, float(max(np.percentile(dt, 99), np.percentile(dc, 99))))
    p1 = Panel([Series(dt, dc, color=C_NEW, marker="dot", size="2pt", alpha=0.5),
                Series(list(lim), list(lim), color="#c0392b", line_only=True, line_style="dashed")],
               xlabel="total-target density (selection-immune)", ylabel="completed catalog density",
               title=f"amplitude anchor: corr = {float(Dc['amp_corr']):.2f}", xrange=lim, yrange=lim)
    tra = wrap_ra(Dc["trust_ra"]); tdec = np.asarray(Dc["trust_dec"]); cv = np.asarray(Dc["trust_cv"])
    j = _sub(len(tra)); tra, tdec, cv = tra[j], tdec[j], cv[j]
    p2 = Panel([Series(tra, tdec, cdata=cv, colormap="magma", marker="dot", size="3pt", alpha=0.9)],
               xlabel=RA_LABEL, ylabel="Dec [deg]",
               title=f"trustworthiness map (median {float(Dc['trust_med']):.2f})", invert_x=True)
    path = os.path.join(figs_dir, "trust.vsz")
    V.grid(path, [p1, p2], rows=1, cols=2, width="30cm", height="13cm")
    return V.embed_tag("figs/trust.vsz", width=900, height=420)


def colorz(D, figs_dir):
    gr = np.asarray(D["cz_gr"]); ri = np.asarray(D["cz_ri"]); z = np.asarray(D["cz_z"])
    j = _sub(len(gr)); gr, ri, z = gr[j], ri[j], z[j]
    p = Panel([Series(gr, ri, cdata=z,
                      colormap="viridis", marker="dot", size="2.5pt", alpha=0.6)],
              xlabel="g - r", ylabel="r - i", title="colour-redshift relation (colour = z)",
              xrange=(float(np.percentile(D["cz_gr"], 1)), float(np.percentile(D["cz_gr"], 99))),
              yrange=(float(np.percentile(D["cz_ri"], 1)), float(np.percentile(D["cz_ri"], 99))))
    path = os.path.join(figs_dir, "colorz.vsz")
    V.scatter(path, p, width="18cm", height="15cm")
    return V.embed_tag("figs/colorz.vsz", width=640, height=540)


def weights(D, figs_dir):
    panels = []
    for key, name, col in [("wcp", "WEIGHT_CP (fiber collisions)", C_NEW),
                           ("wnoz", "WEIGHT_NOZ (redshift failures)", C_ZF),
                           ("wsys", "WEIGHT_SYSTOT (imaging)", C_OBS)]:
        w = np.asarray(D[key]); lo = min(0.6, float(w.min())); hi = min(3.0, float(w.max()))
        hb = np.linspace(lo, hi, 50); h, _ = np.histogram(w, hb); hx, hy = _steps(hb, np.maximum(h, 0.1))
        panels.append(Panel([Series(hx, hy, color=col, line_only=True)],
                            xlabel=name, ylabel="galaxies", title=f"<{name.split()[0]}> = {w.mean():.3f}",
                            ylog=True))
    path = os.path.join(figs_dir, "weights.vsz")
    V.grid(path, panels, rows=1, cols=3, width="33cm", height="11cm")
    return V.embed_tag("figs/weights.vsz", width=960, height=340)


def clpair(D, figs_dir):
    dz = np.asarray(D["dz_pool"]); hb = np.linspace(-0.04, 0.04, 81)
    h, _ = np.histogram(dz, hb, density=True); hx, hy = _steps(hb, h)
    p = Panel([Series(hx, hy, color=C_NEW, line_only=True)],
              xlabel="Delta z of observed close pairs", ylabel="density",
              title="close-pair redshift-separation prior p(Delta z)")
    path = os.path.join(figs_dir, "clpair.vsz")
    V.scatter(path, p, width="18cm", height="12cm")
    return V.embed_tag("figs/clpair.vsz", width=680, height=440)


def photoz(D, figs_dir):
    sp = np.asarray(D["pz_spec"]); ph = np.asarray(D["pz_phot"])
    j = _sub(len(sp), cap=12000)
    lim = (float(sp.min()), float(sp.max()))
    p1 = Panel([Series(sp[j], ph[j], color=C_NEW, marker="dot", size="2pt", alpha=0.45),
                Series(list(lim), list(lim), color="#c0392b", line_only=True, line_style="dashed")],
               xlabel="spectroscopic z", ylabel="photo-z (posterior median)",
               title=f"sigma_NMAD = {float(D['sigma_nmad']):.3f}", xrange=lim, yrange=lim)
    pit = np.asarray(D["pit"]); hb = np.linspace(0, 1, 21); h, _ = np.histogram(pit, hb); hx, hy = _steps(hb, h)
    p2 = Panel([Series(hx, hy, color=C_NEW, line_only=True),
                Series([0, 1], [len(pit)/20, len(pit)/20], color="#c0392b", line_only=True, line_style="dashed", label="uniform (ideal)")],
               xlabel="PIT", ylabel="count", title="posterior calibration")
    path = os.path.join(figs_dir, "photoz.vsz")
    V.grid(path, [p1, p2], rows=1, cols=2, width="28cm", height="12cm")
    return V.embed_tag("figs/photoz.vsz", width=900, height=400)


def systematics(D, figs_dir):
    tc = np.asarray(D["wt_tc"])
    a = np.asarray(D["wt_ens_data"]); b = np.asarray(D["wt_ens_pzonly"])
    mA, sA = a.mean(0), a.std(0); mB, sB = b.mean(0), b.std(0)
    p1 = Panel([Series(tc, mA, yerr=sA, color=C_NEW, marker="circle", size="3pt", line=True, label="photo-z x clustering"),
                Series(tc, mB, yerr=sB, color="#c0392b", marker="square", size="3pt", line=True, line_style="dashed", label="photo-z only")],
               xlabel="theta [deg]", ylabel="w(theta)", title="ensemble w(theta): two completion priors",
               xlog=True, ylog=True)
    ratio = np.abs(mA - mB) / (0.5 * (sA + sB) + 1e-30)
    p2 = Panel([Series(tc, ratio, color="#333333", marker="circle", size="3pt", line=True),
                Series([tc.min(), tc.max()], [1, 1], color="#c0392b", line_only=True, line_style="dashed", label="systematic = statistical")],
               xlabel="theta [deg]", ylabel="Delta_sys / sigma_stat", title="prior-systematic budget", xlog=True)
    path = os.path.join(figs_dir, "systematics.vsz")
    V.grid(path, [p1, p2], rows=1, cols=2, width="28cm", height="12cm")
    return V.embed_tag("figs/systematics.vsz", width=900, height=400)


def samples_nz(D, figs_dir):
    zb = np.asarray(D["nz_bins"])
    wx, wy = _steps(zb, np.asarray(D["nz_wobs"]))
    cx, cy = _steps(zb, np.asarray(D["nz_comp"]))
    p = Panel([Series(wx, wy, color=C_OBS, line_only=True, label="weighted observed n(z)"),
               Series(cx, cy, color=C_NEW, line_only=True, label="equal-weight completed n(z)")],
              xlabel="redshift z", ylabel="galaxies / bin", title="completed n(z) reproduces weighted n(z)")
    path = os.path.join(figs_dir, "samples_nz.vsz")
    V.scatter(path, p, width="20cm", height="13cm")
    return V.embed_tag("figs/samples_nz.vsz", width=760, height=480)
