#!/opt/homebrew/opt/python@3.11/bin/python3.11
"""
Streamlit app for the GraphGP Cosmology Pipeline.

Visualizes density field reconstruction, cosmic web classification,
and label-environment correlations from Quijote halo catalogs.

Run:  streamlit run app.py
"""

import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GraphGP Cosmology",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.environ.get(
    "GRAPHGP_DATA_DIR",
    "/Users/tabel/Research/data/quijote_halos_set_diffuser_data",
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
L_BOX = 1000.0
HAS_RAW_DATA = os.path.isfile(os.path.join(DATA_DIR, "train_halos.npy"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_results(path):
    """Load a pre-computed .npz results file."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.keys()}


@st.cache_data
def load_raw_sim(sim_idx):
    """Load raw simulation data."""
    halos = np.load(os.path.join(DATA_DIR, "train_halos.npy"), mmap_mode="r")
    sim = np.array(halos[sim_idx])
    return {
        "positions": sim[:, :3],
        "velocities": sim[:, 3:6],
        "masses": sim[:, 6],
    }


@st.cache_data
def load_synthetic():
    path = os.path.join(OUTPUT_DIR, "synthetic_validation_results.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.keys()}
    return None


def thin_for_3d(n, max_pts=4000):
    """Return random indices for subsampling large point clouds."""
    if n <= max_pts:
        return np.arange(n)
    rng = np.random.RandomState(0)
    return rng.choice(n, max_pts, replace=False)


def slice_mask(positions, axis, center, thickness):
    """Boolean mask for a thin slab."""
    coord = positions[:, axis]
    return (coord > center - thickness / 2) & (coord < center + thickness / 2)


def clip_axes(fig, x_data=None, y_data=None, pct=(1, 99), margin=0.05):
    """Set axis ranges to percentile bounds with margin, avoiding extreme tails."""
    rng = {}
    for data, axis in [(x_data, "xaxis"), (y_data, "yaxis")]:
        if data is not None:
            lo, hi = np.percentile(data, pct)
            pad = margin * (hi - lo) if hi > lo else 1.0
            rng[f"{axis}.range"] = [lo - pad, hi + pad]
    if rng:
        fig.update_layout(**rng)
    return fig


def equal_axes(fig, x_data, y_data, pct=(1, 99), margin=0.05):
    """Force equal x/y ranges for comparison scatters (1:1 line at 45 deg)."""
    combined = np.concatenate([np.asarray(x_data).ravel(),
                               np.asarray(y_data).ravel()])
    lo, hi = np.percentile(combined, pct)
    pad = margin * (hi - lo) if hi > lo else 1.0
    r = [lo - pad, hi + pad]
    fig.update_layout(**{"xaxis.range": r, "yaxis.range": r})
    return fig


def add_log_buttons(fig, x=True, y=True):
    """Add interactive log/linear toggle buttons to a Plotly figure."""
    buttons = []
    if x and y:
        buttons = [
            dict(label="Lin / Lin", method="relayout",
                 args=[{"xaxis.type": "linear", "yaxis.type": "linear"}]),
            dict(label="Log X", method="relayout",
                 args=[{"xaxis.type": "log", "yaxis.type": "linear"}]),
            dict(label="Log Y", method="relayout",
                 args=[{"xaxis.type": "linear", "yaxis.type": "log"}]),
            dict(label="Log / Log", method="relayout",
                 args=[{"xaxis.type": "log", "yaxis.type": "log"}]),
        ]
    elif x:
        buttons = [
            dict(label="Linear X", method="relayout",
                 args=[{"xaxis.type": "linear"}]),
            dict(label="Log X", method="relayout",
                 args=[{"xaxis.type": "log"}]),
        ]
    elif y:
        buttons = [
            dict(label="Linear Y", method="relayout",
                 args=[{"yaxis.type": "linear"}]),
            dict(label="Log Y", method="relayout",
                 args=[{"yaxis.type": "log"}]),
        ]
    if buttons:
        fig.update_layout(updatemenus=[
            dict(type="buttons", direction="left", buttons=buttons,
                 x=1.0, xanchor="right", y=1.15, yanchor="top",
                 bgcolor="rgba(255,255,255,0.7)", font=dict(size=10)),
        ])
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("GraphGP Cosmology")
st.sidebar.markdown(
    "Density field reconstruction from Quijote halo catalogs "
    "using [GraphGP](https://github.com/al-jshen/graphgp) "
    "(Vecchia approximation)."
)

# PDF documentation links
_PDF_DIR = os.path.dirname(__file__)
_miniproject_path = os.path.join(_PDF_DIR, "miniproject.pdf")
_proposal_path = os.path.join(_PDF_DIR, "proposal_v2.pdf")

st.sidebar.markdown("### Documentation")
if os.path.isfile(_miniproject_path):
    with open(_miniproject_path, "rb") as f:
        st.sidebar.download_button(
            "Mini-project description (PDF)",
            f.read(),
            file_name="miniproject.pdf",
            mime="application/pdf",
        )
if os.path.isfile(_proposal_path):
    with open(_proposal_path, "rb") as f:
        st.sidebar.download_button(
            "Collaboration proposal (PDF)",
            f.read(),
            file_name="proposal_v2.pdf",
            mime="application/pdf",
        )

results_path = os.path.join(OUTPUT_DIR, "gp_reconstruction_results.npz")
has_results = os.path.exists(results_path)

if not has_results:
    st.sidebar.warning("No pre-computed results found. Run `python graphGP_cosmo.py` first.")

tab_names = [
    "Overview",
    "Data Explorer",
    "Density Field",
    "Cosmic Web",
    "Hessian Comparison",
    "Environment Correlations",
    "Synthetic Validation",
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

tabs = st.tabs(tab_names)

# ===== TAB 0: OVERVIEW ====================================================
with tabs[0]:
    st.header("GraphGP Cosmology Pipeline")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
**Goal:** Reconstruct the smooth density field from discrete halo positions,
classify the cosmic web (peaks, filaments, sheets, voids), and test whether
halo properties (mass, velocity) depend on the local tidal environment
beyond their dependence on density alone.

**Method:**
1. Build a Vecchia-approximation neighbor graph on halo positions
2. Parameterize the density field via a Gaussian process in white-noise (xi) space
3. Maximize the Poisson log-likelihood + GP log-prior via gradient descent
4. Alternate between field and kernel hyperparameter optimization
5. Compute the Hessian via GP derivatives (autodiff through the kernel) to classify the cosmic web
6. Test label-environment correlations (Q1-Q4)

**Key equations:**
- Poisson likelihood: `ln L = sum_i ln[n_bar * (1 + delta_i)] - N`
- GP prior (xi-space): `ln p(xi) = -0.5 * ||xi||^2`
- Field: `delta = L * xi` where `L` is the Vecchia Cholesky factor
""")
    with col2:
        if has_results:
            r = load_results(results_path)
            st.metric("Halos", f"{len(r['positions']):,}")
            st.metric("Kernel variance", f"{float(r['kernel_variance']):.4f}")
            st.metric("Kernel scale", f"{float(r['kernel_scale_mpc_h']):.1f} Mpc/h")
            st.metric("Field range",
                       f"[{r['delta'].min():.2f}, {r['delta'].max():.2f}]")
        else:
            st.info("Run the pipeline to see summary metrics.")


# ===== TAB 1: DATA EXPLORER ===============================================
with tabs[1]:
    st.header("Quijote Halo Data Explorer")

    # Determine data source: raw sims (local) or pre-computed results
    _use_raw = HAS_RAW_DATA
    if _use_raw:
        sim_idx = st.slider("Simulation index", 0, 1799, 0, key="sim_slider")
        raw = load_raw_sim(sim_idx)
        pos = raw["positions"]
        log_mass = np.log10(raw["masses"] + 1e-10)
        v_mag = np.linalg.norm(raw["velocities"], axis=1)
        _src_label = f"Sim {sim_idx}"
    elif has_results:
        st.info("Raw simulation files not found. Showing halos from pre-computed results (sim 0).")
        _r = load_results(results_path)
        pos = _r["positions"]  # Mpc/h
        log_mass = np.array(_r["label_a"])
        v_mag = np.array(_r["label_b"])
        _src_label = "Pre-computed (sim 0)"
    else:
        st.warning("Neither raw data nor pre-computed results found.")
        st.stop()

    color_by = st.radio("Color by", ["log10(Mass)", "|Velocity|"], horizontal=True,
                         key="data_color")
    color_val = log_mass if "Mass" in color_by else v_mag
    clab = color_by

    # Label range filters
    st.subheader("Filter halos")
    fc1, fc2 = st.columns(2)
    with fc1:
        mass_range = st.slider(
            "log10(Mass) range",
            float(np.floor(log_mass.min() * 10) / 10),
            float(np.ceil(log_mass.max() * 10) / 10),
            (float(np.floor(log_mass.min() * 10) / 10),
             float(np.ceil(log_mass.max() * 10) / 10)),
            step=0.05,
            key="mass_filter",
        )
    with fc2:
        vel_range = st.slider(
            "|Velocity| range [km/s]",
            float(np.floor(v_mag.min())),
            float(np.ceil(v_mag.max())),
            (float(np.floor(v_mag.min())),
             float(np.ceil(v_mag.max()))),
            step=5.0,
            key="vel_filter",
        )

    # Apply filters
    filt = ((log_mass >= mass_range[0]) & (log_mass <= mass_range[1]) &
            (v_mag >= vel_range[0]) & (v_mag <= vel_range[1]))
    pos_f = pos[filt]
    color_f = color_val[filt]
    log_mass_f = log_mass[filt]
    v_mag_f = v_mag[filt]
    n_filt = int(filt.sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Displayed halos", f"{n_filt:,}")
    col2.metric("Mass range", f"{log_mass_f.min():.1f} - {log_mass_f.max():.1f}" if n_filt else "N/A")
    col3.metric("Vel range", f"{v_mag_f.min():.0f} - {v_mag_f.max():.0f} km/s" if n_filt else "N/A")
    col4.metric("Box", f"{L_BOX:.0f} Mpc/h")

    # 3D scatter (subsampled from filtered set)
    idx = thin_for_3d(n_filt, 3000)
    fig = go.Figure(data=[go.Scatter3d(
        x=pos_f[idx, 0], y=pos_f[idx, 1], z=pos_f[idx, 2],
        mode="markers",
        marker=dict(
            size=1.8,
            color=color_f[idx],
            colorscale="Viridis",
            colorbar=dict(title=clab, thickness=15),
            opacity=0.6,
        ),
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title="x [Mpc/h]", yaxis_title="y [Mpc/h]", zaxis_title="z [Mpc/h]",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        title=f"{_src_label}: {min(len(idx), n_filt)} of {n_filt} filtered halos shown",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Histograms
    col1, col2 = st.columns(2)
    with col1:
        fig_h = px.histogram(x=log_mass, nbins=60, labels={"x": "log10(M [Msun/h])"},
                             title="Halo mass distribution")
        fig_h.update_layout(height=350, showlegend=False)
        add_log_buttons(fig_h)
        st.plotly_chart(fig_h, use_container_width=True)
    with col2:
        fig_v = px.histogram(x=v_mag, nbins=60, labels={"x": "|v| [km/s]"},
                             title="Halo velocity magnitude distribution")
        fig_v.update_layout(height=350, showlegend=False)
        add_log_buttons(fig_v)
        st.plotly_chart(fig_v, use_container_width=True)


# ===== TAB 2: DENSITY FIELD ===============================================
with tabs[2]:
    st.header("Reconstructed Density Field")

    if not has_results:
        st.warning("No results found. Run `python graphGP_cosmo.py` first.")
    else:
        r = load_results(results_path)
        pos_n = r["positions"] / L_BOX  # normalized [0,1]
        delta = r["delta"]
        losses = r["losses"]

        # --- Convergence ---
        st.subheader("MAP Convergence")
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(y=losses, mode="lines", name="-log posterior"))
        fig_conv.update_layout(
            xaxis_title="Optimization step",
            yaxis_title="-log posterior",
            height=350,
        )
        add_log_buttons(fig_conv)
        st.plotly_chart(fig_conv, use_container_width=True)

        # --- Kernel ---
        st.subheader("Learned Kernel C(r)")
        variance = float(r["kernel_variance"])
        scale_mpc = float(r["kernel_scale_mpc_h"])
        scale_box = scale_mpc / L_BOX
        fisher_unc = r["fisher_uncertainties"]
        sig_lv = float(fisher_unc[0])
        sig_ls = float(fisher_unc[1])

        r_plot = np.linspace(0, 200, 500)
        cr = variance * np.exp(-0.5 * (r_plot / scale_mpc) ** 2)

        # +/- 1 sigma bands
        v_hi = np.exp(np.log(variance) + sig_lv)
        s_hi = np.exp(np.log(scale_mpc) + sig_ls * L_BOX / scale_mpc * scale_mpc / L_BOX)
        s_hi = scale_mpc * np.exp(sig_ls)
        v_lo = np.exp(np.log(variance) - sig_lv)
        s_lo = scale_mpc * np.exp(-sig_ls)

        cr_hi = v_hi * np.exp(-0.5 * (r_plot / s_hi) ** 2)
        cr_lo = v_lo * np.exp(-0.5 * (r_plot / s_lo) ** 2)

        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(x=r_plot, y=cr_hi, mode="lines",
                                    line=dict(width=0), showlegend=False))
        fig_k.add_trace(go.Scatter(x=r_plot, y=cr_lo, mode="lines",
                                    line=dict(width=0), fill="tonexty",
                                    fillcolor="rgba(68,114,196,0.2)",
                                    name="1-sigma band"))
        fig_k.add_trace(go.Scatter(x=r_plot, y=cr, mode="lines",
                                    line=dict(color="royalblue", width=2.5),
                                    name="MAP kernel"))
        fig_k.update_layout(
            xaxis_title="r [Mpc/h]",
            yaxis_title="C(r)",
            height=350,
            title=f"Kernel: variance = {variance:.4f} +/- {variance*(np.exp(sig_lv)-1):.4f}, "
                  f"scale = {scale_mpc:.1f} +/- {scale_mpc*(np.exp(sig_ls)-1):.1f} Mpc/h",
        )
        add_log_buttons(fig_k)
        st.plotly_chart(fig_k, use_container_width=True)

        # --- 3D density field ---
        st.subheader("3D Density Field")

        view_mode = st.radio(
            "View",
            ["3D point cloud", "2D slice (points)", "2D slice (interpolated image)"],
            horizontal=True, key="density_view",
        )

        if view_mode == "3D point cloud":
            idx = thin_for_3d(len(pos_n), 3000)
            dmin, dmax = float(np.percentile(delta, 2)), float(np.percentile(delta, 98))
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=pos_n[idx, 0], y=pos_n[idx, 1], z=pos_n[idx, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=delta[idx],
                    colorscale="RdBu_r",
                    cmin=dmin, cmax=dmax,
                    colorbar=dict(title="delta", thickness=15),
                    opacity=0.6,
                ),
            )])
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title="x", yaxis_title="y", zaxis_title="z",
                    aspectmode="cube",
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                height=600,
                title="Reconstructed density (box units [0,1])",
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        elif view_mode == "2D slice (points)":
            axis_name = st.selectbox("Slice axis", ["z", "y", "x"], key="slice_axis")
            axis_idx = {"x": 0, "y": 1, "z": 2}[axis_name]
            center = st.slider("Slice center", 0.0, 1.0, 0.5, 0.01, key="slice_center")
            thickness = st.slider("Slice thickness", 0.02, 0.3, 0.1, 0.01, key="slice_thick")

            mask = slice_mask(pos_n, axis_idx, center, thickness)
            n_in_slice = mask.sum()
            st.caption(f"{n_in_slice} halos in slice")

            other_axes = [i for i in range(3) if i != axis_idx]
            ax_labels = ["x", "y", "z"]

            fig_sl = go.Figure(data=[go.Scatter(
                x=pos_n[mask, other_axes[0]],
                y=pos_n[mask, other_axes[1]],
                mode="markers",
                marker=dict(
                    size=4,
                    color=delta[mask],
                    colorscale="RdBu_r",
                    cmin=float(np.percentile(delta, 5)),
                    cmax=float(np.percentile(delta, 95)),
                    colorbar=dict(title="delta", thickness=15),
                ),
            )])
            fig_sl.update_layout(
                xaxis_title=f"{ax_labels[other_axes[0]]} [box]",
                yaxis_title=f"{ax_labels[other_axes[1]]} [box]",
                xaxis=dict(scaleanchor="y"),
                height=600,
                title=f"{axis_name}-slice at {center:.2f} +/- {thickness/2:.2f}",
            )
            st.plotly_chart(fig_sl, use_container_width=True)

        else:  # Interpolated image
            from scipy.interpolate import RBFInterpolator

            axis_name_img = st.selectbox("Slice axis", ["z", "y", "x"],
                                          key="img_slice_axis")
            axis_idx_img = {"x": 0, "y": 1, "z": 2}[axis_name_img]
            center_img = st.slider("Slice center", 0.0, 1.0, 0.5, 0.01,
                                    key="img_slice_center")
            thickness_img = st.slider("Slab thickness (points used for interpolation)",
                                       0.02, 0.3, 0.1, 0.01, key="img_slice_thick")
            grid_res = st.slider("Grid resolution", 64, 256, 128, 16,
                                  key="img_grid_res")

            mask_img = slice_mask(pos_n, axis_idx_img, center_img, thickness_img)
            n_in = int(mask_img.sum())
            st.caption(f"Interpolating from {n_in} halos in slab")

            other_ax = [i for i in range(3) if i != axis_idx_img]
            ax_labels = ["x", "y", "z"]

            if n_in < 10:
                st.warning("Too few points in slab. Increase thickness or change center.")
            else:
                # Source points: 2D coordinates within the slice
                pts_2d = pos_n[mask_img][:, other_ax]
                vals = delta[mask_img]

                # Uniform grid
                g1 = np.linspace(0, 1, grid_res)
                g2 = np.linspace(0, 1, grid_res)
                G1, G2 = np.meshgrid(g1, g2)
                grid_pts = np.column_stack([G1.ravel(), G2.ravel()])

                # RBF interpolation (thin-plate spline, fast for this size)
                with st.spinner("Interpolating density onto grid..."):
                    rbf = RBFInterpolator(pts_2d, vals, kernel="thin_plate_spline",
                                           smoothing=1e-3)
                    grid_vals = rbf(grid_pts).reshape(grid_res, grid_res)

                dmin_img = float(np.percentile(delta, 5))
                dmax_img = float(np.percentile(delta, 95))

                fig_img = go.Figure(data=[go.Heatmap(
                    z=grid_vals,
                    x=g1, y=g2,
                    colorscale="RdBu_r",
                    zmin=dmin_img, zmax=dmax_img,
                    colorbar=dict(title="delta", thickness=15),
                )])
                fig_img.update_layout(
                    xaxis_title=f"{ax_labels[other_ax[0]]} [box]",
                    yaxis_title=f"{ax_labels[other_ax[1]]} [box]",
                    xaxis=dict(scaleanchor="y"),
                    height=650,
                    title=(f"Interpolated density: {axis_name_img}-slice at "
                           f"{center_img:.2f} +/- {thickness_img/2:.2f}  "
                           f"({grid_res}x{grid_res} grid)"),
                )
                st.plotly_chart(fig_img, use_container_width=True)

        # Delta histogram
        st.subheader("Density field distribution")
        fig_dh = px.histogram(x=delta, nbins=80, labels={"x": "delta"},
                               title="Distribution of reconstructed delta at halo positions")
        fig_dh.update_layout(height=300, showlegend=False)
        add_log_buttons(fig_dh)
        st.plotly_chart(fig_dh, use_container_width=True)


# ===== TAB 3: COSMIC WEB ==================================================
with tabs[3]:
    st.header("Cosmic Web Classification")

    if not has_results:
        st.warning("No results found.")
    else:
        r = load_results(results_path)
        pos_n = r["positions"] / L_BOX
        eigenvalues = r["eigenvalues"]
        labels_geo = r["labels_geo"]
        laplacian = r["laplacian"]
        s_squared = r["s_squared"]

        # Classification counts
        geo_types = ["peak", "filament", "sheet", "void"]
        color_map = {"peak": "#e74c3c", "filament": "#f39c12",
                     "sheet": "#85c1e9", "void": "#1a5276"}

        counts = {g: int(np.sum(labels_geo == g)) for g in geo_types}
        cols = st.columns(4)
        for i, g in enumerate(geo_types):
            cols[i].metric(g.capitalize(), f"{counts[g]:,}  ({100*counts[g]/len(labels_geo):.1f}%)")

        # 3D cosmic web
        st.subheader("3D Cosmic Web")
        view_cw = st.radio("View", ["3D", "2D slice"], horizontal=True, key="cw_view")

        if view_cw == "3D":
            idx = thin_for_3d(len(pos_n), 3000)
            fig_cw = go.Figure()
            for g in geo_types:
                mask_g = (labels_geo[idx] == g)
                if mask_g.sum() > 0:
                    fig_cw.add_trace(go.Scatter3d(
                        x=pos_n[idx][mask_g, 0],
                        y=pos_n[idx][mask_g, 1],
                        z=pos_n[idx][mask_g, 2],
                        mode="markers",
                        marker=dict(size=2, color=color_map[g], opacity=0.5),
                        name=g.capitalize(),
                    ))
            fig_cw.update_layout(
                scene=dict(
                    xaxis_title="x", yaxis_title="y", zaxis_title="z",
                    aspectmode="cube",
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                height=600,
            )
            st.plotly_chart(fig_cw, use_container_width=True)
        else:
            center_cw = st.slider("Slice center", 0.0, 1.0, 0.5, 0.01, key="cw_slice_c")
            thick_cw = st.slider("Thickness", 0.02, 0.3, 0.1, 0.01, key="cw_slice_t")
            mask_sl = slice_mask(pos_n, 2, center_cw, thick_cw)
            fig_cw2 = go.Figure()
            for g in geo_types:
                m = mask_sl & (labels_geo == g)
                if m.sum() > 0:
                    fig_cw2.add_trace(go.Scatter(
                        x=pos_n[m, 0], y=pos_n[m, 1],
                        mode="markers",
                        marker=dict(size=5, color=color_map[g]),
                        name=g.capitalize(),
                    ))
            fig_cw2.update_layout(
                xaxis_title="x [box]", yaxis_title="y [box]",
                xaxis=dict(scaleanchor="y"),
                height=600,
                title=f"z-slice at {center_cw:.2f} +/- {thick_cw/2:.2f}",
            )
            st.plotly_chart(fig_cw2, use_container_width=True)

        # Eigenvalue distributions
        st.subheader("Hessian Eigenvalue Distributions")
        fig_eig = go.Figure()
        names = ["lambda_1 (largest)", "lambda_2", "lambda_3 (smallest)"]
        colors_eig = ["#e74c3c", "#27ae60", "#2980b9"]
        for i in range(3):
            fig_eig.add_trace(go.Histogram(
                x=eigenvalues[:, i], nbinsx=80, name=names[i],
                marker_color=colors_eig[i], opacity=0.55,
            ))
        fig_eig.update_layout(
            barmode="overlay", height=350,
            xaxis_title="Eigenvalue", yaxis_title="Count",
        )
        clip_axes(fig_eig, x_data=eigenvalues.ravel())
        add_log_buttons(fig_eig)
        st.plotly_chart(fig_eig, use_container_width=True)

        # Tidal shear & laplacian
        col1, col2 = st.columns(2)
        with col1:
            fig_lap = px.histogram(x=laplacian, nbins=80, title="Laplacian (tr H)",
                                    labels={"x": "Laplacian"})
            fig_lap.update_layout(height=300, showlegend=False)
            clip_axes(fig_lap, x_data=laplacian)
            add_log_buttons(fig_lap)
            st.plotly_chart(fig_lap, use_container_width=True)
        with col2:
            fig_s2 = px.histogram(x=np.log10(s_squared + 1), nbins=80,
                                   title="log10(1 + s^2) Tidal Shear",
                                   labels={"x": "log10(1 + s^2)"})
            fig_s2.update_layout(height=300, showlegend=False)
            add_log_buttons(fig_s2)
            st.plotly_chart(fig_s2, use_container_width=True)


# ===== TAB 4: HESSIAN COMPARISON ==========================================
with tabs[4]:
    st.header("Hessian Method Comparison")
    st.markdown("""
Compare the **GP-derivative Hessian** (analytically correct — differentiates the kernel)
with the **quadratic-fit Hessian** (ad-hoc local least-squares, ignores the GP kernel).
""")

    if not has_results:
        st.warning("No results found. Run `python graphGP_cosmo.py` first.")
    else:
        r = load_results(results_path)

        has_qf = "eigenvalues_qf" in r
        if not has_qf:
            st.warning("Quadratic-fit results not found in saved data. "
                       "Re-run `python graphGP_cosmo.py` to generate comparison data.")
        else:
            eigenvalues_gp = r["eigenvalues"]
            eigenvalues_qf = r["eigenvalues_qf"]
            labels_gp = r["labels_geo"]
            labels_qf = r["labels_geo_qf"]
            laplacian_gp = r["laplacian"]
            laplacian_qf = r["laplacian_qf"]
            s2_gp = r["s_squared"]
            s2_qf = r["s_squared_qf"]
            pos_n = r["positions"] / L_BOX
            N = len(eigenvalues_gp)

            # --- Classification comparison ---
            st.subheader("Cosmic Web Classification")
            geo_types = ["peak", "filament", "sheet", "void"]
            color_map_cw = {"peak": "#e74c3c", "filament": "#f39c12",
                            "sheet": "#85c1e9", "void": "#1a5276"}

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**GP Derivatives**")
                for g in geo_types:
                    c = int(np.sum(labels_gp == g))
                    st.write(f"  {g.capitalize()}: {c} ({100*c/N:.1f}%)")
            with col2:
                st.markdown("**Quadratic Fit**")
                for g in geo_types:
                    c = int(np.sum(labels_qf == g))
                    st.write(f"  {g.capitalize()}: {c} ({100*c/N:.1f}%)")

            agreement = np.mean(labels_gp == labels_qf)
            st.metric("Classification Agreement", f"{100*agreement:.1f}%")

            # --- Eigenvalue scatter ---
            st.subheader("Eigenvalue Comparison (GP vs Quadratic Fit)")
            eig_names = ["lambda_1 (largest)", "lambda_2", "lambda_3 (smallest)"]
            cols_eig = st.columns(3)
            for i, col in enumerate(cols_eig):
                with col:
                    from scipy.stats import pearsonr
                    r_val, _ = pearsonr(eigenvalues_gp[:, i], eigenvalues_qf[:, i])
                    idx_s = thin_for_3d(N, 2000)
                    fig_sc = go.Figure(data=[go.Scatter(
                        x=eigenvalues_gp[idx_s, i], y=eigenvalues_qf[idx_s, i],
                        mode="markers",
                        marker=dict(size=2, opacity=0.3, color="steelblue"),
                    )])
                    # 1:1 line spanning the clipped range
                    combined = np.concatenate([eigenvalues_gp[:, i], eigenvalues_qf[:, i]])
                    lo, hi = np.percentile(combined, [1, 99])
                    pad = 0.05 * (hi - lo)
                    lims = [lo - pad, hi + pad]
                    fig_sc.add_trace(go.Scatter(
                        x=lims, y=lims, mode="lines",
                        line=dict(color="red", dash="dash"), showlegend=False,
                    ))
                    fig_sc.update_layout(
                        xaxis_title="GP derivative",
                        yaxis_title="Quadratic fit",
                        height=350,
                        title=f"{eig_names[i]} (r={r_val:.3f})",
                    )
                    equal_axes(fig_sc, eigenvalues_gp[:, i], eigenvalues_qf[:, i])
                    add_log_buttons(fig_sc)
                    st.plotly_chart(fig_sc, use_container_width=True)

            # --- Eigenvalue distributions overlay ---
            st.subheader("Eigenvalue Distributions")
            colors_gp = ["rgba(231,76,60,0.5)", "rgba(39,174,96,0.5)", "rgba(41,128,185,0.5)"]
            colors_qf = ["rgba(231,76,60,0.25)", "rgba(39,174,96,0.25)", "rgba(41,128,185,0.25)"]
            fig_dist = go.Figure()
            for i in range(3):
                fig_dist.add_trace(go.Histogram(
                    x=eigenvalues_gp[:, i], nbinsx=80,
                    name=f"GP {eig_names[i]}", marker_color=colors_gp[i],
                    opacity=0.6,
                ))
                fig_dist.add_trace(go.Histogram(
                    x=eigenvalues_qf[:, i], nbinsx=80,
                    name=f"QF {eig_names[i]}", marker_color=colors_qf[i],
                    opacity=0.4,
                ))
            fig_dist.update_layout(
                barmode="overlay", height=400,
                xaxis_title="Eigenvalue", yaxis_title="Count",
            )
            all_eig = np.concatenate([eigenvalues_gp.ravel(), eigenvalues_qf.ravel()])
            clip_axes(fig_dist, x_data=all_eig)
            add_log_buttons(fig_dist)
            st.plotly_chart(fig_dist, use_container_width=True)

            # --- Laplacian and tidal shear comparison ---
            st.subheader("Laplacian & Tidal Shear")
            col1, col2 = st.columns(2)
            with col1:
                r_lap, _ = pearsonr(laplacian_gp, laplacian_qf)
                idx_s = thin_for_3d(N, 2000)
                fig_lap = go.Figure(data=[go.Scatter(
                    x=laplacian_gp[idx_s], y=laplacian_qf[idx_s],
                    mode="markers",
                    marker=dict(size=2, opacity=0.3, color="steelblue"),
                )])
                combined_l = np.concatenate([laplacian_gp, laplacian_qf])
                lo_l, hi_l = np.percentile(combined_l, [1, 99])
                pad_l = 0.05 * (hi_l - lo_l)
                lims_l = [lo_l - pad_l, hi_l + pad_l]
                fig_lap.add_trace(go.Scatter(
                    x=lims_l, y=lims_l, mode="lines",
                    line=dict(color="red", dash="dash"), showlegend=False,
                ))
                fig_lap.update_layout(
                    xaxis_title="GP Laplacian", yaxis_title="QF Laplacian",
                    height=350, title=f"Laplacian (r={r_lap:.3f})",
                )
                equal_axes(fig_lap, laplacian_gp, laplacian_qf)
                add_log_buttons(fig_lap)
                st.plotly_chart(fig_lap, use_container_width=True)
            with col2:
                r_s2, _ = pearsonr(s2_gp, s2_qf)
                fig_s2 = go.Figure(data=[go.Scatter(
                    x=s2_gp[idx_s], y=s2_qf[idx_s],
                    mode="markers",
                    marker=dict(size=2, opacity=0.3, color="darkorange"),
                )])
                combined_s = np.concatenate([s2_gp, s2_qf])
                lo_s, hi_s = np.percentile(combined_s, [1, 99])
                pad_s = 0.05 * (hi_s - lo_s)
                lims_s = [lo_s - pad_s, hi_s + pad_s]
                fig_s2.add_trace(go.Scatter(
                    x=lims_s, y=lims_s, mode="lines",
                    line=dict(color="red", dash="dash"), showlegend=False,
                ))
                fig_s2.update_layout(
                    xaxis_title="GP s^2", yaxis_title="QF s^2",
                    height=350, title=f"Tidal shear s^2 (r={r_s2:.3f})",
                )
                equal_axes(fig_s2, s2_gp, s2_qf)
                add_log_buttons(fig_s2)
                st.plotly_chart(fig_s2, use_container_width=True)

            # --- Spatial comparison (2D slice) ---
            st.subheader("Spatial Classification Comparison (z-slice)")
            center_hc = st.slider("Slice center", 0.0, 1.0, 0.5, 0.01, key="hc_slice_c")
            thick_hc = st.slider("Thickness", 0.02, 0.3, 0.1, 0.01, key="hc_slice_t")
            mask_hc = slice_mask(pos_n, 2, center_hc, thick_hc)

            col1, col2 = st.columns(2)
            for col, labels, title in [(col1, labels_gp, "GP Derivatives"),
                                        (col2, labels_qf, "Quadratic Fit")]:
                with col:
                    fig_cw = go.Figure()
                    for g in geo_types:
                        m = mask_hc & (labels == g)
                        if m.sum() > 0:
                            fig_cw.add_trace(go.Scatter(
                                x=pos_n[m, 0], y=pos_n[m, 1],
                                mode="markers",
                                marker=dict(size=5, color=color_map_cw[g]),
                                name=g.capitalize(),
                            ))
                    fig_cw.update_layout(
                        xaxis_title="x [box]", yaxis_title="y [box]",
                        xaxis=dict(scaleanchor="y"),
                        height=500, title=title,
                    )
                    st.plotly_chart(fig_cw, use_container_width=True)


# ===== TAB 5: ENVIRONMENT CORRELATIONS ====================================
with tabs[5]:
    st.header("Label-Environment Correlations (Q1-Q4)")

    if not has_results:
        st.warning("No results found.")
    else:
        r = load_results(results_path)
        delta = r["delta"]
        label_a = r["label_a"]
        label_b = r["label_b"]
        s_squared = r["s_squared"]
        eigenvalues = r["eigenvalues"]
        labels_geo = r["labels_geo"]
        laplacian = r["laplacian"]

        label_a_name = "log10(M)"
        label_b_name = "|v| [km/s]"

        from scipy.stats import pearsonr
        from numpy.polynomial import polynomial as P

        def _partial_corr(x, y, z):
            cx = P.polyfit(z, x, 1)
            cy = P.polyfit(z, y, 1)
            rx = x - P.polyval(z, cx)
            ry = y - P.polyval(z, cy)
            return pearsonr(rx, ry)

        # ------ Q1 ------
        st.subheader("Q1: Do labels correlate with density?")
        r_a_d, p_a_d = pearsonr(delta, label_a)
        r_b_d, p_b_d = pearsonr(delta, label_b)

        col1, col2 = st.columns(2)
        col1.metric(f"Corr({label_a_name}, delta)", f"{r_a_d:+.4f}",
                     delta=f"p = {p_a_d:.2e}")
        col2.metric(f"Corr({label_b_name}, delta)", f"{r_b_d:+.4f}",
                     delta=f"p = {p_b_d:.2e}")

        # Scatter plots
        col1, col2 = st.columns(2)
        with col1:
            idx_s = thin_for_3d(len(delta), 2000)
            fig_q1a = go.Figure(data=[go.Scatter(
                x=delta[idx_s], y=label_a[idx_s], mode="markers",
                marker=dict(size=2, opacity=0.3, color="steelblue"),
            )])
            fig_q1a.update_layout(xaxis_title="delta", yaxis_title=label_a_name,
                                   height=350, title=f"{label_a_name} vs delta")
            add_log_buttons(fig_q1a)
            st.plotly_chart(fig_q1a, use_container_width=True)
        with col2:
            fig_q1b = go.Figure(data=[go.Scatter(
                x=delta[idx_s], y=label_b[idx_s], mode="markers",
                marker=dict(size=2, opacity=0.3, color="darkorange"),
            )])
            fig_q1b.update_layout(xaxis_title="delta", yaxis_title=label_b_name,
                                   height=350, title=f"{label_b_name} vs delta")
            add_log_buttons(fig_q1b)
            st.plotly_chart(fig_q1b, use_container_width=True)

        # ------ Q2 ------
        st.subheader("Q2: Partial correlation with tidal shear (at fixed delta)")
        st.markdown("Tests whether labels depend on the *shape* of the local "
                     "potential (s^2 = tidal shear) beyond density alone.")
        r_a_s2, p_a_s2 = _partial_corr(label_a, s_squared, delta)
        r_b_s2, p_b_s2 = _partial_corr(label_b, s_squared, delta)

        col1, col2 = st.columns(2)
        col1.metric(f"Corr({label_a_name}, s^2 | delta)", f"{r_a_s2:+.4f}",
                     delta=f"p = {p_a_s2:.2e}")
        col2.metric(f"Corr({label_b_name}, s^2 | delta)", f"{r_b_s2:+.4f}",
                     delta=f"p = {p_b_s2:.2e}")

        # Residual scatter
        col1, col2 = st.columns(2)
        with col1:
            ca = P.polyfit(delta, label_a, 1)
            cs = P.polyfit(delta, s_squared, 1)
            res_a = label_a - P.polyval(delta, ca)
            res_s = s_squared - P.polyval(delta, cs)
            fig_r1 = go.Figure(data=[go.Scatter(
                x=res_s[idx_s], y=res_a[idx_s], mode="markers",
                marker=dict(size=2, opacity=0.3, color="steelblue"),
            )])
            fig_r1.update_layout(
                xaxis_title="s^2 residual (delta removed)",
                yaxis_title=f"{label_a_name} residual",
                height=350,
                title="Q2: Residuals after removing density",
            )
            add_log_buttons(fig_r1)
            st.plotly_chart(fig_r1, use_container_width=True)
        with col2:
            cb = P.polyfit(delta, label_b, 1)
            res_b = label_b - P.polyval(delta, cb)
            fig_r2 = go.Figure(data=[go.Scatter(
                x=res_s[idx_s], y=res_b[idx_s], mode="markers",
                marker=dict(size=2, opacity=0.3, color="darkorange"),
            )])
            fig_r2.update_layout(
                xaxis_title="s^2 residual (delta removed)",
                yaxis_title=f"{label_b_name} residual",
                height=350,
                title="Q2: Residuals after removing density",
            )
            add_log_buttons(fig_r2)
            st.plotly_chart(fig_r2, use_container_width=True)

        # ------ Q3 ------
        st.subheader("Q3: Partial correlation with prolateness (at fixed delta)")
        denom = np.where(np.abs(eigenvalues[:, 2]) > 1e-10, eigenvalues[:, 2], 1e-10)
        prolateness = eigenvalues[:, 0] / denom
        r_a_pr, p_a_pr = _partial_corr(label_a, prolateness, delta)
        r_b_pr, p_b_pr = _partial_corr(label_b, prolateness, delta)

        col1, col2 = st.columns(2)
        col1.metric(f"Corr({label_a_name}, l1/l3 | delta)", f"{r_a_pr:+.4f}",
                     delta=f"p = {p_a_pr:.2e}")
        col2.metric(f"Corr({label_b_name}, l1/l3 | delta)", f"{r_b_pr:+.4f}",
                     delta=f"p = {p_b_pr:.2e}")

        # ------ Q4 ------
        st.subheader("Q4: Mean labels by cosmic web type")
        geo_types = ["peak", "filament", "sheet", "void"]
        rows = []
        for g in geo_types:
            mask = labels_geo == g
            if mask.sum() > 0:
                rows.append({
                    "Type": g.capitalize(),
                    "N": int(mask.sum()),
                    f"Mean {label_a_name}": float(label_a[mask].mean()),
                    f"Mean {label_b_name}": float(label_b[mask].mean()),
                })
        st.table(rows)

        # Bar chart
        fig_q4 = make_subplots(rows=1, cols=2,
                                subplot_titles=[f"Mean {label_a_name}", f"Mean {label_b_name}"])
        types = [row["Type"] for row in rows]
        colors_q4 = ["#e74c3c", "#f39c12", "#85c1e9", "#1a5276"]
        fig_q4.add_trace(go.Bar(
            x=types, y=[row[f"Mean {label_a_name}"] for row in rows],
            marker_color=colors_q4[:len(types)], showlegend=False,
        ), row=1, col=1)
        fig_q4.add_trace(go.Bar(
            x=types, y=[row[f"Mean {label_b_name}"] for row in rows],
            marker_color=colors_q4[:len(types)], showlegend=False,
        ), row=1, col=2)
        fig_q4.update_layout(height=350)
        st.plotly_chart(fig_q4, use_container_width=True)


# ===== TAB 6: SYNTHETIC VALIDATION ========================================
with tabs[6]:
    st.header("Synthetic Ground-Truth Validation")

    syn = load_synthetic()
    if syn is None:
        st.warning("No synthetic results found. Run `python synthetic_test.py` first.")
    else:
        st.markdown("""
**Setup:** Generate a known GP field with true kernel parameters,
Poisson-thin to simulate observed halos, assign labels with known
functional forms, then run the reconstruction and check recovery.
""")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Field corr", f"{float(syn['r_field']):.3f}",
                     delta="PASS" if float(syn['r_field']) > 0.5 else "FAIL")
        col2.metric("Var error", f"{100*float(syn['var_err']):.1f}%",
                     delta="PASS" if float(syn['var_err']) < 0.5 else "FAIL",
                     delta_color="inverse")
        col3.metric("Scale error", f"{100*float(syn['scale_err']):.1f}%",
                     delta="PASS" if float(syn['scale_err']) < 0.5 else "FAIL",
                     delta_color="inverse")
        col4.metric("Classification", f"{100*float(syn['classification_agreement']):.1f}%",
                     delta="PASS" if float(syn['classification_agreement']) > 0.4 else "FAIL")

        # Kernel comparison
        st.subheader("Kernel Recovery")
        col1, col2 = st.columns(2)
        with col1:
            true_var = float(syn["true_var"])
            true_scale = float(syn["true_scale"])
            learned_var = float(syn["learned_var"])
            learned_scale = float(syn["learned_scale"])

            st.markdown(f"""
| Parameter | True | Learned | Error |
|-----------|------|---------|-------|
| Variance  | {true_var:.4f} | {learned_var:.4f} | {100*float(syn['var_err']):.1f}% |
| Scale     | {true_scale:.4f} | {learned_scale:.4f} | {100*float(syn['scale_err']):.1f}% |
""")

        with col2:
            r_plot = np.linspace(0, 0.3, 300)
            cr_true = true_var * np.exp(-0.5 * (r_plot / true_scale) ** 2)
            cr_learned = learned_var * np.exp(-0.5 * (r_plot / learned_scale) ** 2)
            fig_kv = go.Figure()
            fig_kv.add_trace(go.Scatter(x=r_plot, y=cr_true, name="True kernel",
                                         line=dict(color="green", width=2, dash="dash")))
            fig_kv.add_trace(go.Scatter(x=r_plot, y=cr_learned, name="Learned kernel",
                                         line=dict(color="royalblue", width=2)))
            fig_kv.update_layout(xaxis_title="r [box units]", yaxis_title="C(r)",
                                  height=300)
            add_log_buttons(fig_kv)
            st.plotly_chart(fig_kv, use_container_width=True)

        # Field scatter
        st.subheader("Field Recovery")
        true_d = syn["true_delta"]
        recon_d = syn["recon_delta"]

        fig_fs = go.Figure()
        idx_sv = thin_for_3d(len(true_d), 2000)
        fig_fs.add_trace(go.Scatter(
            x=true_d[idx_sv], y=recon_d[idx_sv], mode="markers",
            marker=dict(size=2.5, opacity=0.3, color="steelblue"),
            name="Halos",
        ))
        combined_d = np.concatenate([true_d, recon_d])
        lo_d, hi_d = np.percentile(combined_d, [1, 99])
        pad_d = 0.05 * (hi_d - lo_d)
        lims = [lo_d - pad_d, hi_d + pad_d]
        fig_fs.add_trace(go.Scatter(
            x=lims, y=lims, mode="lines",
            line=dict(color="red", dash="dash"), name="Perfect recovery",
        ))
        fig_fs.update_layout(
            xaxis_title="True delta",
            yaxis_title="Reconstructed delta",
            height=450,
            title=f"Correlation = {float(syn['r_field']):.3f}",
        )
        equal_axes(fig_fs, true_d, recon_d)
        add_log_buttons(fig_fs)
        st.plotly_chart(fig_fs, use_container_width=True)

        # Q2 check
        st.subheader("Q2 Signal Detection")
        col1, col2 = st.columns(2)
        col1.metric("label_a ~ s^2|delta (expect significant)",
                     f"r = {float(syn['r_a_s2']):+.4f}",
                     delta=f"p = {float(syn['p_a_s2']):.2e}")
        col2.metric("label_b ~ s^2|delta (expect non-significant)",
                     f"r = {float(syn['r_b_s2']):+.4f}",
                     delta=f"p = {float(syn['p_b_s2']):.2e}")

        st.markdown("""
**Interpretation:** `label_a` was constructed with explicit tidal shear
dependence (`0.5*delta + 0.3*s^2 + noise`), so its partial correlation
with s^2 at fixed delta should be significant. `label_b` depends only on
density (`0.4*delta + noise`), so its partial correlation should be weak.
""")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("GraphGP Cosmology Pipeline | March 2026")
