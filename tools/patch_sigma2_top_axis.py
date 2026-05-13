"""One-shot patcher: add a top R_ref [Mpc/h] axis to each Quaia panel.

Inserts an ``axis-function`` widget that maps the bottom θ[deg] axis
to comoving radius R = D_M(z) × θ × π/180 in the project's fiducial
cosmology (Om=0.31, h=0.68). One coefficient per Quaia panel because
each is at a different z.

Idempotent: if ``r_top`` already exists in a panel, that panel is
skipped.

Run once:
    python tools/patch_sigma2_top_axis.py
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from twopt_density.distance import DistanceCosmo, comoving_distance


VSZ_FILE = Path(__file__).resolve().parent.parent / "vsz" / "sigma2.vsz"
ARTIFACT = Path(__file__).resolve().parent.parent / "output" / "quaia_full_dd.npz"
Z_PLOT_IDX = [8, 24, 40, 56]


def _coefficients() -> list[float]:
    """Return the per-panel θ[deg] → R[Mpc/h] coefficient (D_M × π/180)."""
    fid = DistanceCosmo(Om=0.31, h=0.68)
    art = np.load(ARTIFACT)
    z_edges = art["z_q_edges"]
    z_mids = 0.5 * (z_edges[:-1] + z_edges[1:])
    z_plot = [float(z_mids[i]) for i in Z_PLOT_IDX]
    return [float(comoving_distance(jnp.asarray(z), fid)) * math.pi / 180
            for z in z_plot]


def _axis_function_block(coef: float, t_min: float, t_max: float) -> str:
    """Veusz Set() block for one top R_ref axis.

    Uses Veusz's linked axis-function: ``linked=True`` +
    ``linkedaxis='x'`` makes the widget inherit the bottom-x range
    automatically, so we don't need mint/maxt. ``function='t * coef'``
    maps θ[deg] → R[Mpc/h] using the per-z fiducial-cosmology
    coefficient. (t_min/t_max are kept as a fallback.)
    """
    return (
        f"Add('axis-function', name='r_top', autoadd=False)\n"
        f"To('r_top')\n"
        f"Set('function', 't * {coef:.4f}')\n"
        f"Set('linked', True)\n"
        f"Set('linkedaxis', 'x')\n"
        f"Set('mint', {t_min})\n"
        f"Set('maxt', {t_max})\n"
        f"Set('label', 'R_{{ref}} [Mpc/h]')\n"
        f"Set('direction', 'horizontal')\n"
        f"Set('otherPosition', 1.0)\n"
        f"Set('log', False)\n"
        f"To('..')\n"
    )


def _x_range(block: str) -> tuple[float, float]:
    """Extract the bottom x-axis (min, max) from a panel block so the
    R_ref top axis spans the same θ range."""
    x_open = block.index("To('x')")
    x_close = block.index("To('..')", x_open)
    section = block[x_open:x_close]
    m_min = re.search(r"Set\('min',\s*([\d.eE+\-]+)\)", section)
    m_max = re.search(r"Set\('max',\s*([\d.eE+\-]+)\)", section)
    if not (m_min and m_max):
        raise ValueError("could not parse x min/max in panel block")
    return float(m_min.group(1)), float(m_max.group(1))


def main() -> None:
    coefs = _coefficients()
    src = VSZ_FILE.read_text()
    for slot, coef in enumerate(coefs):
        marker = f"Add('graph', name='panel_quaia_{slot}', autoadd=False)\n"
        if marker not in src:
            print(f"  panel_quaia_{slot}: marker not found, skipping",
                  file=sys.stderr)
            continue
        # Find the panel block: marker → next "Add('graph', ..." OR EOF
        start = src.index(marker)
        m = re.search(r"\nAdd\('graph',", src[start + len(marker):])
        end = start + len(marker) + (m.start() if m else len(src) - start - len(marker))
        block = src[start:end]

        t_min, t_max = _x_range(block)

        # Strip a stale r_top block if present (so reruns refresh
        # `function`/`mint`/`maxt` from the current x-axis range).
        block = re.sub(
            r"Add\('axis-function', name='r_top', autoadd=False\)\n"
            r"To\('r_top'\)\n(?:.*?\n)*?To\('\.\.'\)\n",
            "",
            block,
            flags=re.DOTALL,
        )

        # Insert the axis-function block right after the y-axis block.
        # Heuristic: find the first `To('..')` AFTER the `To('y')` line.
        y_open = block.index("To('y')")
        y_close_offset = block.index("To('..')", y_open) + len("To('..')") + 1
        new_block = (
            block[:y_close_offset]
            + _axis_function_block(coef, t_min, t_max)
            + block[y_close_offset:]
        )
        src = src[:start] + new_block + src[end:]
        print(f"  panel_quaia_{slot}: r_top set "
              f"(coef={coef:.4f}, t∈[{t_min}, {t_max}])")
    VSZ_FILE.write_text(src)
    print(f"wrote {VSZ_FILE}")


if __name__ == "__main__":
    main()
