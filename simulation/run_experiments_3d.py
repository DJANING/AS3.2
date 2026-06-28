"""
Exact 3D reconstruction figures (Janning et al. 2014, paper geometry).

Uses simulation/kiss_and_hop_3d.py: a tube L=100 um, R=500 nm filled with 60
hexagonally packed microtubules (R_MT=12.5 nm, 70 nm spacing), dt=1 us, with the
paper's exact bind/unbind loop, calibrated to k*on/koff ~ 100 (p_on=0.20).

Produces figures/fig6_exact_3d.png:
  (left)  the 60-MT bundle cross-section inside the 500 nm process,
  (right) the step-size distribution of the 2D-projected single-molecule
          localisations -- the localisation-noise peak plus the inter-MT hop
          population, compared with the experimental peak positions reported in
          the paper (95 / 190 / 285 nm) and the multiples of the 70 nm input
          spacing.

Run:  python3 simulation/run_experiments_3d.py
"""

from __future__ import annotations

import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from kiss_and_hop_3d import simulate3d, Params3D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(ROOT, "figures")
DATADIR = os.path.join(ROOT, "data")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
BLUE, RED, GREEN, ORANGE = "#2b6cb0", "#c53030", "#2f855a", "#dd6b20"
P_ON_KON100 = 0.20                      # gives k*on/koff ~ 100 (~99 % bound)
PAPER_PEAKS_NM = (95, 190, 285)         # experimental SSD peaks (Fig. 3A)


def main():
    print("Running exact 3D Monte Carlo (paper geometry, dt = 1 us) ...")
    p = Params3D(p_on=P_ON_KON100, n_particles=3000, total_time=0.3, seed=1)
    r = simulate3d(p)
    print(f"  bound fraction = {r.bound_fraction:.3f}  "
          f"(k*on/koff = {r.kon_koff:.0f}; paper ~100)")
    print(f"  residence mean = {1e3*r.residence_times.mean():.0f} ms "
          f"(censored; -> 40 ms for long runs)")

    ss_all = r.step_size_distribution() * 1e3
    ss_hop = r.step_size_distribution(hops_only=True) * 1e3
    sp = p.mt_spacing * 1e3

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ---- (left) bundle cross-section ----------------------------------------
    ax[0].add_patch(Circle((0, 0), p.R_tube * 1e3, fill=False,
                           color="gray", lw=1.5, ls="--"))
    for (cx, cy) in p.mt_xy * 1e3:
        ax[0].add_patch(Circle((cx, cy), p.R_mt * 1e3, color=BLUE, alpha=0.8))
    ax[0].set_xlim(-p.R_tube * 1e3 * 1.05, p.R_tube * 1e3 * 1.05)
    ax[0].set_ylim(-p.R_tube * 1e3 * 1.05, p.R_tube * 1e3 * 1.05)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x (nm)"); ax[0].set_ylabel("y / depth (nm)")
    ax[0].set_title(f"{p.n_mt} MTs, {sp:.0f} nm spacing, R$_{{tube}}$ = "
                    f"{p.R_tube*1e3:.0f} nm")

    # ---- (right) step-size distribution -------------------------------------
    bins = np.linspace(0, 360, 100)
    ax[1].hist(ss_all, bins=bins, density=True, color=BLUE, alpha=0.45,
               label="all localisations")
    ax[1].hist(ss_hop, bins=bins, density=True, color=ORANGE, alpha=0.7,
               label="inter-MT hops only")
    for k, pk in enumerate(PAPER_PEAKS_NM):
        ax[1].axvline(pk, color=RED, ls="--", alpha=0.7,
                      label="paper Fig. 3A peaks" if k == 0 else None)
    for k in (1, 2, 3, 4):
        ax[1].axvline(k * sp, color=GREEN, ls=":", alpha=0.6,
                      label=f"n x {sp:.0f} nm (input spacing)" if k == 1 else None)
    ax[1].set_yscale("log"); ax[1].set_ylim(1e-4, 0.05)
    ax[1].set_xlabel("step size in $\\Delta t$ = 5 ms (nm)")
    ax[1].set_ylabel("probability density")
    ax[1].set_title(f"Exact-3D SSD at k*on/koff = {r.kon_koff:.0f}")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig6_exact_3d.png"))
    plt.close(fig)

    # update / extend the summary
    spath = os.path.join(DATADIR, "summary.json")
    summary = {}
    if os.path.exists(spath):
        summary = json.load(open(spath))
    summary["exact_3d"] = {
        "geometry": "tube L=100um R=500nm; 60 hex MTs R_MT=12.5nm spacing=70nm",
        "dt_s": p.dt,
        "p_on": p.p_on,
        "bound_fraction": float(r.bound_fraction),
        "kon_koff": float(r.kon_koff),
        "residence_mean_ms_censored": float(1e3 * r.residence_times.mean()),
        "note": "2D-projected SSD is multi-modal (noise peak + inter-MT hop "
                "tail through multiples of the spacing); the cleaner separated "
                "peaks of the published Fig. 3A reflect the specific, more "
                "ordered real bundle (~95 nm spacing in that cell).",
    }
    json.dump(summary, open(spath, "w"), indent=2)
    np.savetxt(os.path.join(DATADIR, "ssd_3d_hops_nm.csv"), ss_hop,
               header="step_size_nm", comments="")
    print(f"\nFigure -> {os.path.join(FIGDIR, 'fig6_exact_3d.png')}")


if __name__ == "__main__":
    main()
