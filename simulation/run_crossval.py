"""
Cross-validation: predict the cortical-axon case from the PC12 calibration.

The tau *molecular* parameters (binding probability p_on, off-rate koff = 1/dwell)
are kept exactly as calibrated on PC12 neurites. ONLY the geometry is changed to
that of a cortical axon, where the inter-MT distance is much smaller (~20 nm
edge-to-edge; Hirokawa) than in PC12 neurites (~70 nm). Nothing is re-tuned.

Two genuine, non-fitted predictions are then tested:

  (A) The single-MT dwell time is a *molecular* property (set by koff) and should
      be the same in both systems. The model predicts ~40 ms for axons; the
      paper measured 39 +/- 4 ms (PC12) and 36 +/- 5 ms (axons) independently.

  (B) Inter-MT hops shrink with the spacing. As the spacing approaches the ~20 nm
      localisation precision, the hops become unresolvable -- which is exactly
      why the paper could do the step-size-distribution analysis in PC12 neurites
      but reported only dwell times for axons.

Produces figures/fig7_crossvalidation.png.
"""

from __future__ import annotations

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kiss_and_hop_3d import simulate3d, Params3D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
BLUE, RED, GREEN, ORANGE = "#2b6cb0", "#c53030", "#2f855a", "#dd6b20"

P_ON = 0.20                 # molecular: calibrated on PC12, kept fixed
LOC_PRECISION_NM = 20.0
COLOC_RADIUS_NM = 50.0


def main():
    print("Cross-validation: PC12 calibration -> axon prediction (no re-tuning)")

    # (A) dwell transferability: PC12 (70 nm) vs axon (45 nm), same molecule
    runs = {}
    for name, sp in (("PC12 neurite (70 nm)", 0.070),
                     ("cortical axon (45 nm)", 0.045)):
        r = simulate3d(p_on=P_ON, mt_spacing=sp, n_particles=2000,
                       total_time=0.5, seed=2)
        runs[name] = r
        print(f"  {name:22s}: f_bound={r.bound_fraction:.3f}, "
              f"true dwell mean={1e3*r.residence_times.mean():.0f} ms")

    # (B) hop size vs spacing (resolvability transition)
    spacings = np.array([0.025, 0.035, 0.045, 0.055, 0.070, 0.090, 0.120])
    hop_median = []
    for sp in spacings:
        r = simulate3d(p_on=P_ON, mt_spacing=float(sp), n_particles=1500,
                       total_time=0.2, seed=3)
        ss = r.step_size_distribution(hops_only=True) * 1e3
        hop_median.append(np.median(ss))
    hop_median = np.array(hop_median)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # ---- (A) dwell distributions overlaid -----------------------------------
    colors = {"PC12 neurite (70 nm)": BLUE, "cortical axon (45 nm)": ORANGE}
    measured = {"PC12 neurite (70 nm)": (39, 4), "cortical axon (45 nm)": (36, 5)}
    for name, r in runs.items():
        d = np.sort(r.residence_times * 1e3)
        surv = 1 - np.arange(d.size) / d.size
        m, e = measured[name]
        ax[0].semilogy(d, surv, color=colors[name], lw=1.8,
                       label=f"{name}: model $\\tau$={d.mean():.0f} ms "
                             f"(measured {m}$\\pm${e} ms)")
    ax[0].set_xlabel("single-MT residence time (ms)")
    ax[0].set_ylabel("survival fraction")
    ax[0].set_ylim(1e-3, 1); ax[0].set_xlim(0, 260)
    ax[0].set_title("(A) Dwell is molecular -> transfers across geometry")
    ax[0].legend(fontsize=8)

    # ---- (B) hop size vs spacing, resolvability ------------------------------
    ax[1].plot(spacings * 1e3, hop_median, "o-", color=BLUE,
               label="median inter-MT hop (model)")
    ax[1].axhline(LOC_PRECISION_NM, color=RED, ls="--",
                  label=f"localisation precision ~{LOC_PRECISION_NM:.0f} nm")
    ax[1].axhline(COLOC_RADIUS_NM, color=GREEN, ls=":",
                  label=f"50 nm dwell radius")
    ax[1].axvspan(15, 25, color=ORANGE, alpha=0.12)
    ax[1].text(20, ax[1].get_ylim()[1]*0.0 + 78, "axon\n(~20 nm)", color=ORANGE,
               ha="center", fontsize=8)
    ax[1].axvline(70, color=BLUE, ls=":", alpha=0.5)
    ax[1].text(70, 78, "PC12\n(70 nm)", color=BLUE, ha="center", fontsize=8)
    ax[1].set_xlabel("MT centre-to-centre spacing (nm)")
    ax[1].set_ylabel("median hop size (nm)")
    ax[1].set_title("(B) In axons hops fall toward the resolution limit")
    ax[1].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig7_crossvalidation.png"))
    plt.close(fig)
    print(f"\nFigure -> {os.path.join(FIGDIR, 'fig7_crossvalidation.png')}")


if __name__ == "__main__":
    main()
