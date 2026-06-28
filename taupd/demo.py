"""
taupd proof-of-concept demo.

(A) Recovery / identifiability: simulate noisy FDAP curves with a known bound
    fraction, fit them back, and show the recovered bound fraction tracks the
    truth with honest confidence intervals.
(B) Robustness / QC: the same, but with unmodelled photobleaching in the data --
    showing how a common artefact biases the readout (why bleaching QC matters).
(C) Pharmacodynamics: a compound that raises the bound fraction with dose; the
    tool fits each dose, computes the free-tau pool (aggregation substrate) and
    its EC50.

Run:  python3 -m taupd.demo      (from the repo root)
"""

from __future__ import annotations
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .assay import simulate_fdap
from .inverse import fit_fdap
from .model import free_tau_pool
from .pd import hill_dose_response, classify_moa

# tau parameters (Janning/Igaev 2014)
D_FREE = 14.4          # um^2/s
SIGMA = 2.5            # um   activation half-width (~5 um spot)
NOISE = 0.02
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)
BLUE, RED, GREEN, ORANGE = "#2b6cb0", "#c53030", "#2f855a", "#dd6b20"


def recovery(tau_bleach=None, seed0=0):
    truths = np.linspace(0.30, 0.97, 12)
    rec, err = [], []
    for i, fb in enumerate(truths):
        t, F = simulate_fdap(fb, D_FREE, SIGMA, noise=NOISE, n_cells=8,
                             tau_bleach=tau_bleach, seed=seed0 + i)
        r = fit_fdap(t, F, SIGMA, D_FREE, model="effective")
        rec.append(r.derived["bound_fraction"])
        err.append(r.derived["bound_fraction_ci68"])
    return truths, np.array(rec), np.array(err)


def main():
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))

    # (A) clean recovery
    tr, rc, er = recovery(tau_bleach=None, seed0=0)
    ax[0].plot([0, 1], [0, 1], color="gray", ls="--")
    ax[0].errorbar(tr, rc, yerr=er, fmt="o", color=BLUE, capsize=3)
    ax[0].set_xlabel("true bound fraction")
    ax[0].set_ylabel("recovered bound fraction")
    ax[0].set_title("(A) Identifiability: clean recovery")
    ax[0].set_xlim(0.2, 1.0); ax[0].set_ylim(0.2, 1.0)

    # (B) robustness to unmodelled photobleaching
    trb, rcb, erb = recovery(tau_bleach=2.0, seed0=100)   # 2 s bleach lifetime
    ax[1].plot([0, 1], [0, 1], color="gray", ls="--")
    ax[1].errorbar(trb, rcb, yerr=erb, fmt="s", color=ORANGE, capsize=3)
    ax[1].set_xlabel("true bound fraction")
    ax[1].set_ylabel("recovered bound fraction")
    ax[1].set_title("(B) QC: unmodelled bleaching biases the readout")
    ax[1].set_xlim(0.2, 1.0); ax[1].set_ylim(0.2, 1.0)

    # (C) dose-response: compound raises bound fraction 0.50 -> 0.95, EC50=100 nM
    doses = np.array([0, 1, 3, 10, 30, 100, 300, 1000, 3000])   # nM
    ec50_true, fb_lo, fb_hi, hill = 100.0, 0.50, 0.95, 1.3
    tau_total = 1.0                                             # arbitrary units
    fb_true = fb_lo + (fb_hi - fb_lo) / (1 + (ec50_true / np.maximum(doses, 1e-9)) ** hill)
    free_pool, free_err = [], []
    for i, fb in enumerate(fb_true):
        t, F = simulate_fdap(fb, D_FREE, SIGMA, noise=NOISE, n_cells=8, seed=500 + i)
        r = fit_fdap(t, F, SIGMA, D_FREE, tau_total=tau_total, model="effective")
        free_pool.append(r.derived["free_tau_pool"])
        free_err.append(r.derived["free_tau_pool_ci68"])
    free_pool = np.array(free_pool)
    dr = hill_dose_response(doses, free_pool, response_name="free tau pool")

    dd = np.logspace(0, 3.6, 200)
    from taupd.pd import _hill
    ax[2].errorbar(np.maximum(doses, 0.5), free_pool, yerr=free_err, fmt="o",
                   color=GREEN, capsize=3, label="fitted free pool")
    ax[2].plot(dd, _hill(dd, dr.bottom, dr.top, dr.ec50, dr.hill_n),
               color=RED, lw=2, label=f"Hill fit, EC50={dr.ec50:.0f} nM")
    ax[2].axvline(ec50_true, color="gray", ls=":", label=f"true EC50={ec50_true:.0f} nM")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("compound dose (nM)")
    ax[2].set_ylabel("free tau pool (a.u.)")
    ax[2].set_title("(C) Dose-response on the aggregation substrate")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "taupd_demo.png")
    fig.savefig(out)
    plt.close(fig)

    # MoA call: vehicle (dose 0) vs saturating dose
    t, F0 = simulate_fdap(fb_true[0], D_FREE, SIGMA, noise=NOISE, n_cells=8, seed=9)
    t, F1 = simulate_fdap(fb_true[-1], D_FREE, SIGMA, noise=NOISE, n_cells=8, seed=10)
    base = fit_fdap(t, F0, SIGMA, D_FREE, tau_total=tau_total)
    treat = fit_fdap(t, F1, SIGMA, D_FREE, tau_total=tau_total)
    moa = classify_moa(base, treat)

    print("=== taupd P0 demo ===")
    print(f"(A) clean recovery: mean |recovered-true| = {np.mean(np.abs(rc-tr)):.3f}")
    print(f"(B) with bleaching: mean bias = {np.mean(rcb-trb):+.3f} "
          f"(systematic, -> bleaching QC required)")
    print(f"(C) {dr.summary()}  [true EC50 = {ec50_true:.0f} nM]")
    print(f"MoA call (vehicle vs saturating): {moa['call']}")
    print(f"   delta bound fraction = {moa['delta_bound_fraction']:+.3f}")
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
