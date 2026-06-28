"""
Reproduce the key figures of Janning et al. (2014, MBoC) from the kiss-and-hop
Monte Carlo simulation, using the parameters reported in the paper.

Figures (-> figures/) and data (-> data/):
  1. residence-time / dwell-time distribution  (paper Fig. 4) -- measured the
     paper's way: consecutive frames within a 50 nm radius -> single
     exponential, mean ~ 40 ms.
  2. bound fraction and the pseudo-equilibrium constant k*on/koff.
  3. step-size distribution, SSD            (paper Fig. 3) -- peaks at multiples
     of the MT spacing = the signature of hopping between microtubules, and the
     SSD at several k*on/koff (the paper matches the data at k*on/koff ~ 100).
  4. MSD / effective longitudinal diffusion -- tiny because tau is ~99 % bound,
     i.e. tau does not run ahead of (or impede) axonal transport.
  5. reaction-diffusion / effective-diffusion relation linking the bound
     fraction to D_eff = (1-f_b) D_free + f_b D_bound (Igaev et al. 2014).

Paper parameters used here (Janning et al. 2014, Materials and Methods):
  D_free(PAGFP-htau) = 14.4 um^2/s,  MT spacing = 70 nm,  dwell ~ 40 ms,
  camera frame = 5 ms,  localisation precision ~ 20 nm,  simulation dt = 1 us
  (here 10 us for speed),  k*on/koff ~ 100  (=> ~99 % of tau bound).

Run:  python3 simulation/run_experiments.py
"""

from __future__ import annotations

import json
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kiss_and_hop import Params, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(ROOT, "figures")
DATADIR = os.path.join(ROOT, "data")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)

# Values from Janning et al. 2014 (Materials and Methods)
TAU_DWELL_PAPER = 0.040       # s   single-molecule dwell time (~40 ms)
D_FREE_PAPER = 14.4           # um^2/s  free PAGFP-htau diffusion constant
D_3xPAGFP_TRACER = 13.9       # um^2/s  free-diffusion control (3xPAGFP), NOT tau
MT_SPACING_PAPER = 0.070      # um  average MT spacing in PC12 neurites
P_BIND_KON100 = 0.08          # on-rate giving k*on/koff ~ 100 (~99% bound)

plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
BLUE, RED, GREEN, ORANGE = "#2b6cb0", "#c53030", "#2f855a", "#dd6b20"


# ---------------------------------------------------------------------------
# 1. Dwell-time distribution (paper Fig. 4)
# ---------------------------------------------------------------------------
def fig_dwell(res, summary):
    # the paper's method: consecutive frames within a 50 nm radius
    dwell = res.dwell_time_colocalization(radius=0.050) * 1e3      # ms
    dwell = dwell[dwell > 0]
    tau_coloc = dwell.mean()
    # the true per-MT residence times for comparison
    rt = res.residence_times * 1e3
    tau_true = rt.mean()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    bins = np.linspace(0, np.percentile(dwell, 99.5), 45)
    ax[0].hist(dwell, bins=bins, density=True, color=BLUE, alpha=0.55,
               label=f"colocalisation dwell (50 nm, 5 ms frames)\nN={dwell.size}")
    xs = np.linspace(0, bins[-1], 300)
    ax[0].plot(xs, np.exp(-xs / tau_coloc) / tau_coloc, color=RED, lw=2,
               label=f"single-exp, $\\tau$ = {tau_coloc:.0f} ms")
    ax[0].set_xlabel("dwell time (ms)")
    ax[0].set_ylabel("probability density")
    ax[0].set_title("Dwell time as measured in the paper (Fig. 4)")
    ax[0].legend(fontsize=8)

    # log-survival: a single exponential is a straight line
    for data, c, lab in ((rt, GREEN, f"true per-MT residence, $\\tau$={tau_true:.0f} ms"),
                          (dwell, BLUE, f"50 nm colocalisation, $\\tau$={tau_coloc:.0f} ms")):
        s = np.sort(data); surv = 1 - np.arange(s.size) / s.size
        ax[1].semilogy(s, surv, color=c, lw=1.5, label=lab)
    ax[1].set_xlabel("time (ms)"); ax[1].set_ylabel("survival fraction")
    ax[1].set_ylim(1e-3, 1); ax[1].set_title("Single-exponential residence")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_dwell_time.png"))
    plt.close(fig)

    summary["dwell_coloc_ms"] = float(tau_coloc)
    summary["residence_true_ms"] = float(tau_true)
    np.savetxt(os.path.join(DATADIR, "dwell_times_ms.csv"), dwell,
               header="dwell_time_ms", comments="")


# ---------------------------------------------------------------------------
# 2. Bound fraction + k*on/koff
# ---------------------------------------------------------------------------
def fig_bound_fraction(res, summary):
    fb_t = res.bound.mean(axis=0)
    fb = res.bound_fraction
    tau_free = res.free_times.mean()
    fb_analytic = res.params.tau_dwell / (res.params.tau_dwell + tau_free)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res.t * 1e3, fb_t, color=BLUE, lw=1.0, alpha=0.8,
            label="MC, instantaneous")
    ax.axhline(fb, color=RED, ls="--",
               label=f"time average = {fb:.3f}  (k*on/koff = {res.kon_koff:.0f})")
    ax.axhline(0.99, color=GREEN, ls=":",
               label="paper: ~0.99 bound (k*on/koff ~ 100)")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("fraction of tau bound to MTs")
    ax.set_ylim(0.9, 1.005)
    ax.set_title("Tau is ~99 % MT-bound despite 40 ms contacts")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_bound_fraction.png"))
    plt.close(fig)

    summary["bound_fraction"] = float(fb)
    summary["kon_koff"] = float(res.kon_koff)
    summary["mean_free_time_ms"] = float(tau_free * 1e3)


# ---------------------------------------------------------------------------
# 3. Step-size distribution, SSD (paper Fig. 3) -- THE Monte Carlo figure
# ---------------------------------------------------------------------------
def fig_ssd(base: Params, summary):
    sp = base.mt_spacing * 1e3                                   # nm
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    # (A) SSD at the matching k*on/koff ~ 100, with the hop peaks annotated
    r = simulate(p_bind=P_BIND_KON100, mt_spacing=base.mt_spacing,
                 D_free=base.D_free, D_bound=base.D_bound,
                 tau_dwell=base.tau_dwell, total_time=0.6,
                 n_molecules=4000, seed=11)
    ss = r.step_size_distribution() * 1e3                       # nm
    ss_hop = r.step_size_distribution(hops_only=True) * 1e3     # inter-MT jumps
    bins = np.linspace(0, 5.5 * sp, 90)
    ax[0].hist(ss, bins=bins, density=True, color=BLUE, alpha=0.5,
               label="all steps")
    ax[0].hist(ss_hop, bins=bins, density=True, color=ORANGE, alpha=0.7,
               label="steps with an inter-MT hop")
    for k in (1, 2, 3, 4):
        ax[0].axvline(k * sp, color=RED, ls="--", alpha=0.6)
        ax[0].text(k * sp, 0.92, f"1$\\to${1+k}", color=RED, ha="center",
                   fontsize=8, transform=ax[0].get_xaxis_transform())
    ax[0].set_xlabel("step size in $\\Delta t$ = 5 ms (nm)")
    ax[0].set_ylabel("probability density")
    ax[0].set_yscale("log"); ax[0].set_ylim(1e-4, 0.06)
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title(f"SSD at k*on/koff = {r.kon_koff:.0f}  (peaks at n x {sp:.0f} nm)")

    # (B) SSD at several k*on/koff (paper Fig. 3B): the hop tail grows as tau
    # spends more time unbound (lower k*on/koff)
    for pb, c in ((0.01, GREEN), (0.08, BLUE), (1.0, ORANGE)):
        rr = simulate(p_bind=pb, mt_spacing=base.mt_spacing, D_free=base.D_free,
                      D_bound=base.D_bound, tau_dwell=base.tau_dwell,
                      total_time=0.4, n_molecules=2500, seed=12)
        s2 = rr.step_size_distribution() * 1e3
        h, edges = np.histogram(s2, bins=bins, density=True)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        ax[1].plot(ctr, h, color=c, lw=1.6,
                   label=f"k*on/koff = {rr.kon_koff:.0f}")
    for k in (1, 2, 3):
        ax[1].axvline(k * sp, color="gray", ls=":", alpha=0.5)
    ax[1].set_xlabel("step size in $\\Delta t$ = 5 ms (nm)")
    ax[1].set_ylabel("probability density")
    ax[1].set_yscale("log"); ax[1].set_ylim(1e-4, 0.06)
    ax[1].set_title("SSD vs k*on/koff (paper Fig. 3B)")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_step_size_distribution.png"))
    plt.close(fig)

    summary["ssd_kon100_kon_koff"] = float(r.kon_koff)
    np.savetxt(os.path.join(DATADIR, "ssd_kon100_nm.csv"), ss,
               header="step_size_nm", comments="")


# ---------------------------------------------------------------------------
# 4. MSD / effective diffusion
# ---------------------------------------------------------------------------
def fig_msd(res, summary):
    lag, msd = res.msd()
    D_eff = res.effective_diffusion()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lag * 1e3, msd * 1e6, "o", ms=3, color=BLUE, label="MC MSD")
    ax.plot(lag * 1e3, 2 * D_eff * lag * 1e6, color=RED, lw=2,
            label=f"2$D_{{eff}}$t,  $D_{{eff}}$={D_eff:.3f} um$^2$/s")
    ax.set_xlabel("lag time (ms)")
    ax.set_ylabel("MSD along axis (x10$^{-3}$ um$^2$)")
    ax.set_title("Tiny longitudinal $D_{eff}$: tau stays put between hops")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_msd.png"))
    plt.close(fig)

    summary["D_eff_from_msd"] = float(D_eff)
    np.savetxt(os.path.join(DATADIR, "msd.csv"),
               np.column_stack([lag, msd]),
               header="lag_s,msd_um2", delimiter=",", comments="")


# ---------------------------------------------------------------------------
# 5. Effective diffusion vs bound fraction (Igaev 2014 two-state relation)
# ---------------------------------------------------------------------------
def fig_effective_diffusion(base: Params, summary):
    p_binds = [0.004, 0.008, 0.02, 0.05, 0.12, 0.3, 0.7, 1.0]
    f_bound, D_eff = [], []
    for pb in p_binds:
        r = simulate(p_bind=float(pb), mt_spacing=base.mt_spacing,
                     D_free=base.D_free, D_bound=base.D_bound,
                     tau_dwell=base.tau_dwell, total_time=0.4,
                     n_molecules=900, seed=5)
        f_bound.append(r.bound_fraction); D_eff.append(r.effective_diffusion())
    f_bound, D_eff = np.array(f_bound), np.array(D_eff)

    fb = np.linspace(0, 1, 100)
    theory = (1 - fb) * base.D_free + fb * base.D_bound

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(f_bound, D_eff, "o", ms=7, color=BLUE, label="MC (vary on-rate)")
    ax.plot(fb, theory, color=RED, lw=2,
            label="$(1-f_b)D_{free}+f_b D_{bound}$  (Igaev 2014)")
    ax.axvline(0.99, color=GREEN, ls=":", label="physiological $f_b$ ~ 0.99")
    ax.annotate("free tau\n$D_{free}$ = 14.4", (0.02, base.D_free),
                fontsize=8, color=RED, va="top")
    ax.set_xlabel("bound fraction $f_b$")
    ax.set_ylabel("effective diffusion (um$^2$/s)")
    ax.set_title("At $f_b$ ~ 0.99 the effective transport of tau is ~0")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig5_Deff_vs_bound_fraction.png"))
    plt.close(fig)

    np.savetxt(os.path.join(DATADIR, "Deff_vs_bound_fraction.csv"),
               np.column_stack([p_binds, f_bound, D_eff]),
               header="p_bind,bound_fraction,D_eff_um2_s",
               delimiter=",", comments="")


# ---------------------------------------------------------------------------
def main():
    print("Running main kiss-and-hop simulation (paper parameters) ...")
    base = Params(tau_dwell=TAU_DWELL_PAPER, mt_spacing=MT_SPACING_PAPER,
                  D_free=D_FREE_PAPER, D_bound=0.0, p_bind=P_BIND_KON100,
                  capture_radius=0.0125, process_width=1.0,
                  frame_interval=0.005, loc_precision=0.020,
                  n_molecules=3000, total_time=1.0, dt=1e-5,
                  record_stride=25, seed=0)
    res = simulate(base)

    summary = {
        "paper": "Janning et al. 2014, Mol Biol Cell 25:3541 "
                 "(doi:10.1091/mbc.e14-06-1099)",
        "inputs": {
            "tau_dwell_ms": base.tau_dwell * 1e3,
            "D_free_um2_s": base.D_free,
            "D_bound_um2_s": base.D_bound,
            "mt_spacing_nm": base.mt_spacing * 1e3,
            "capture_radius_nm": base.capture_radius * 1e3,
            "frame_interval_ms": base.frame_interval * 1e3,
            "loc_precision_nm": base.loc_precision * 1e3,
            "p_bind": base.p_bind,
            "dt_s": base.dt,
            "n_molecules": base.n_molecules,
            "total_time_s": base.total_time,
        },
        "note_3xPAGFP_tracer_um2_s": D_3xPAGFP_TRACER,
    }

    fig_dwell(res, summary)
    fig_bound_fraction(res, summary)
    print("Reproducing step-size distribution (Fig. 3) ...")
    fig_ssd(base, summary)
    fig_msd(res, summary)
    print("Running on-rate sweep (effective diffusion) ...")
    fig_effective_diffusion(base, summary)

    with open(os.path.join(DATADIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(f"  dwell (50 nm coloc.)   : {summary['dwell_coloc_ms']:.0f} ms "
          f"(paper ~ {TAU_DWELL_PAPER*1e3:.0f} ms)")
    print(f"  true per-MT residence  : {summary['residence_true_ms']:.0f} ms")
    print(f"  bound fraction         : {summary['bound_fraction']:.3f}  "
          f"(k*on/koff = {summary['kon_koff']:.0f}; paper ~100)")
    print(f"  D_eff (MSD)            : {summary['D_eff_from_msd']:.3f} um^2/s")
    print(f"\nFigures -> {FIGDIR}\nData    -> {DATADIR}")


if __name__ == "__main__":
    main()
