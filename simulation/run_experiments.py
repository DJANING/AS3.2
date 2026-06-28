"""
Reproduce the key observables of Janning et al. (2014, MBoC) from the
kiss-and-hop Monte Carlo simulation and save figures + summary data.

Produces (in figures/ and data/):
  1. residence-time distribution  -> single exponential, mean ~ 40 ms,
     and the "observed" distribution after frame-rate binning + photobleaching
     (the experimental recovery of the dwell time).
  2. bound fraction               -> approach to the stationary state + value.
  3. FDAP transient               -> fluorescence-decay-after-photoactivation
     curve and the effective diffusion constant from a reaction-diffusion fit.
  4. MSD / effective diffusion    -> short-time (free) vs long-time (effective)
     longitudinal diffusion.
  5. parameter sweep              -> D_eff and bound fraction vs MT spacing,
     relating the model to the measured D_eff ~ 13.9 um^2/s (Igaev et al. 2014).

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

# Measured value from the companion FDAP study (Igaev et al. 2014, Biophys J)
D_EFF_MEASURED = 13.9          # um^2/s
TAU_DWELL_PAPER = 0.040        # s   (~40 ms single-molecule residence time)

plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
BLUE, RED, GREEN = "#2b6cb0", "#c53030", "#2f855a"


# ---------------------------------------------------------------------------
# 1. Residence-time distribution
# ---------------------------------------------------------------------------
def fig_residence(res, summary):
    rt = res.residence_times * 1e3                      # ms
    tau_true = rt.mean()

    # "Experimental" observation: finite camera frame time + photobleaching.
    # Each true dwell is competed against an exponential bleaching lifetime and
    # then discretised to the camera frame interval. Bleached intervals are
    # right-censored (the molecule disappears before it detaches).
    frame_ms = 5.0                                       # camera frame interval
    tau_bleach_ms = 250.0                                # mean bleaching lifetime
    rng = np.random.default_rng(1)
    bleach = rng.exponential(tau_bleach_ms, size=rt.size)
    observed = np.minimum(rt, bleach)
    not_bleached = rt <= bleach
    observed = np.ceil(observed / frame_ms) * frame_ms   # frame discretisation
    # MLE of an exponential with right-censoring (Kaplan-Meier-free estimator):
    # tau_hat = sum(observed) / number_of_uncensored_events
    tau_obs = observed.sum() / max(1, not_bleached.sum())

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # linear histogram + exponential pdf
    bins = np.linspace(0, rt.max(), 60)
    ax[0].hist(rt, bins=bins, density=True, color=BLUE, alpha=0.55,
               label=f"MC residence times (N={rt.size})")
    xs = np.linspace(0, rt.max(), 300)
    ax[0].plot(xs, np.exp(-xs / tau_true) / tau_true, color=RED, lw=2,
               label=f"single-exp fit, $\\tau$ = {tau_true:.1f} ms")
    ax[0].set_xlabel("residence time on one MT (ms)")
    ax[0].set_ylabel("probability density")
    ax[0].set_title("Single-MT residence is single-exponential")
    ax[0].legend()

    # log-survival (a single exponential is a straight line here)
    s = np.sort(rt)
    surv = 1.0 - np.arange(s.size) / s.size
    ax[1].semilogy(s, surv, color=BLUE, lw=1.5, label="MC survival")
    ax[1].semilogy(xs, np.exp(-xs / tau_true), color=RED, ls="--",
                   label=f"exp, $\\tau$={tau_true:.1f} ms")
    # observed (frame-binned + bleaching)
    so = np.sort(observed)
    survo = 1.0 - np.arange(so.size) / so.size
    ax[1].semilogy(so, survo, color=GREEN, lw=1.5,
                   label=f"observed (5 ms frames + bleach), $\\tau$={tau_obs:.1f} ms")
    ax[1].set_xlabel("residence time (ms)")
    ax[1].set_ylabel("survival fraction")
    ax[1].set_title("Recovered dwell time vs experimental limits")
    ax[1].set_ylim(1e-3, 1)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_residence_time.png"))
    plt.close(fig)

    summary["residence_mean_ms"] = float(tau_true)
    summary["residence_observed_ms"] = float(tau_obs)
    summary["residence_events"] = int(rt.size)
    np.savetxt(os.path.join(DATADIR, "residence_times_ms.csv"), rt,
               header="residence_time_ms", comments="")


# ---------------------------------------------------------------------------
# 2. Bound fraction
# ---------------------------------------------------------------------------
def fig_bound_fraction(res, summary):
    fb_t = res.bound.mean(axis=0)
    fb = res.bound_fraction
    # analytic stationary value from the two-state average
    tau_free = res.free_times.mean()
    fb_analytic = res.params.tau_dwell / (res.params.tau_dwell + tau_free)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res.t * 1e3, fb_t, color=BLUE, lw=1.2, label="MC, instantaneous")
    ax.axhline(fb, color=RED, ls="--",
               label=f"time average = {fb:.3f}")
    ax.axhline(fb_analytic, color=GREEN, ls=":",
               label=f"$\\tau_d/(\\tau_d+\\tau_{{assoc}})$ = {fb_analytic:.3f}")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("fraction of tau bound to MTs")
    ax.set_ylim(0, 1.02)
    ax.set_title("Tau is mostly MT-associated despite 40 ms contacts")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_bound_fraction.png"))
    plt.close(fig)

    summary["bound_fraction"] = float(fb)
    summary["bound_fraction_analytic"] = float(fb_analytic)
    summary["mean_free_time_ms"] = float(tau_free * 1e3)


# ---------------------------------------------------------------------------
# 3. FDAP transient
# ---------------------------------------------------------------------------
def fig_fdap(res, summary):
    """Fluorescence Decay After Photoactivation.

    All molecules are 'activated' at t=0 at x=0. The fluorescence inside an
    activation window [-w, w] decays as molecules diffuse out longitudinally.
    In the effective-diffusion limit x(t) ~ Normal(0, 2 D_eff t), so

        F(t) = P(|x| < w) = erf( w / sqrt(4 D_eff t) ).
    """
    x = res.x
    # choose the activation half-width from the spread so the curve is informative
    w = float(2.0 * np.sqrt(2.0 * res.effective_diffusion() * res.t[-1] + 1e-9))
    w = max(w, 0.3)
    F = (np.abs(x) < w).mean(axis=0)
    t = res.t.copy()

    # fit the reaction-diffusion (effective diffusion) curve, skip t=0
    m = t > 0
    def fdap_model(tt, D):
        return erf(w / np.sqrt(4.0 * D * tt))
    popt, _ = curve_fit(fdap_model, t[m], F[m], p0=[res.effective_diffusion()],
                        bounds=(1e-4, 1e4))
    D_fit = float(popt[0])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t * 1e3, F, "o", ms=3, color=BLUE, alpha=0.6,
            label=f"MC FDAP (window $\\pm${w:.2f} um)")
    ax.plot(t[m] * 1e3, fdap_model(t[m], D_fit), color=RED, lw=2,
            label=f"reaction-diffusion fit\n$D_{{eff}}$ = {D_fit:.2f} um$^2$/s")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("normalised fluorescence in window")
    ax.set_title("FDAP transient and effective diffusion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_fdap.png"))
    plt.close(fig)

    summary["fdap_window_um"] = w
    summary["D_eff_from_fdap"] = D_fit


# ---------------------------------------------------------------------------
# 4. MSD / effective diffusion
# ---------------------------------------------------------------------------
def fig_msd(res, summary):
    lag, msd = res.msd()
    D_eff = res.effective_diffusion()
    # short-time slope (first few lags) reflects the free-diffusion excursions
    k = max(3, lag.size // 10)
    D_short = float(np.polyfit(lag[:k], msd[:k], 1)[0] / 2.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lag * 1e3, msd, "o", ms=3, color=BLUE, label="MC MSD")
    ax.plot(lag * 1e3, 2 * D_eff * lag, color=RED, lw=2,
            label=f"2$D_{{eff}}$t,  $D_{{eff}}$={D_eff:.2f} um$^2$/s")
    ax.set_xlabel("lag time (ms)")
    ax.set_ylabel("MSD along axis (um$^2$)")
    ax.set_title("Longitudinal MSD -> effective diffusion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_msd.png"))
    plt.close(fig)

    summary["D_eff_from_msd"] = float(D_eff)
    summary["D_short_time"] = D_short
    np.savetxt(os.path.join(DATADIR, "msd.csv"),
               np.column_stack([lag, msd]),
               header="lag_s,msd_um2", delimiter=",", comments="")


# ---------------------------------------------------------------------------
# 5. Parameter sweep: how D_eff and bound fraction depend on MT spacing
# ---------------------------------------------------------------------------
def fig_sweep(base: Params, summary):
    spacings = np.array([0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0])
    D_eff, f_bound = [], []
    for sp in spacings:
        r = simulate(mt_spacing=float(sp), total_time=0.4, n_molecules=800,
                     D_free=base.D_free, D_bound=base.D_bound,
                     tau_dwell=base.tau_dwell, seed=3)
        D_eff.append(r.effective_diffusion())
        f_bound.append(r.bound_fraction)
    D_eff, f_bound = np.array(D_eff), np.array(f_bound)

    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(spacings * 1e3, D_eff, "o-", color=BLUE, label="$D_{eff}$ (MC)")
    ax1.axhline(D_EFF_MEASURED, color=GREEN, ls="--",
                label=f"measured $D_{{eff}}$ = {D_EFF_MEASURED} um$^2$/s")
    ax1.set_xlabel("MT centre-to-centre spacing (nm)")
    ax1.set_ylabel("effective diffusion (um$^2$/s)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_title("Sparser MTs -> faster effective diffusion")

    ax2 = ax1.twinx()
    ax2.plot(spacings * 1e3, f_bound, "s--", color=RED, label="bound fraction")
    ax2.set_ylabel("bound fraction", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(0, 1.02)
    ax2.grid(False)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right",
               fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig5_sweep_spacing.png"))
    plt.close(fig)

    np.savetxt(os.path.join(DATADIR, "sweep_spacing.csv"),
               np.column_stack([spacings, D_eff, f_bound]),
               header="mt_spacing_um,D_eff_um2_s,bound_fraction",
               delimiter=",", comments="")


# ---------------------------------------------------------------------------
# 6. Effective diffusion vs bound fraction (the two-state master relation)
# ---------------------------------------------------------------------------
def fig_effective_diffusion_regime(base: Params, summary):
    """Vary the on-rate (binding probability) to span the bound fraction and
    show that the longitudinal effective diffusion follows the two-state law

        D_eff = (1 - f_bound) * D_free + f_bound * D_bound .

    The densely bound single-molecule regime (f_bound ~ 0.99, D_eff ~ 0.2)
    and the FDAP regime that yields the measured D_eff ~ 13.9 um^2/s
    (f_bound ~ 0.3) are two ends of the same curve.
    """
    p_binds = [0.0015, 0.003, 0.006, 0.012, 0.03, 0.08, 0.2, 0.5, 1.0]
    f_bound, D_eff = [], []
    for pb in p_binds:
        r = simulate(p_bind=float(pb), mt_spacing=base.mt_spacing,
                     D_free=base.D_free, D_bound=base.D_bound,
                     tau_dwell=base.tau_dwell, total_time=0.4,
                     n_molecules=900, seed=5)
        f_bound.append(r.bound_fraction)
        D_eff.append(r.effective_diffusion())
    f_bound, D_eff = np.array(f_bound), np.array(D_eff)

    fb_line = np.linspace(0, 1, 100)
    theory = (1 - fb_line) * base.D_free + fb_line * base.D_bound

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(f_bound, D_eff, "o", ms=7, color=BLUE, label="MC (vary on-rate)")
    ax.plot(fb_line, theory, color=RED, lw=2,
            label="$(1-f_b)D_{free}+f_b D_{bound}$")
    ax.axhline(D_EFF_MEASURED, color=GREEN, ls="--",
               label=f"measured $D_{{eff}}$ = {D_EFF_MEASURED} um$^2$/s")
    # bound fraction that reproduces the measured value
    fb_match = (base.D_free - D_EFF_MEASURED) / (base.D_free - base.D_bound)
    ax.axvline(fb_match, color=GREEN, ls=":", alpha=0.7)
    ax.annotate(f"$f_b$ = {fb_match:.2f}", (fb_match, D_EFF_MEASURED),
                textcoords="offset points", xytext=(6, 6), color=GREEN)
    ax.set_xlabel("bound fraction $f_b$")
    ax.set_ylabel("effective diffusion (um$^2$/s)")
    ax.set_title("One model spans single-molecule and FDAP regimes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig6_Deff_vs_bound_fraction.png"))
    plt.close(fig)

    summary["bound_fraction_matching_measured_Deff"] = float(fb_match)
    np.savetxt(os.path.join(DATADIR, "Deff_vs_bound_fraction.csv"),
               np.column_stack([p_binds, f_bound, D_eff]),
               header="p_bind,bound_fraction,D_eff_um2_s",
               delimiter=",", comments="")


# ---------------------------------------------------------------------------
def main():
    print("Running main kiss-and-hop simulation ...")
    base = Params(tau_dwell=TAU_DWELL_PAPER, mt_spacing=0.30,
                  D_free=20.0, D_bound=0.10, process_width=1.5,
                  n_molecules=2500, total_time=1.0, record_stride=25, seed=0)
    res = simulate(base)

    summary = {
        "paper": "Janning et al. 2014, Mol Biol Cell 25:3541 "
                 "(doi:10.1091/mbc.e14-06-1099)",
        "inputs": {
            "tau_dwell_ms": base.tau_dwell * 1e3,
            "D_free_um2_s": base.D_free,
            "D_bound_um2_s": base.D_bound,
            "mt_spacing_um": base.mt_spacing,
            "capture_radius_um": base.capture_radius,
            "process_width_um": base.process_width,
            "n_molecules": base.n_molecules,
            "total_time_s": base.total_time,
            "dt_s": base.dt,
        },
    }

    fig_residence(res, summary)
    fig_bound_fraction(res, summary)
    fig_fdap(res, summary)
    fig_msd(res, summary)
    print("Running MT-spacing parameter sweep ...")
    fig_sweep(base, summary)
    print("Running on-rate sweep (effective diffusion regime) ...")
    fig_effective_diffusion_regime(base, summary)

    with open(os.path.join(DATADIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(f"  residence time (MC)      : {summary['residence_mean_ms']:.1f} ms "
          f"(paper ~ {TAU_DWELL_PAPER*1e3:.0f} ms)")
    print(f"  residence time (observed): {summary['residence_observed_ms']:.1f} ms")
    print(f"  bound fraction           : {summary['bound_fraction']:.3f}")
    print(f"  mean free (assoc.) time  : {summary['mean_free_time_ms']:.2f} ms")
    print(f"  D_eff (MSD)              : {summary['D_eff_from_msd']:.2f} um^2/s")
    print(f"  D_eff (FDAP fit)         : {summary['D_eff_from_fdap']:.2f} um^2/s")
    print(f"\nFigures -> {FIGDIR}\nData    -> {DATADIR}")


if __name__ == "__main__":
    main()
