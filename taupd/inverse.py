"""
Inverse problem: fit an FDAP transient and report the drug-relevant parameters
with confidence intervals.

Primary output is the BOUND FRACTION (and hence the free-tau pool) from the
effective-diffusion fit, which is the robustly identifiable case. If a koff is
requested and the regime allows it, the two-component model is fitted instead.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import curve_fit

from .model import (fdap_effective, fdap_two_component, bound_fraction_from_Deff,
                    free_tau_pool, regime_is_effective_diffusion)


@dataclass
class FitResult:
    model: str                      # "effective" or "two_component"
    sigma: float
    D_free: float
    D_bound: float
    params: dict                    # point estimates
    ci68: dict                      # +/- 1 sigma (68%) half-widths
    rmse: float
    tau_total: float | None = None
    derived: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"FDAP fit ({self.model} model), RMSE = {self.rmse:.4f}"]
        for k, v in self.params.items():
            ci = self.ci68.get(k, float("nan"))
            lines.append(f"  {k:14s} = {v:.4g}  +/- {ci:.2g}")
        for k, v in self.derived.items():
            lines.append(f"  {k:14s} = {v:.4g}")
        return "\n".join(lines)


def fit_fdap(t, F, sigma, D_free, D_bound=0.0, tau_total=None,
             model="auto", koff_guess=25.0, p0_bound=0.8):
    """Fit an FDAP curve and return bound fraction, free pool (and koff if
    the reaction regime applies), with 68% confidence intervals.

    Parameters
    ----------
    t, F      : arrays, time (s) and normalised fluorescence (F(0)~1).
    sigma     : activation-region half-width (um).
    D_free    : free tau diffusion constant (um^2/s), measured independently.
    D_bound   : bound tau diffusion (um^2/s), ~0 in vivo.
    tau_total : total tau concentration; if given, the free pool is reported.
    model     : "effective", "two_component", or "auto".
    """
    t = np.asarray(t, float); F = np.asarray(F, float)

    if model == "auto":
        model = ("two_component"
                 if not regime_is_effective_diffusion(koff_guess, D_free, sigma)
                 else "effective")

    if model == "effective":
        # fit D_eff
        def f(tt, D_eff):
            return fdap_effective(tt, D_eff, sigma)
        D_eff0 = (1 - p0_bound) * D_free + p0_bound * D_bound
        D_eff0 = max(D_eff0, 1e-3)
        popt, pcov = curve_fit(f, t, F, p0=[D_eff0],
                               bounds=(1e-4, D_free * 1.5))
        D_eff = float(popt[0]); dD = float(np.sqrt(pcov[0, 0]))
        f_b = bound_fraction_from_Deff(D_eff, D_free, D_bound)
        # propagate the D_eff uncertainty to f_bound (linear map)
        df_b = dD / (D_free - D_bound)
        rmse = float(np.sqrt(np.mean((F - f(t, *popt)) ** 2)))
        params = {"D_eff": D_eff}
        ci68 = {"D_eff": dD}
        derived = {"bound_fraction": f_b, "bound_fraction_ci68": df_b}
        if tau_total is not None:
            derived["free_tau_pool"] = free_tau_pool(f_b, tau_total)
            derived["free_tau_pool_ci68"] = df_b * tau_total
        return FitResult("effective", sigma, D_free, D_bound, params, ci68,
                         rmse, tau_total, derived)

    # two-component reaction-dominant model -> f_bound and koff
    def g(tt, f_bound, koff):
        return fdap_two_component(tt, f_bound, koff, D_free, sigma)
    popt, pcov = curve_fit(g, t, F, p0=[p0_bound, koff_guess],
                           bounds=([0.0, 1e-3], [1.0, 1e4]))
    f_b, koff = float(popt[0]), float(popt[1])
    sd = np.sqrt(np.diag(pcov))
    rmse = float(np.sqrt(np.mean((F - g(t, *popt)) ** 2)))
    params = {"bound_fraction": f_b, "koff": koff}
    ci68 = {"bound_fraction": float(sd[0]), "koff": float(sd[1])}
    derived = {"dwell_ms": 1e3 / koff}
    if tau_total is not None:
        derived["free_tau_pool"] = free_tau_pool(f_b, tau_total)
        derived["free_tau_pool_ci68"] = float(sd[0]) * tau_total
    return FitResult("two_component", sigma, D_free, D_bound, params, ci68,
                     rmse, tau_total, derived)
