"""
Forward FDAP models and the kinetic <-> pharmacodynamic conversions.

FDAP (fluorescence decay after photoactivation): a region of the cell process is
photoactivated at t=0 and the normalised fluorescence in that region decays as
labelled tau leaves it. The shape of the decay reports how mobile tau is, which
depends on how much of it is bound to (immobile) microtubules.

Two regimes (see Igaev et al. 2014):

  * Effective-diffusion regime (large activation spot / fast binding-unbinding):
    bound and free tau are locally equilibrated, so the process is pure diffusion
    with an effective constant
        D_eff = (1 - f_bound) * D_free + f_bound * D_bound
    For a 1-D Gaussian-activated profile of width sigma the normalised FDAP is
        F(t) = 1 / sqrt(1 + t / tau_D),    tau_D = sigma^2 / (2 D_eff).
    This is the standard, robustly identifiable case and yields the BOUND
    FRACTION (hence the free-tau pool). It does NOT resolve koff.

  * Reaction-dominant regime (small spot / slow koff): a fast diffusive component
    (free tau leaving) plus a slow exponential (bound tau, decaying at koff):
        F(t) = (1 - f_bound)/sqrt(1 + t/tau_D) + f_bound * exp(-koff t).
    Only valid when tau_D << 1/koff; then it additionally yields koff (dwell).

All lengths in micrometres (um), times in seconds (s), D in um^2/s,
concentrations in whatever unit the user supplies for total tau (the free pool
is returned in the same unit).
"""

from __future__ import annotations
import numpy as np


def fdap_effective(t, D_eff, sigma):
    """Normalised FDAP for the effective-diffusion regime (Gaussian profile)."""
    t = np.asarray(t, dtype=float)
    tau_D = sigma * sigma / (2.0 * D_eff)
    return 1.0 / np.sqrt(1.0 + t / tau_D)


def fdap_two_component(t, f_bound, koff, D_free, sigma):
    """Normalised FDAP for the reaction-dominant regime.

    Free tau leaves diffusively; bound tau is released at rate koff and then
    leaves quickly. Valid only when sigma^2/(2 D_free) << 1/koff.
    """
    t = np.asarray(t, dtype=float)
    tau_D = sigma * sigma / (2.0 * D_free)
    free = (1.0 - f_bound) / np.sqrt(1.0 + t / tau_D)
    bound = f_bound * np.exp(-koff * t)
    return free + bound


# ---- kinetic <-> pharmacodynamic conversions ------------------------------- #
def bound_fraction_from_Deff(D_eff, D_free, D_bound=0.0):
    """Invert D_eff = (1-f_b) D_free + f_b D_bound for the bound fraction f_b."""
    f_b = (D_free - D_eff) / (D_free - D_bound)
    return float(np.clip(f_b, 0.0, 1.0))


def Deff_from_bound_fraction(f_bound, D_free, D_bound=0.0):
    """Forward map: bound fraction -> effective diffusion constant."""
    return (1.0 - f_bound) * D_free + f_bound * D_bound


def free_tau_pool(f_bound, tau_total):
    """The free (unbound) tau concentration -- the aggregation substrate.

    free = (1 - f_bound) * tau_total
    Returned in the same unit as `tau_total`.
    """
    return (1.0 - f_bound) * tau_total


def regime_is_effective_diffusion(koff, D_free, sigma, ratio=5.0):
    """True if the effective-diffusion regime applies: diffusion out of the
    activation region is slow compared with binding turnover
    (tau_D * koff >> 1)."""
    tau_D = sigma * sigma / (2.0 * D_free)
    return tau_D * koff >= ratio
