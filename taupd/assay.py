"""
Assay simulator -- synthetic FDAP data with realistic measurement noise.

Used for (a) the recovery / identifiability test of the inverse fit and (b)
assay power analysis: given the expected noise and acquisition, what shift in
bound fraction (i.e. what compound effect size) is detectable?
"""

from __future__ import annotations
import numpy as np

from .model import fdap_effective, Deff_from_bound_fraction


def simulate_fdap(f_bound, D_free, sigma, t=None, D_bound=0.0,
                  noise=0.02, tau_bleach=None, n_cells=1, seed=0):
    """Generate a synthetic normalised FDAP transient for a given bound fraction.

    Parameters
    ----------
    f_bound    : true bound fraction to simulate.
    D_free     : free tau diffusion (um^2/s); sigma: activation half-width (um).
    t          : time points (s); default 0..2 s, 60 points.
    noise      : SD of additive Gaussian measurement noise on F.
    tau_bleach : optional photobleaching lifetime (s); multiplies F by exp(-t/tb).
    n_cells    : average this many independent noisy curves (HCS well replicate).

    Returns (t, F).
    """
    rng = np.random.default_rng(seed)
    if t is None:
        t = np.linspace(0.0, 2.0, 60)
    t = np.asarray(t, float)
    D_eff = Deff_from_bound_fraction(f_bound, D_free, D_bound)
    F0 = fdap_effective(t, D_eff, sigma)
    if tau_bleach is not None:
        F0 = F0 * np.exp(-t / tau_bleach)
    F = F0 + rng.normal(0.0, noise, size=(n_cells, t.size))
    return t, F.mean(axis=0)
