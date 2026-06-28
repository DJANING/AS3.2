"""Recovery / sanity tests for taupd. Run: python3 -m pytest tests/ -q"""
import numpy as np

from taupd import (simulate_fdap, fit_fdap, free_tau_pool,
                   bound_fraction_from_Deff, Deff_from_bound_fraction,
                   hill_dose_response)

D_FREE, SIGMA = 14.4, 2.5


def test_roundtrip_conversion():
    for fb in (0.1, 0.5, 0.9, 0.99):
        D = Deff_from_bound_fraction(fb, D_FREE)
        assert abs(bound_fraction_from_Deff(D, D_FREE) - fb) < 1e-9


def test_recovery_clean():
    # recovered bound fraction should track the truth within ~0.05 (no artefacts)
    errs = []
    for i, fb in enumerate(np.linspace(0.3, 0.95, 8)):
        t, F = simulate_fdap(fb, D_FREE, SIGMA, noise=0.02, n_cells=8, seed=i)
        r = fit_fdap(t, F, SIGMA, D_FREE, model="effective")
        errs.append(abs(r.derived["bound_fraction"] - fb))
    assert np.mean(errs) < 0.05


def test_free_pool_and_ci():
    t, F = simulate_fdap(0.8, D_FREE, SIGMA, noise=0.02, n_cells=8, seed=1)
    r = fit_fdap(t, F, SIGMA, D_FREE, tau_total=10.0, model="effective")
    assert 0.7 < r.derived["bound_fraction"] < 0.9
    assert abs(r.derived["free_tau_pool"] - free_tau_pool(
        r.derived["bound_fraction"], 10.0)) < 1e-6
    assert r.derived["bound_fraction_ci68"] > 0


def test_dose_response_ec50():
    doses = np.array([0, 1, 3, 10, 30, 100, 300, 1000, 3000.0])
    ec50 = 100.0
    fb = 0.5 + 0.45 / (1 + (ec50 / np.maximum(doses, 1e-9)) ** 1.3)
    free = []
    for i, b in enumerate(fb):
        t, F = simulate_fdap(b, D_FREE, SIGMA, noise=0.02, n_cells=8, seed=i)
        r = fit_fdap(t, F, SIGMA, D_FREE, tau_total=1.0, model="effective")
        free.append(r.derived["free_tau_pool"])
    dr = hill_dose_response(doses, np.array(free), "free tau pool")
    # recovered EC50 within a factor of 2 of the truth
    assert 0.5 * ec50 < dr.ec50 < 2.0 * ec50
