"""
taupd -- kinetic pharmacodynamics for tau-microtubule targeting compounds.

A proof-of-concept toolkit that turns live-cell FDAP/FRAP measurements of the
tau-microtubule interaction into mechanistic, drug-relevant parameters:

    bound fraction  ->  free tau pool (aggregation substrate)  ->  dose-response

It is built on the kiss-and-hop / reaction-diffusion framework of
Janning et al. 2014 (Mol Biol Cell) and Igaev et al. 2014 (Biophys J).

Scope (honest): this is a PRECLINICAL / RESEARCH tool for mechanism-of-action
and pharmacodynamic readouts. The proximal kinetic layer (bound fraction, free
pool) is well founded; the distal link free-pool -> aggregation -> efficacy is a
hypothesis the tool helps *test*, not a validated predictor. Not a medical
device.
"""

from .model import fdap_effective, fdap_two_component, bound_fraction_from_Deff, \
    free_tau_pool, Deff_from_bound_fraction
from .inverse import fit_fdap, FitResult
from .assay import simulate_fdap
from .pd import hill_dose_response, DoseResponse, classify_moa

__all__ = [
    "fdap_effective", "fdap_two_component", "bound_fraction_from_Deff",
    "Deff_from_bound_fraction", "free_tau_pool",
    "fit_fdap", "FitResult", "simulate_fdap",
    "hill_dose_response", "DoseResponse", "classify_moa",
]

__version__ = "0.0.1"
