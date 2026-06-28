"""
Pharmacodynamics layer: dose-response (EC50) and mechanism-of-action calls.

A compound is characterised by how it shifts the fitted parameters relative to
vehicle. The drug-relevant response is usually the BOUND FRACTION or, better,
the FREE TAU POOL (the aggregation substrate).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit


def _hill(dose, bottom, top, ec50, n):
    return bottom + (top - bottom) / (1.0 + (ec50 / np.maximum(dose, 1e-12)) ** n)


@dataclass
class DoseResponse:
    ec50: float
    hill_n: float
    bottom: float
    top: float
    ec50_ci68: float
    response_name: str

    def summary(self) -> str:
        return (f"{self.response_name}: EC50 = {self.ec50:.3g} "
                f"(+/- {self.ec50_ci68:.2g}), Hill n = {self.hill_n:.2f}, "
                f"range [{self.bottom:.3g}, {self.top:.3g}]")


def hill_dose_response(doses, response, response_name="response"):
    """Fit a 4-parameter Hill curve response(dose) and return EC50 etc.

    `response` may be bound fraction, free-tau pool, or any monotonic readout.
    Works for both increasing and decreasing responses.
    """
    doses = np.asarray(doses, float)
    response = np.asarray(response, float)
    lo, hi = float(response.min()), float(response.max())
    span = (hi - lo) or 1.0
    pos = doses[doses > 0]
    ec50_0 = float(np.median(pos)) if pos.size else 1.0
    p0 = [response[np.argmin(doses)], response[np.argmax(doses)], ec50_0, 1.0]
    bounds = ([lo - span, lo - span, 1e-9, 0.1],
              [hi + span, hi + span, 1e9, 8.0])
    popt, pcov = curve_fit(_hill, doses, response, p0=p0, bounds=bounds,
                           maxfev=20000)
    sd = np.sqrt(np.diag(pcov))
    return DoseResponse(ec50=float(popt[2]), hill_n=float(popt[3]),
                        bottom=float(popt[0]), top=float(popt[1]),
                        ec50_ci68=float(sd[2]), response_name=response_name)


def classify_moa(baseline, treated, d_free_changed=False, tau_total_changed=False):
    """Rule-based mechanism-of-action call from baseline vs treated fits.

    baseline, treated : FitResult objects (effective-diffusion model).
    Returns a short label + the signed shifts. This is a transparent heuristic,
    not a trained classifier -- it states which knob moved.
    """
    fb0 = baseline.derived.get("bound_fraction")
    fb1 = treated.derived.get("bound_fraction")
    dfb = fb1 - fb0
    if tau_total_changed:
        call = "tau-lowering (total tau reduced; free pool falls without f_b change)"
    elif abs(dfb) < (baseline.derived.get("bound_fraction_ci68", 0.0)
                     + treated.derived.get("bound_fraction_ci68", 0.0)):
        call = "no measurable target engagement (bound fraction unchanged)"
    elif dfb > 0 and d_free_changed:
        call = "MT-side stabiliser (more binding sites -> higher bound fraction)"
    elif dfb > 0:
        call = "affinity increase (e.g. dephosphorylation / kinase inhibition)"
    else:
        call = "detachment promoter (bound fraction reduced -> free pool rises)"
    return {"call": call, "delta_bound_fraction": dfb,
            "bound_fraction_baseline": fb0, "bound_fraction_treated": fb1}
