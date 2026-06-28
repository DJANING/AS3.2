# taupd — kinetic pharmacodynamics for tau-targeting compounds

A proof-of-concept toolkit that turns live-cell **FDAP/FRAP** measurements of the
tau–microtubule interaction into mechanistic, drug-relevant parameters:

```
FDAP curve (± compound)  →  bound fraction  →  free tau pool  →  dose-response / EC50
                                              (aggregation substrate)
```

Built on the kiss-and-hop / reaction-diffusion framework of Janning et al. 2014
(*Mol Biol Cell*) and Igaev et al. 2014 (*Biophys J*); see the repository root
for the underlying simulation and its validation.

## Why this could help drug testing

Several tau-targeting strategies act — directly or indirectly — on tau's
residence on microtubules, and therefore shift the **bound fraction** and the
**free tau pool** (the species available to aggregate):

| compound class | kinetic signature |
|---|---|
| MT-side stabiliser (epothilone, …) | ↑ bound fraction (more sites) |
| kinase inhibitor (GSK3β, MARK, …) | ↑ affinity (dephospho) → ↑ bound fraction |
| tau-lowering (ASO) | ↓ total tau → ↓ free pool |
| detachment promoter | ↓ bound fraction → ↑ free pool |

`taupd` reads these out in living cells, enabling **mechanism-of-action
fingerprinting**, **target engagement**, **dose-response on a proximal PD
marker**, and **assay power analysis** before an expensive screen.

## What's in P0

| module | does |
|---|---|
| `model` | forward FDAP (effective-diffusion + reaction-dominant) and kinetic↔PD conversions |
| `inverse` | fit an FDAP curve → bound fraction (+ koff when resolvable) with 68 % CIs |
| `assay` | synthetic FDAP with noise/bleaching — recovery tests and assay power |
| `pd` | Hill dose-response / EC50 and a transparent mechanism-of-action call |
| `demo` | the proof-of-concept figure (`figures/taupd_demo.png`) |

```bash
python3 -m taupd.demo            # recovery + dose-response demo
python3 -m pytest tests/ -q      # recovery / identifiability tests
```

Representative demo output: clean bound-fraction recovery (mean error 0.002),
free-pool dose-response **EC50 = 98 nM** (true 100 nM), and a correct
mechanism-of-action call — plus an explicit demonstration that **unmodelled
photobleaching biases the readout** (so bleaching QC is mandatory).

## Honest scope and limitations

- **Preclinical / research tool, not a medical device.** No clinical or
  diagnostic claim.
- The **proximal** layer (bound fraction, free pool from FDAP) is well founded
  and identifiable. The **distal** link *free pool → aggregation → efficacy* is a
  hypothesis the tool helps test, **not** a validated predictor — never sell it
  as an efficacy readout.
- Requires a **fluorescently tagged tau reporter** (PAGFP/Halo); translatability
  to endogenous tau must be argued separately.
- FDAP can trade off `k*on` and `koff`; the effective-diffusion fit deliberately
  reports the well-identified **bound fraction** and only reports `koff` when the
  small-spot / reaction-dominant regime applies.
- **Validation is the make-or-break next step**: reference compounds with known
  mechanism (e.g. epothilone D, a GSK3β inhibitor, tau knockdown) must produce
  the predicted kinetic signatures. P0 demonstrates identifiability on synthetic
  data only.
