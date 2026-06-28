# Tau &ldquo;kiss-and-hop&rdquo; &mdash; data &amp; Monte&nbsp;Carlo reconstruction

A reconstruction and simulation of the model in:

> **Janning D, Igaev M, Sündermann F, Brühmann J, Beutel O, Heinisch JJ,
> Bakota L, Piehler J, Junge W, Brandt R (2014).**
> *Single-molecule tracking of tau reveals fast kiss-and-hop interaction with
> microtubules in living neurons.*
> **Molecular Biology of the Cell 25(22):3541&ndash;3551.**
> doi:[10.1091/mbc.e14-06-1099](https://doi.org/10.1091/mbc.e14-06-1099)

with the kinetic / effective-diffusion link to the companion paper:

> Igaev M, Janning D, Sündermann F, Niewidok B, Brandt R, Junge W (2014).
> *A Refined Reaction-Diffusion Model of Tau-Microtubule Dynamics and Its
> Application in FDAP Analysis.* Biophys J 107(11):2567&ndash;2578.

There are two parts:

| Part | What | Where |
|------|------|-------|
| Quantitative MC | Vectorised NumPy simulation + figures (residence distribution, bound fraction, FDAP, MSD / effective diffusion, parameter sweeps) | [`simulation/`](simulation/), [`figures/`](figures/), [`data/`](data/) |
| Interactive web | Live in-browser Monte Carlo of tau hopping between microtubules, with sliders | [`index.html`](index.html) (GitHub Pages) |

---

## The biology &amp; the model

Tau is a microtubule-associated protein (MAP) of the neuronal axon. Ensemble
FRAP/FDAP experiments had suggested that tau stays bound to a microtubule (MT)
for **seconds**. Single-molecule tracking in this paper showed instead that tau
dwells on one MT for only **~40&nbsp;ms** before detaching, and rapidly rebinds a
neighbouring MT &mdash; the **kiss-and-hop** mechanism. This resolves a paradox:
tau can regulate MT dynamics without blocking axonal transport, because every
individual contact is extremely short.

We model a single tau molecule moving in the plane of a neuritic process that
is densely filled with **parallel microtubules**:

```
 y (transverse, bounded by the process width)
 ^
 |======================================  MT  (y = k · spacing)
 |        o  <- free tau (diffuses, D_free)
 |======================================  MT
 |   ●  <- bound tau (kiss): stays on the MT, slides slowly (D_bound),
 |         detaches with rate k_off = 1/τ_dwell  (τ_dwell ≈ 40 ms)
 |======================================  MT
 +-------------------------------------->  x (along the axis, parallel to MTs)
```

* **Free state** &ndash; 2-D Brownian motion with diffusion constant `D_free`.
  On contact with an MT (within `capture_radius`) tau binds with probability
  `p_bind` per step (`p_bind = 1` &rarr; diffusion-limited capture).
* **Bound state** &ndash; tau is held at the MT&rsquo;s transverse position, may
  slide/piggyback along the axis with a small `D_bound`, and detaches
  stochastically so that the **residence time is exponential with mean
  `τ_dwell`**.
* On detaching, tau cannot instantly rebind the same MT &mdash; it must first
  diffuse out of that MT&rsquo;s capture zone (a genuine *hop*).

### Equations reproduced

* **Residence time** (single MT): `P(t) = (1/τ_d) · exp(−t/τ_d)`, `τ_d ≈ 40 ms`.
* **Bound fraction** (two-state stationary):
  `f_bound = τ_dwell / (τ_dwell + τ_assoc)`, with `τ_assoc` the mean free/hop
  time between two contacts.
* **Effective longitudinal diffusion** (fast-exchange / effective-medium limit):
  `D_eff = (1 − f_bound)·D_free + f_bound·D_bound`.
* **FDAP transient** (all molecules activated at `x = 0`, window `±w`):
  `F(t) = erf( w / sqrt(4·D_eff·t) )`.

---

## What the simulation shows

Running `simulation/run_experiments.py` produces these figures:

| Figure | Result |
|--------|--------|
| `fig1_residence_time.png` | Single-MT residence is a **single exponential**, mean ≈ 40 ms; the dwell time is still recovered after adding camera frame-time (5 ms) and photobleaching. |
| `fig2_bound_fraction.png` | Despite 40 ms contacts, tau is **mostly MT-associated** in a dense array; the MC time-average matches `τ_d/(τ_d+τ_assoc)` exactly. |
| `fig3_fdap.png` | FDAP decay fitted by the reaction-diffusion curve &rarr; effective diffusion constant. |
| `fig4_msd.png` | Longitudinal MSD &rarr; `D_eff` (consistent with the FDAP fit). |
| `fig5_sweep_spacing.png` | Sparser MTs &rarr; faster effective diffusion / lower bound fraction. |
| `fig6_Deff_vs_bound_fraction.png` | One model spans both regimes: the MC points fall on `D_eff = (1−f_b)D_free + f_b·D_bound`, crossing the **measured `D_eff ≈ 13.9 µm²/s`** at `f_b ≈ 0.31`. |

Representative output (default dense-axon parameters):

```
residence time (MC)      : 38.4 ms   (paper ~ 40 ms)
residence time (observed): 41.5 ms   (5 ms frames + bleaching)
bound fraction           : 0.993     (= analytic two-state value)
mean free (assoc.) time  : 0.28 ms
D_eff (MSD)              : 0.24 µm²/s = D_eff (FDAP fit) : 0.25 µm²/s
```

### Two regimes, one model

The **single-molecule regime** (dense mature axon) has `f_bound ≈ 0.99` and a
small longitudinal `D_eff` &mdash; tau is almost always within reach of an MT,
yet each contact lasts only ~40 ms. The **FDAP regime** that yields the measured
`D_eff ≈ 13.9 µm²/s` corresponds to a lower bound fraction (`f_b ≈ 0.3`), reached
in the model by lowering the on-rate (`p_bind`) or the MT density. `fig6` shows
both ends of the same `D_eff(f_bound)` line.

---

## Parameters

Defaults (`simulation/kiss_and_hop.py`, class `Params`). Units: µm, s, µm²/s.

| Symbol | Name | Default | Source |
|--------|------|---------|--------|
| `τ_dwell` | residence time on one MT | **0.040 s** | Janning 2014 (single molecule) |
| `D_free` | free cytoplasmic diffusion of tau | 20 µm²/s | order-of-magnitude for a ~50 kDa protein |
| `D_bound` | sliding/piggyback diffusion while bound | 0.10 µm²/s | assumed (1-D MT diffusion + slow transport) |
| `mt_spacing` | transverse MT centre-to-centre spacing | 0.30 µm | representative axon (≈11 MTs/µm²) |
| `capture_radius` | tau reach to bind an MT | 0.025 µm | ~MT radius + tau reach |
| `process_width` | transverse extent of the process | 1.5 µm | typical neurite |
| `p_bind` | capture probability per step (on-rate knob) | 1.0 | diffusion-limited |

> **Note on fidelity.** The kinetics (`τ_dwell ≈ 40 ms`, single-exponential
> residence) and the measured `D_eff ≈ 13.9 µm²/s` are taken from the papers.
> The geometric parameters (`mt_spacing`, `capture_radius`, `process_width`)
> and `D_free`/`D_bound` are physically reasonable values, not digitised from
> the paper&rsquo;s supplement &mdash; the publisher/PMC full text was not
> reachable from the build environment. They are exposed as `Params` fields so
> exact values can be dropped in. The structure of the model and the
> relationships it reproduces are faithful to the papers.

---

## Run it

```bash
pip install -r requirements.txt

# core engine smoke test
python3 simulation/kiss_and_hop.py

# full reconstruction -> figures/ and data/
python3 simulation/run_experiments.py
```

Interactive version: open `index.html` in a browser, or visit the GitHub Pages
site for this repository. Drag the sliders for dwell time, MT spacing, free
diffusion and on-rate and watch the bound fraction, residence histogram and
effective diffusion respond live.

### Code layout

```
simulation/kiss_and_hop.py    core vectorised MC engine (Params, simulate, Result)
simulation/run_experiments.py reproduces all figures + writes data/summary.json
figures/                      generated PNGs
data/                         generated CSV + summary.json
index.html                    interactive in-browser simulation (GitHub Pages)
```
