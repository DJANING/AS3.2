# Tau &ldquo;kiss-and-hop&rdquo; &mdash; data &amp; Monte&nbsp;Carlo reconstruction

A reconstruction and simulation of the model in:

> **Janning D, Igaev M, Sündermann F, Brühmann J, Beutel O, Heinisch JJ,
> Bakota L, Piehler J, Junge W, Brandt R (2014).**
> *Single-molecule tracking of tau reveals fast kiss-and-hop interaction with
> microtubules in living neurons.*
> **Molecular Biology of the Cell 25(22):3541&ndash;3551.**
> doi:[10.1091/mbc.e14-06-1099](https://doi.org/10.1091/mbc.e14-06-1099)

with the kinetic / reaction-diffusion link to the companion paper:

> Igaev M, Janning D, Sündermann F, Niewidok B, Brandt R, Junge W (2014).
> *A Refined Reaction-Diffusion Model of Tau-Microtubule Dynamics and Its
> Application in FDAP Analysis.* Biophys J 107(11):2567&ndash;2578.

The parameters below are taken from the papers' Materials and Methods.

| Part | What | Where |
|------|------|-------|
| Quantitative MC | Vectorised NumPy simulation + figures (dwell time, bound fraction, **step-size distribution**, MSD / effective diffusion) | [`simulation/`](simulation/), [`figures/`](figures/), [`data/`](data/) |
| Interactive web | Live in-browser Monte Carlo of tau hopping between microtubules, with sliders | [`index.html`](index.html) (GitHub Pages) |

---

## The biology &amp; the model

Ensemble FRAP/FDAP had suggested tau stays bound to a microtubule (MT) for
**seconds**. Single-molecule tracking in this paper showed instead that tau
dwells on one MT for only **~40&nbsp;ms** before detaching and rapidly rebinding a
neighbouring MT &mdash; the **kiss-and-hop** mechanism. Because each contact is
so short, tau can regulate MT dynamics without blocking axonal transport.

We model a single tau molecule moving in the plane of a neuritic process filled
with **parallel microtubules**:

```
 y (transverse, bounded by the process width)
 ^
 |======================================  MT  (y = k · 70 nm)
 |        o  <- free tau (diffuses, D_free = 14.4 µm²/s)
 |======================================  MT
 |   ●  <- bound tau (kiss): held on the MT, detaches with rate
 |         k_off = 1/τ_dwell  (τ_dwell ≈ 40 ms), then hops
 |======================================  MT
 +-------------------------------------->  x (along the axis, parallel to MTs)
```

### The actual Monte Carlo algorithm (Janning 2014, Materials and Methods)

> *"The time step was equal to Δt = 1&nbsp;μs. At each time step, each bound
> particle had a probability p<sub>off</sub> of unbinding &hellip; each free
> particle was first checked for the act of hitting a MT surface. When hit, the
> particle had a probability p<sub>on</sub> of being attached &hellip; a random
> displacement (Δx, Δy, Δz) &hellip; normally distributed &hellip; with SD equal
> to √(2DΔt) &hellip; Reflecting boundary conditions &hellip; iterated for
> 10⁶&ndash;10⁷ time steps."*

This repo implements exactly that loop (in 2D, the plane that the single-molecule
imaging projects onto; `dt = 10 µs` for speed). The on-/off-rates are tuned
through the single knob `p_bind`, summarised by the **pseudo-equilibrium
constant `k*on/koff = f_bound/(1−f_bound)`**. The paper finds that the simulated
step-size distribution matches the data at **`k*on/koff ≈ 100`, i.e. ~99 % of tau
bound** &mdash; reproduced here.

### Equations

* **Residence time** (single MT): `P(t) = (1/τ_d)·exp(−t/τ_d)`, `τ_d ≈ 40 ms`.
* **Bound fraction**: `f_bound = τ_dwell/(τ_dwell + τ_assoc)`;
  `k*on/koff = f_bound/(1−f_bound)`.
* **Two-state reaction-diffusion** (Igaev 2014):
  `∂[T_F]/∂t = D_F∇²[T_F] − k*on[T_F] + koff[T_B]`,
  `∂[T_B]/∂t = D_B∇²[T_B] + k*on[T_F] − koff[T_B]`.
* **Effective diffusion** (fast-exchange limit):
  `D_eff = (1−f_bound)·D_free + f_bound·D_bound`.

---

## What the simulation reproduces

`python3 simulation/run_experiments.py` produces:

| Figure | Paper | Result |
|--------|-------|--------|
| `fig1_dwell_time.png` | Fig. 4 | Dwell time measured the paper's way (consecutive frames within a 50 nm radius, 5 ms frames) is a **single exponential**, τ ≈ 40–47 ms; true per-MT residence ≈ 38 ms. |
| `fig2_bound_fraction.png` | &mdash; | Tau is **~99 % MT-bound** (`k*on/koff ≈ 107`, matching the paper's ~100). |
| `fig3_step_size_distribution.png` | **Fig. 3** | **The Monte Carlo figure.** The step-size distribution shows the localisation-noise peak plus an inter-MT **hop** population peaking near the MT spacing, with higher-order hops (1→2, 1→3, …) in the tail. Panel B: the hop tail grows as `k*on/koff` falls. |
| `fig4_msd.png` | &mdash; | Longitudinal `D_eff ≈ 0.13 µm²/s` &mdash; tiny, because tau is ~99 % bound: it does not run ahead of, or impede, axonal transport. |
| `fig5_Deff_vs_bound_fraction.png` | &mdash; | `D_eff = (1−f_b)D_free + f_b·D_bound` (Igaev 2014); at the physiological `f_b ≈ 0.99` the effective transport of tau is ≈ 0. |
| `fig6_exact_3d.png` | **Fig. 3** | **Exact 3D reconstruction** (see below): the 60-MT hex bundle and the SSD of the 2D-projected localisations. |

Representative output (paper parameters):

```
dwell (50 nm coloc.)   : 47 ms   (paper ~ 40 ms; PC12 39±4, axons 36±5)
true per-MT residence  : 38 ms
bound fraction         : 0.991   (k*on/koff = 107; paper ~100)
D_eff (MSD)            : 0.134 µm²/s
```

---

## Exact 3D reconstruction (`kiss_and_hop_3d.py`)

`simulation/kiss_and_hop_3d.py` implements the paper's Monte Carlo *verbatim*,
in its full 3D geometry rather than the 2D idealisation above:

* simulation space = a **tube of L = 100 µm, R = 500 nm**;
* filled with **60 parallel microtubules**, R<sub>MT</sub> = 12.5 nm, hexagonally
  packed at **70 nm** nearest-neighbour spacing (Jacobs &amp; Stevens 1986);
* **dt = 1 µs**; each step: bound particles unbind with `p_off = dt/τ_dwell` and
  detach **perpendicular to the MT surface**; free particles that hit an MT
  surface attach with `p_on`, else take a 3D Gaussian step √(2D·dt); **reflecting**
  tube wall;
* calibrated to **k\*on/koff ≈ 100** (here `p_on = 0.20` → f_bound ≈ 0.99), the
  value at which the paper's simulated SSD matches the data.

`python3 simulation/run_experiments_3d.py` produces `fig6_exact_3d.png`. The
single-molecule image is the **2D projection** of this 3D process (imaging plane
= longitudinal × one transverse axis; the depth axis is integrated out), exactly
as in the experiment.

**Honest result.** The projected SSD is **multi-modal**: a localisation-noise
peak, then an inter-MT **hop** population whose tail extends through multiples of
the spacing out to the reported 95 / 190 / 285 nm. With the full 3D depth it does
**not** show the cleanly *separated* peaks of the published Fig. 3A, because
projecting a 3D hex bundle onto 2D smears neighbouring-shell distances together
(and produces the extra short-distance peak the paper itself notes, from MTs that
neighbour in projection but are distant in depth).

**Why the experiment resolves the peaks better — TIRF.** The imaging was done by
**TIRF** microscopy, whose evanescent field only illuminates a thin slice
(~100–150 nm) just above the coverslip. The experiment therefore sees a
quasi-2D sheet of MTs, not the whole 3D bundle, which removes much of the
depth-projection smearing. `fig6` shows this directly: restricting the simulated
localisations to a 150 nm TIRF slice (right panel, orange) trims the long
cross-bundle tail relative to the full-depth SSD (blue). In the model the effect
is only *partial* (a 150 nm slice still spans ~3 hexagonal rows, and the residual
blur from the 20 nm localisation precision and the longitudinal motion during a
hop remains), so it sharpens rather than fully separates the peaks. The cleaner
published peaks reflect TIRF **plus** the specific, more ordered real bundle
(≈95 nm spacing in that cell) and the trajectory pre-processing.

The single-exponential ~40 ms dwell, the ~99 % bound fraction and
k\*on/koff ≈ 100 are reproduced exactly. The idealised 2D SSD (`fig3`) is kept as
the didactic version where the hop peaks sit cleanly at multiples of the spacing.

---

## Parameters (from the papers)

| Symbol | Name | Value | Source |
|--------|------|-------|--------|
| `τ_dwell` | dwell time on one MT | **40 ms** | Janning 2014, single molecule |
| `D_free` | free PAGFP-htau diffusion | **14.4 µm²/s** | Janning 2014, Stokes–Einstein (η=3.4 cP, T=310 K) |
| `D_bound` | diffusion of bound tau | **0** (negligible *in vivo*) | Igaev 2014 |
| `mt_spacing` | average MT spacing (PC12 neurites) | **70 nm** | Janning 2014 |
| `capture_radius` | MT radius / binding reach | 12.5 nm | MT geometry |
| `frame_interval` | camera frame time | 5 ms | Janning 2014 (440 frames = 2.2 s) |
| `loc_precision` | localisation precision | ~20 nm | Janning 2014 (Halo-tau 16.7 nm) |
| `k*on/koff` | pseudo-equilibrium constant | **~100** (⇒ ~99 % bound) | Janning 2014, SSD fit |
| `dt` | integration step | 1 µs (10 µs here) | Janning 2014 |

> **Note on the 13.9 µm²/s value.** `13.9 ± 1.0 µm²/s` in the paper is the
> diffusion constant of the free **3×PAGFP tracer** (a control), *not* an
> effective diffusion of tau. It is used to calibrate the viscosity and is
> recorded in `data/summary.json` for reference only. The diffusion constant of
> free tau used in the Monte Carlo is **14.4 µm²/s**.

The geometric/integration choices (2D projection instead of 3D, `dt = 10 µs`)
are simplifications for speed; everything is exposed as `Params` fields in
`simulation/kiss_and_hop.py`.

---

## Run it

```bash
pip install -r requirements.txt

python3 simulation/kiss_and_hop.py        # core engine smoke test
python3 simulation/run_experiments.py     # full reconstruction -> figures/ + data/
```

Interactive version: open `index.html` (or the GitHub Pages site). Drag the
sliders for dwell time, MT spacing, free diffusion and on-rate and watch the
bound fraction, residence histogram and effective diffusion respond live.

### Code layout

```
simulation/kiss_and_hop.py        2D MC engine (Params, simulate, Result)
                                  + SSD and 50 nm-colocalisation dwell analysis
simulation/kiss_and_hop_3d.py     exact 3D MC engine (paper geometry, dt = 1 us)
simulation/run_experiments.py     reproduces fig1-fig5 + writes data/summary.json
simulation/run_experiments_3d.py  reproduces the exact-3D figure (fig6)
figures/                          generated PNGs
data/                             generated CSV + summary.json
index.html                        interactive in-browser simulation (GitHub Pages)
```
