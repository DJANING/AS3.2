"""
Kiss-and-hop Monte Carlo simulation of tau-microtubule interaction.

Reconstruction of the model from:

    Janning D, Igaev M, Suendermann F, Bruehmann J, Beutel O, Heinisch JJ,
    Bakota L, Piehler J, Junge W, Brandt R (2014)
    "Single-molecule tracking of tau reveals fast kiss-and-hop interaction
    with microtubules in living neurons."
    Mol Biol Cell 25(22):3541-3551. doi:10.1091/mbc.e14-06-1099

and the companion reaction-diffusion / FDAP analysis:

    Igaev M, Janning D, Suendermann F, Niewidok B, Brandt R, Junge W (2014)
    "A Refined Reaction-Diffusion Model of Tau-Microtubule Dynamics and Its
    Application in FDAP Analysis." Biophys J 107(11):2567-2578.

----------------------------------------------------------------------------
Model
----------------------------------------------------------------------------
A single tau molecule moves in the cross-sectional / longitudinal plane of a
neuritic process that is densely filled with parallel microtubules (MTs).

Coordinate convention (2D):
    x : along the process axis  (parallel to the MTs)   -> long, "open"
    y : transverse to the axis  (perpendicular to MTs)  -> bounded by width

Microtubules are modelled as infinite lines parallel to the x-axis, located at
discrete transverse positions  y = k * mt_spacing.  A tau molecule that comes
within `capture_radius` of an MT binds to it ("kiss"). While bound it stays at
that transverse position and may slide/piggyback along x with a small diffusion
constant D_bound. It detaches stochastically with a single rate

    k_off = 1 / tau_dwell           (tau_dwell ~ 40 ms in the paper)

so the residence time on one MT is exponentially distributed with mean
`tau_dwell` -- the central single-molecule observation of the paper. After
detaching it diffuses freely (D_free) until it reaches the next MT ("hop").

Because rebinding to a neighbouring MT is fast (association time ~ ms), the
molecule appears MT-associated for long stretches even though each individual
contact lasts only ~40 ms. This is the resolution of the apparent discrepancy
between fast single-molecule dwell times (~40 ms) and slow ensemble
FRAP/FDAP off-rates (~seconds) that the paper reports.

The macroscopic effective longitudinal diffusion constant follows from the
two-state average:

    D_eff = f_free * D_free + f_bound * D_bound
    f_bound = tau_dwell / (tau_dwell + tau_assoc)

where tau_assoc is the mean free (hopping) time between two contacts.

All lengths are in micrometres (um), all times in seconds (s); diffusion
constants therefore in um^2/s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Params:
    # --- kinetics ------------------------------------------------------------
    # Defaults are the values used in Janning et al. (2014).
    tau_dwell: float = 0.040      # s   mean residence time on a single MT (~40 ms)
    D_free: float = 14.4          # um^2/s  diffusion of free PAGFP-htau
                                  #      (Janning 2014, Materials and Methods)
    D_bound: float = 0.0          # um^2/s  bound tau is ~immobile in vivo
                                  #      (Igaev 2014: bound diffusion negligible)
    p_bind: float = 1.0           # binding probability per step inside capture zone
                                  # (the on-rate knob; 1.0 = diffusion-limited).
                                  # Lowering it lowers k*on/koff and the bound
                                  # fraction (Fig. 3B varies k*on/koff).

    # --- geometry ------------------------------------------------------------
    mt_spacing: float = 0.070     # um   average MT centre-to-centre spacing in
                                  #      PC12 neurites (Janning 2014; ~70 nm)
    process_width: float = 1.0    # um   transverse extent of the process
    capture_radius: float = 0.0125  # um  MT radius (~12.5 nm); tau binds on contact

    # --- imaging (for SSD / dwell reconstruction) ----------------------------
    frame_interval: float = 0.005   # s   camera frame time (~5 ms)
    loc_precision: float = 0.020    # um  single-molecule localisation precision (~20 nm)

    # --- integration ---------------------------------------------------------
    dt: float = 1.0e-5            # s    time step (must be << tau_dwell and
                                  #      << gap-crossing time; paper used 1 us)
    n_molecules: int = 2000
    total_time: float = 0.5       # s    simulated trajectory length
    record_stride: int = 25       # store the trajectory every Nth step
                                  # (event times for residence/free are always
                                  #  recorded at full dt resolution)
    seed: int = 0

    # derived (filled in __post_init__)
    n_steps: int = field(init=False)
    mt_positions: np.ndarray = field(init=False)

    def __post_init__(self):
        self.n_steps = int(round(self.total_time / self.dt))
        # MT transverse positions across the process (reflecting boundaries)
        n_mt = int(np.floor(self.process_width / self.mt_spacing)) + 1
        self.mt_positions = np.arange(n_mt) * self.mt_spacing

    @property
    def k_off(self) -> float:
        return 1.0 / self.tau_dwell


@dataclass
class Result:
    """Container for simulation output."""
    params: Params
    t: np.ndarray                 # (n_rec,) time axis of recorded frames
    x: np.ndarray                 # (n_molecules, n_rec) longitudinal position
    y: np.ndarray                 # (n_molecules, n_rec) transverse position
    bound: np.ndarray             # (n_molecules, n_rec) bool, bound state
    dt_record: float              # time between recorded frames (= dt * stride)
    residence_times: np.ndarray   # 1D, durations of completed bound intervals (s)
    free_times: np.ndarray        # 1D, durations of completed free intervals (s)

    # ---- derived observables -------------------------------------------------
    @property
    def bound_fraction(self) -> float:
        """Time- and ensemble-averaged fraction of molecules bound to an MT."""
        # discard the first 10% as burn-in towards the stationary state
        burn = self.bound.shape[1] // 10
        return float(self.bound[:, burn:].mean())

    @property
    def kon_koff(self) -> float:
        """Pseudo-equilibrium constant k*on/koff = f_bound / (1 - f_bound).

        Janning et al. (2014) report that the simulated SSD matches the data at
        k*on/koff ~ 10^2, implying ~99% of tau bound.
        """
        fb = self.bound_fraction
        return fb / max(1e-9, 1.0 - fb)

    def step_size_distribution(self, rng=None, hops_only=False):
        """Single-molecule step-size distribution (SSD), as in Janning Fig. 3.

        Pseudotrajectories are sampled at the camera frame interval with
        Gaussian localisation noise; the distribution of first-step
        displacements |r(t+dt) - r(t)| reveals discrete peaks at multiples of
        the MT spacing -- the signature of hopping between microtubules.
        Returns the 1-D array of step sizes in micrometres.

        If `hops_only` is True, only steps in which the molecule changed the
        microtubule it is nearest to are kept (i.e. genuine inter-MT jumps),
        which isolates the hop peaks from the localisation-noise peak.
        """
        p = self.params
        rng = rng or np.random.default_rng(7)
        # frames spaced by the camera interval out of the recorded trajectory
        fstride = max(1, int(round(p.frame_interval / self.dt_record)))
        xf = self.x[:, ::fstride]
        yf = self.y[:, ::fstride]
        # add localisation noise to each (x, y) localisation
        xn = xf + rng.normal(0.0, p.loc_precision, size=xf.shape)
        yn = yf + rng.normal(0.0, p.loc_precision, size=yf.shape)
        # first-step displacement of every (shortened) pseudotrajectory
        dx = xn[:, 1:] - xn[:, :-1]
        dy = yn[:, 1:] - yn[:, :-1]
        step = np.sqrt(dx * dx + dy * dy)
        if hops_only:
            mt = np.round(yf / p.mt_spacing)            # noise-free MT index
            moved = mt[:, 1:] != mt[:, :-1]
            return step[moved].ravel()
        return step.ravel()

    def dwell_time_colocalization(self, radius=0.050):
        """Dwell time measured the way the paper does it (Fig. 4 / Methods):

        count how many consecutive frames a localised signal stays within
        `radius` (default 50 nm) of where it was, and fit the resulting
        frequency histogram to a single exponential. Returns (durations_s,).
        """
        p = self.params
        fstride = max(1, int(round(p.frame_interval / self.dt_record)))
        xf = self.x[:, ::fstride]
        yf = self.y[:, ::fstride]
        durations = []
        for i in range(xf.shape[0]):
            x0, y0 = xf[i, 0], yf[i, 0]
            run = 1
            for j in range(1, xf.shape[1]):
                if (xf[i, j] - x0) ** 2 + (yf[i, j] - y0) ** 2 <= radius ** 2:
                    run += 1
                else:
                    durations.append(run * p.frame_interval)
                    x0, y0, run = xf[i, j], yf[i, j], 1
            durations.append(run * p.frame_interval)
        return np.asarray(durations)

    def msd(self, max_lag_frac: float = 0.25):
        """Ensemble mean-squared displacement of the longitudinal coordinate.

        Returns (lag_times, msd). Uses lags up to `max_lag_frac` of the trace.
        """
        n_frames = self.x.shape[1]
        max_lag = max(1, int(n_frames * max_lag_frac))
        lags = np.unique(np.linspace(1, max_lag, 60).astype(int))
        msd = np.empty(lags.size)
        for i, lag in enumerate(lags):
            d = self.x[:, lag:] - self.x[:, :-lag]
            msd[i] = np.mean(d * d)
        return lags * self.dt_record, msd

    def effective_diffusion(self) -> float:
        """Effective 1-D diffusion constant from a linear MSD fit (MSD = 2 D t)."""
        lag_t, msd = self.msd()
        # least-squares slope through the origin
        slope = np.dot(lag_t, msd) / np.dot(lag_t, lag_t)
        return slope / 2.0


def simulate(params: Params | None = None, **overrides) -> Result:
    """Run the vectorised kiss-and-hop Monte Carlo simulation.

    The simulation is vectorised over molecules and steps through time. State
    transitions (bind / release) are detected each step to accumulate the
    distribution of residence and free times.
    """
    p = params or Params(**overrides)
    if overrides and params is not None:
        raise ValueError("pass either a Params object or overrides, not both")

    rng = np.random.default_rng(p.seed)
    n, steps, dt = p.n_molecules, p.n_steps, p.dt

    sigma_free = np.sqrt(2.0 * p.D_free * dt)
    sigma_bound = np.sqrt(2.0 * p.D_bound * dt)
    p_release = 1.0 - np.exp(-dt / p.tau_dwell)   # exact per-step release prob
    half_gap = p.mt_spacing                        # for nearest-MT lookup

    # state arrays
    x = np.zeros(n)
    # start uniformly across the transverse extent
    y = rng.uniform(0.0, p.process_width, size=n)
    bound = np.zeros(n, dtype=bool)
    bound_y = np.zeros(n)                           # MT position a molecule is on
    state_start = np.zeros(n)                       # time current interval started
    forbidden_k = np.full(n, -1, dtype=int)         # MT index temporarily not
                                                    # re-bindable after detaching,
                                                    # until the molecule leaves its
                                                    # capture zone (prevents an
                                                    # unphysical instant rebind to
                                                    # the MT just left)

    # recorded (downsampled) trajectory for MSD / FDAP
    stride = max(1, p.record_stride)
    rec_idx = np.arange(0, steps, stride)
    n_rec = rec_idx.size
    x_traj = np.empty((n, n_rec), dtype=np.float32)
    y_traj = np.empty((n, n_rec), dtype=np.float32)
    bound_traj = np.empty((n, n_rec), dtype=bool)
    rec_ptr = 0

    residence_times: list[np.ndarray] = []
    free_times: list[np.ndarray] = []

    def nearest_mt(yy):
        """Index, position and distance of the nearest MT for each molecule."""
        k = np.round(yy / p.mt_spacing)
        k = np.clip(k, 0, p.mt_positions.size - 1).astype(int)
        ymt = k * p.mt_spacing
        return k, ymt, np.abs(yy - ymt)

    t = 0.0
    for s in range(steps):
        free = ~bound

        # --- free molecules: 2D diffusion --------------------------------------
        if free.any():
            x[free] += rng.normal(0.0, sigma_free, size=free.sum())
            y[free] += rng.normal(0.0, sigma_free, size=free.sum())
            # reflecting transverse boundaries [0, width]
            np.clip(y, 0.0, p.process_width, out=y)
            kmt, ymt, dist = nearest_mt(y)
            # a molecule that left an MT regains the right to re-bind it only
            # once it has diffused out of that MT's capture zone
            cleared = free & (forbidden_k >= 0) & (dist >= p.capture_radius)
            forbidden_k[cleared] = -1
            can_bind = free & (dist < p.capture_radius) & (kmt != forbidden_k)
            if p.p_bind < 1.0:
                can_bind &= rng.random(n) < p.p_bind
            if can_bind.any():
                # record the free interval that just ended
                free_times.append(t - state_start[can_bind])
                bound[can_bind] = True
                bound_y[can_bind] = ymt[can_bind]
                y[can_bind] = ymt[can_bind]          # snap to the MT
                forbidden_k[can_bind] = -1
                state_start[can_bind] = t

        # --- bound molecules: slide along x, may release -----------------------
        b = bound
        if b.any():
            x[b] += rng.normal(0.0, sigma_bound, size=b.sum())
            release = b & (rng.random(n) < p_release)
            if release.any():
                # record the residence interval that just ended
                residence_times.append(t - state_start[release])
                bound[release] = False
                state_start[release] = t
                # tau detaches into the cytoplasm at the MT position; it cannot
                # immediately rebind this MT -- it must first diffuse out of the
                # capture zone (a genuine hop). forbidden_k enforces this.
                forbidden_k[release] = np.round(
                    bound_y[release] / p.mt_spacing).astype(int)

        if s % stride == 0:
            x_traj[:, rec_ptr] = x
            y_traj[:, rec_ptr] = y
            bound_traj[:, rec_ptr] = bound
            rec_ptr += 1
        t += dt

    res = (np.concatenate(residence_times) if residence_times
           else np.array([]))
    fre = (np.concatenate(free_times) if free_times else np.array([]))

    return Result(
        params=p,
        t=rec_idx[:rec_ptr] * dt,
        x=x_traj[:, :rec_ptr],
        y=y_traj[:, :rec_ptr],
        bound=bound_traj[:, :rec_ptr],
        dt_record=dt * stride,
        residence_times=res,
        free_times=fre,
    )


if __name__ == "__main__":
    r = simulate()
    print(f"residence events : {r.residence_times.size}")
    print(f"mean residence   : {1e3*r.residence_times.mean():.1f} ms "
          f"(input tau_dwell = {1e3*r.params.tau_dwell:.0f} ms)")
    print(f"mean free time   : {1e3*r.free_times.mean():.2f} ms")
    print(f"bound fraction   : {r.bound_fraction:.3f}")
    print(f"k*on/koff        : {r.kon_koff:.0f}  (paper: ~100 -> 99% bound)")
    print(f"effective D      : {r.effective_diffusion():.2f} um^2/s "
          f"(D_free = {r.params.D_free} um^2/s)")
    ss = r.step_size_distribution() * 1e3
    print(f"SSD steps        : {ss.size}, median {np.median(ss):.0f} nm "
          f"(MT spacing = {r.params.mt_spacing*1e3:.0f} nm)")
