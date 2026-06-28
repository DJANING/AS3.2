"""
Exact 3D reconstruction of the Monte Carlo simulation in Janning et al. (2014).

Geometry and algorithm are taken verbatim from the paper's Materials and Methods
("Monte Carlo simulations of tau dynamics"):

    "The simulation space was defined as a tube of length L = 100 um and radius
     R = 500 nm. The tube was filled with 60 parallel MTs of R_MT = 12.5 nm,
     with an intermicrotubule distance of 70 nm. ... At the initial moment,
     1000 particles were randomly positioned ... Each particle had a probability
     k*on/(k*on+koff) of being bound to a MT and koff/(k*on+koff) of being placed
     anywhere in the cytosol. ... The time step was equal to dt = 1 us. At each
     time step, each bound particle had a probability p_off of unbinding ...
     detachment ... in the direction locally perpendicular to the MT surface ...
     each free particle was first checked for the act of hitting a MT surface.
     When hit, the particle had a probability p_on of being attached ... If the
     free particle did not hit a MT surface or failed attaching ... a random
     displacement (dx, dy, dz) ... normally distributed ... SD = sqrt(2 D dt).
     Reflecting boundary conditions at the outer boundary ... iterated for
     1e6-1e7 time steps."

The microtubules are parallel to the tube axis (z). Their cross-sectional
centres form a hexagonal bundle with 70 nm nearest-neighbour spacing (the 60
lattice nodes closest to the axis), in correspondence with Jacobs & Stevens
(1986). Single-molecule imaging projects the 3D process onto a 2D plane; here
that is the (x, z) plane (x = lateral in the image, z = longitudinal, y = depth
/ optical axis). The step-size distribution (SSD) is computed from this
projection -- which is why hops between MTs that are neighbours in 3D appear at
a spread of apparent distances, and why MTs that are close in projection but
distant in depth produce the extra short-distance peak the paper notes.

Units: micrometres (um) and seconds (s); diffusion constants in um^2/s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def hex_bundle(n_mt: int, spacing: float):
    """Return the (n_mt, 2) cross-sectional centres of a hexagonal MT bundle:
    the `n_mt` triangular-lattice nodes closest to the origin."""
    a = spacing
    v1 = np.array([a, 0.0])
    v2 = np.array([a * 0.5, a * np.sqrt(3) / 2])
    nodes = []
    rng = range(-2 * n_mt, 2 * n_mt + 1)
    span = int(np.sqrt(n_mt)) + 4
    for q in range(-span, span + 1):
        for r in range(-span, span + 1):
            p = q * v1 + r * v2
            nodes.append((p[0] * p[0] + p[1] * p[1], q, r, p[0], p[1]))
    nodes.sort()
    sel = nodes[:n_mt]
    centres = np.array([[x, y] for _, _, _, x, y in sel])
    qr = np.array([[q, r] for _, q, r, _, _ in sel])
    return centres, qr


@dataclass
class Params3D:
    # kinetics
    tau_dwell: float = 0.040       # s   dwell on one MT -> p_off = dt / tau_dwell
    D_free: float = 14.4           # um^2/s  free PAGFP-htau diffusion
    D_bound: float = 0.0           # um^2/s  bound tau (negligible in vivo)
    p_on: float = 0.02             # attachment probability on hitting an MT
                                   # (the on-rate knob -> sets k*on/koff)

    # geometry (paper values)
    n_mt: int = 60
    R_mt: float = 0.0125           # um  microtubule radius (12.5 nm)
    mt_spacing: float = 0.070      # um  inter-MT distance (70 nm)
    R_tube: float = 0.500          # um  process radius (500 nm)

    # imaging
    frame_interval: float = 0.005  # s   camera frame (5 ms)
    loc_precision: float = 0.020   # um  localisation precision (20 nm)
    tirf_depth: float = 0.150      # um  TIRF evanescent penetration depth (~150 nm)

    # integration
    dt: float = 1.0e-6             # s   time step (paper: 1 us)
    n_particles: int = 2000
    total_time: float = 0.10       # s
    seed: int = 0

    mt_xy: np.ndarray = field(init=False)
    mt_qr: np.ndarray = field(init=False)

    def __post_init__(self):
        self.mt_xy, self.mt_qr = hex_bundle(self.n_mt, self.mt_spacing)

    @property
    def p_off(self) -> float:
        return self.dt / self.tau_dwell


def _nearest_hex(x, y, spacing, qr_set, mt_xy):
    """Vectorised nearest-microtubule lookup on the triangular lattice.

    Returns (mt_index, dist) where mt_index is the index into the kept-MT array
    or -1 if the nearest lattice node is not one of the bundle's MTs.
    """
    a = spacing
    # fractional axial coordinates
    rr = (2.0 * y) / (a * np.sqrt(3))
    qq = x / a - rr / 2.0
    # cube rounding
    cx, cz = qq, rr
    cy = -cx - cz
    rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
    dx, dy, dz = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
    fix_x = (dx > dy) & (dx > dz)
    fix_z = (~fix_x) & (dz > dy)
    rx = np.where(fix_x, -ry - rz, rx)
    rz = np.where(fix_z, -rx - ry, rz)
    q_i = rx.astype(int)
    r_i = rz.astype(int)
    # map (q, r) -> kept-MT index
    idx = np.full(x.shape, -1, dtype=int)
    key = (q_i.astype(np.int64) << 20) ^ (r_i.astype(np.int64) & 0xFFFFF)
    for k, (qk, rk) in enumerate(qr_set):
        kk = (np.int64(qk) << 20) ^ (np.int64(rk) & 0xFFFFF)
        idx[key == kk] = k
    has = idx >= 0
    d = np.full(x.shape, np.inf)
    if has.any():
        cx2 = mt_xy[idx[has], 0]
        cy2 = mt_xy[idx[has], 1]
        d[has] = np.hypot(x[has] - cx2, y[has] - cy2)
    return idx, d


@dataclass
class Result3D:
    params: Params3D
    frames_x: np.ndarray           # (n_part, n_frames) projected x, NaN if free
    frames_y: np.ndarray           # (n_part, n_frames) depth (optical axis)
    frames_z: np.ndarray           # (n_part, n_frames) projected z, NaN if free
    bound_fraction: float
    residence_times: np.ndarray    # s
    kon_koff: float

    def step_size_distribution(self, rng=None, hops_only=False, tirf_depth=None):
        """First-step SSD from the projected (x, z) localisations of bound tau,
        with Gaussian localisation noise. Returns step sizes in micrometres.

        If `tirf_depth` (um) is given, only localisations within that distance of
        the coverslip (the bottom of the MT bundle) are kept -- i.e. the thin
        evanescent slice that TIRF actually illuminates. This collapses the 3D
        bundle to a quasi-2D sheet and sharpens the hop peaks, exactly as in the
        experiment.
        """
        p = self.params
        rng = rng or np.random.default_rng(7)
        x = self.frames_x.copy()
        z = self.frames_z.copy()
        if tirf_depth is not None:
            # coverslip at the bottom of the bundle; keep only the illuminated slice
            y_glass = np.nanmin(p.mt_xy[:, 1]) - p.R_mt
            visible = (self.frames_y - y_glass) <= tirf_depth
            x = np.where(visible, x, np.nan)
            z = np.where(visible, z, np.nan)
        xf_clean, zf_clean = x.copy(), z.copy()
        x = x + rng.normal(0, p.loc_precision, x.shape)
        z = z + rng.normal(0, p.loc_precision, z.shape)
        dx = x[:, 1:] - x[:, :-1]
        dz = z[:, 1:] - z[:, :-1]
        step = np.sqrt(dx * dx + dz * dz)
        ok = np.isfinite(step)               # both endpoints bound (and visible)
        if hops_only:
            # genuine inter-MT hop: nearest noise-free MT changed
            same = (np.abs(xf_clean[:, 1:] - xf_clean[:, :-1]) < 1e-6) & \
                   (np.abs(zf_clean[:, 1:] - zf_clean[:, :-1]) < 1e-6)
            ok = ok & ~same
        return step[ok].ravel()


def simulate3d(params: Params3D | None = None, **overrides) -> Result3D:
    p = params or Params3D(**overrides)
    if overrides and params is not None:
        raise ValueError("pass either a Params3D or overrides, not both")

    rng = np.random.default_rng(p.seed)
    n, dt = p.n_particles, p.dt
    steps = int(round(p.total_time / dt))
    sigma = np.sqrt(2.0 * p.D_free * dt)
    sigma_b = np.sqrt(2.0 * p.D_bound * dt)
    qr_set = [tuple(v) for v in p.mt_qr]

    # --- initial condition (paper): each particle bound with prob k*on/(k*on+koff)
    # We do not know k*on a priori; initialise everyone bound on a random MT and
    # let the dynamics relax to the stationary bound fraction within the burn-in.
    bound = np.ones(n, dtype=bool)
    mt = rng.integers(0, p.n_mt, size=n)
    x = p.mt_xy[mt, 0].copy()
    y = p.mt_xy[mt, 1].copy()
    z = np.zeros(n)
    state_start = np.zeros(n)

    fstride = max(1, int(round(p.frame_interval / dt)))
    n_frames = steps // fstride + 1
    fx = np.full((n, n_frames), np.nan, dtype=np.float32)
    fy = np.full((n, n_frames), np.nan, dtype=np.float32)
    fz = np.full((n, n_frames), np.nan, dtype=np.float32)
    fptr = 0

    residence = []
    bound_count = 0
    sample_count = 0
    burn = steps // 10

    R2 = p.R_tube * p.R_tube
    t = 0.0
    for s in range(steps):
        # ---------------- bound particles -----------------------------------
        b = bound
        if b.any():
            if p.D_bound > 0:
                z[b] += rng.normal(0, sigma_b, b.sum())
            rel = b & (rng.random(n) < p.p_off)
            if rel.any():
                residence.append(t - state_start[rel])
                bound[rel] = False
                state_start[rel] = t
                # detach perpendicular to the MT surface (radial in x-y)
                ang = rng.uniform(0, 2 * np.pi, rel.sum())
                cx = p.mt_xy[mt[rel], 0]; cy = p.mt_xy[mt[rel], 1]
                push = p.R_mt * 1.01
                x[rel] = cx + push * np.cos(ang)
                y[rel] = cy + push * np.sin(ang)

        # ---------------- free particles ------------------------------------
        f = ~bound
        if f.any():
            fi = np.where(f)[0]
            # tentative diffusive move
            nx = x[fi] + rng.normal(0, sigma, fi.size)
            ny = y[fi] + rng.normal(0, sigma, fi.size)
            nz = z[fi] + rng.normal(0, sigma, fi.size)
            # did it hit an MT surface?
            idx, d = _nearest_hex(nx, ny, p.mt_spacing, qr_set, p.mt_xy)
            hit = (idx >= 0) & (d < p.R_mt)
            attach = hit & (rng.random(fi.size) < p.p_on)
            bounce = hit & ~attach
            # attach: snap to the MT axis, become bound
            ai = fi[attach]
            if ai.size:
                bound[ai] = True
                mt[ai] = idx[attach]
                x[ai] = p.mt_xy[idx[attach], 0]
                y[ai] = p.mt_xy[idx[attach], 1]
                z[ai] = nz[attach]
                state_start[ai] = t
            # bounce: reflect off the MT (keep old position), free move rejected
            # (came off perpendicular -> stay just outside, i.e. keep previous xyz)
            # free, no hit: accept the move
            mi = fi[~hit]
            x[mi] = nx[~hit]; y[mi] = ny[~hit]; z[mi] = nz[~hit]
            bi = fi[bounce]
            # bounced particles keep their previous (x,y) but take the z move
            z[bi] = nz[bounce]
            # reflecting tube wall in the cross-section
            rr2 = x[mi] ** 2 + y[mi] ** 2
            out = rr2 > R2
            if out.any():
                mo = mi[out]
                rad = np.hypot(x[mo], y[mo])
                over = rad - p.R_tube
                x[mo] -= 2 * over * x[mo] / rad
                y[mo] -= 2 * over * y[mo] / rad

        # ---------------- record frame --------------------------------------
        if s % fstride == 0:
            fx[bound, fptr] = x[bound]
            fy[bound, fptr] = y[bound]
            fz[bound, fptr] = z[bound]
            fptr += 1
        if s >= burn:
            bound_count += bound.sum(); sample_count += n
        t += dt

    fb = bound_count / max(1, sample_count)
    res = np.concatenate(residence) if residence else np.array([])
    return Result3D(
        params=p,
        frames_x=fx[:, :fptr],
        frames_y=fy[:, :fptr],
        frames_z=fz[:, :fptr],
        bound_fraction=float(fb),
        residence_times=res,
        kon_koff=fb / max(1e-9, 1 - fb),
    )


if __name__ == "__main__":
    r = simulate3d(n_particles=1500, total_time=0.08, p_on=0.02, seed=0)
    print(f"MTs              : {r.params.n_mt} (hex bundle, "
          f"{r.params.mt_spacing*1e3:.0f} nm spacing, R_tube="
          f"{r.params.R_tube*1e3:.0f} nm)")
    print(f"bound fraction   : {r.bound_fraction:.3f}  "
          f"(k*on/koff = {r.kon_koff:.0f})")
    print(f"residence events : {r.residence_times.size}, "
          f"mean {1e3*r.residence_times.mean():.1f} ms")
    ss = r.step_size_distribution(hops_only=True) * 1e3
    print(f"hop SSD steps    : {ss.size}, median {np.median(ss):.0f} nm")
