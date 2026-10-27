from dataclasses import dataclass, asdict
import time
import numpy as np
from scipy.fft import dctn, idctn

@dataclass(frozen=True)
class Params:
    tau: float = 0.35
    gamma: float = 0.1
    K: float = 0.3
    n: int = 1
    epsilon: float = 1.0
    fidelity: float = 5.0
    dt: float = 0.2
    T: float = 200.0
    stabilizer: float | None = None
    method: str = 'sbdf2'
    coefficient: str = 'dynamic'
    constant_a: float = 1.0
    spacing: float = 1.0
    record_every: int = 100

class Operators:

    def __init__(self, shape, p: Params, initial=None):
        self.shape = tuple(shape)
        self.p = p
        self.h = float(p.spacing)
        self.q = np.zeros(shape)
        q_cont = np.zeros(shape)
        for ax, N in enumerate(shape):
            sh = [1] * len(shape)
            sh[ax] = N
            k = np.arange(N)
            self.q += (4 / self.h ** 2 * np.sin(np.pi * k / (2 * N)) ** 2).reshape(sh)
            q_cont += (np.pi * k / (N * self.h)).reshape(sh) ** 2
        self.sm = np.exp(-0.5 * p.epsilon ** 2 * q_cont)
        self.frozen = None
        if p.coefficient == 'frozen':
            if initial is None:
                raise ValueError('Frozen coefficient requires initial data')
            self.frozen = self.dynamic_coefficient(initial)

    def transform(self, u):
        return dctn(u, type=2, norm='ortho', workers=1)

    def inverse(self, u):
        return idctn(u, type=2, norm='ortho', workers=1)

    def smooth(self, u):
        return self.inverse(self.transform(u) * self.sm)

    def grad_sq(self, u):
        out = np.zeros_like(u)
        for ax in range(u.ndim):
            der = np.empty_like(u)
            mid = [slice(None)] * u.ndim
            lo = mid.copy()
            hi = mid.copy()
            mid[ax] = slice(1, -1)
            lo[ax] = slice(None, -2)
            hi[ax] = slice(2, None)
            der[tuple(mid)] = (u[tuple(hi)] - u[tuple(lo)]) / (2 * self.h)
            at0 = [slice(None)] * u.ndim
            at1 = at0.copy()
            at0[ax] = 0
            at1[ax] = 1
            der[tuple(at0)] = (u[tuple(at1)] - u[tuple(at0)]) / (2 * self.h)
            at0[ax] = -1
            at1[ax] = -2
            der[tuple(at0)] = (u[tuple(at0)] - u[tuple(at1)]) / (2 * self.h)
            out += der * der
        return out

    def dynamic_coefficient(self, u):
        p = self.p
        if p.gamma == 1:
            return np.ones_like(u)
        s = self.grad_sq(self.smooth(u))
        return p.gamma + (1 - p.gamma) / (1 + s / p.K ** 2) ** p.n

    def coefficient(self, u):
        if self.p.coefficient == 'constant':
            return self.p.constant_a
        if self.p.coefficient == 'frozen':
            return self.frozen
        return self.dynamic_coefficient(u)

    def lap(self, u, a=1.0):
        out = np.zeros_like(u)
        scalar = np.isscalar(a)
        for ax in range(u.ndim):
            lo = [slice(None)] * u.ndim
            hi = lo.copy()
            lo[ax] = slice(None, -1)
            hi[ax] = slice(1, None)
            lo, hi = (tuple(lo), tuple(hi))
            face = a if scalar else 0.5 * (a[lo] + a[hi])
            flux = face * (u[hi] - u[lo]) / self.h ** 2
            out[lo] += flux
            out[hi] -= flux
        return out

    def rhs(self, u, obs, lam):
        w = 4 * self.p.tau * (u * u * u - u) - self.lap(u, self.coefficient(u)) / self.p.tau
        return self.lap(w) + lam * (obs - u)

def solve(u0, obs, mask, params=None, *, fidelity_array=None, save_times=None):
    p = params or Params()
    u = np.asarray(u0, dtype=float).copy()
    obs = np.asarray(obs, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if u.shape != obs.shape or u.shape != mask.shape:
        raise ValueError('Shape mismatch')
    if not (p.tau > 0 and 0 < p.gamma <= 1 and (p.K > 0) and (p.epsilon > 0) and (p.spacing > 0) and (p.dt > 0) and (p.T >= 0)):
        raise ValueError('Invalid positive regularization or discretization parameter')
    if p.coefficient not in ('constant', 'dynamic', 'frozen'):
        raise ValueError('Invalid coefficient')
    if p.constant_a <= 0 or p.fidelity < 0:
        raise ValueError('Invalid diffusion/fidelity')
    if p.method not in ('euler', 'sbdf2'):
        raise ValueError('Invalid method')
    lam = p.fidelity * ~mask if fidelity_array is None else np.asarray(fidelity_array, float)
    lam = np.broadcast_to(lam, u.shape).copy()
    if np.min(lam) < 0:
        raise ValueError('Negative fidelity')
    lmax = float(np.max(lam))
    op = Operators(u.shape, p, initial=u)
    a0 = max(1.0, p.constant_a) if p.coefficient == 'constant' else 1.0
    stab = 16 * p.tau if p.stabilizer is None else p.stabilizer
    A = a0 * op.q ** 2 / p.tau + stab * op.q + lmax
    steps = max(1, int(np.ceil(p.T / p.dt))) if p.T else 0
    dt = p.T / steps if steps else p.dt
    den1 = 1 + dt * A
    den2 = 1.5 + dt * A

    def explicit(v):
        inner = 4 * p.tau * (v * v * v - v) - stab * v - op.lap(v, op.coefficient(v) - a0) / p.tau
        return op.lap(inner) + (lmax - lam) * v + lam * obs
    history = []

    def record(step, v):
        rhs = op.rhs(v, obs, lam)
        history.append(dict(step=int(step), time=float(step * dt), mean=float(v.mean()), maximum=float(v.max()), minimum=float(v.min()), rhs_rms=float(np.sqrt(np.mean(rhs ** 2))), mass_rhs_residual=float(rhs.mean() - np.mean(lam * (obs - v)))))
    record(0, u)
    started = time.perf_counter()
    snapshots = {} if save_times is not None else None
    next_times = sorted((float(t) for t in save_times or []))
    if next_times and next_times[0] <= 0:
        snapshots[0.0] = u.copy()
        next_times = [t for t in next_times if t > 0]
    u_prev = None
    r_prev = None
    for step in range(1, steps + 1):
        r = explicit(u)
        if p.method == 'euler' or step == 1:
            new = op.inverse(op.transform(u + dt * r) / den1)
        else:
            new = op.inverse(op.transform(2 * u - 0.5 * u_prev + dt * (2 * r - r_prev)) / den2)
        u_prev, u = (u, new)
        r_prev = r
        if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 5:
            raise FloatingPointError(f'Unstable iterate at step {step}, dt={dt:g}, max={np.nanmax(np.abs(u)):g}')
        if p.record_every and (step % p.record_every == 0 or step == steps):
            record(step, u)
        while next_times and step * dt >= next_times[0] - 1e-12:
            snapshots[float(step * dt)] = u.copy()
            next_times.pop(0)
    diag = dict(parameters=asdict(p), actual_dt=float(dt), steps=steps, seconds=time.perf_counter() - started, history=history, final_rhs_rms=history[-1]['rhs_rms'] if history[-1]['step'] == steps else float(np.sqrt(np.mean(op.rhs(u, obs, lam) ** 2))), final_min=float(u.min()), final_max=float(u.max()), final_mean=float(u.mean()))
    if snapshots is not None:
        diag['snapshots'] = snapshots
    return (u, diag)
