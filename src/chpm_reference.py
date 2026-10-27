from dataclasses import dataclass
import numpy as np
from scipy.fft import dctn, idctn
from scipy.integrate import solve_ivp

@dataclass(frozen=True)
class Parameters:
    tau: float = 2.0
    gamma: float = 0.2
    K: float = 0.5
    n: int = 1
    epsilon: float = 0.08

class CHPM:

    def __init__(self, shape, lengths=None, parameters=None, fidelity=0.0, observation=0.0, source=None):
        self.shape = tuple(shape)
        self.lengths = tuple(lengths or (1.0,) * len(self.shape))
        self.p = parameters or Parameters()
        self.h = tuple((L / N for L, N in zip(self.lengths, self.shape)))
        self.fidelity = np.broadcast_to(fidelity, self.shape).copy()
        self.observation = np.broadcast_to(observation, self.shape).copy()
        self.source = source
        assert self.p.tau > 0 and self.p.K > 0 and (self.p.epsilon > 0)
        assert 0 < self.p.gamma <= 1 and self.p.n >= 1
        q = np.zeros(self.shape)
        for ax, (N, L) in enumerate(zip(self.shape, self.lengths)):
            size = [1] * len(self.shape)
            size[ax] = N
            q += (np.pi * np.arange(N) / L).reshape(size) ** 2
        self.smoothing_multiplier = np.exp(-0.5 * self.p.epsilon ** 2 * q)

    def smooth(self, u):
        return idctn(dctn(u, type=2, norm='ortho') * self.smoothing_multiplier, type=2, norm='ortho')

    def gradient_squared(self, u):
        total = np.zeros_like(u)
        for ax, h in enumerate(self.h):
            pad = [(0, 0)] * u.ndim
            pad[ax] = (1, 1)
            v = np.pad(u, pad, mode='symmetric')
            lo = [slice(None)] * u.ndim
            hi = [slice(None)] * u.ndim
            lo[ax] = slice(0, -2)
            hi[ax] = slice(2, None)
            total += ((v[tuple(hi)] - v[tuple(lo)]) / (2 * h)) ** 2
        return total

    def coefficient(self, u):
        s = self.gradient_squared(self.smooth(u))
        p = self.p
        return p.gamma + (1 - p.gamma) / (1 + s / p.K ** 2) ** p.n

    def divergence(self, u, a=1.0):
        a = np.broadcast_to(a, u.shape)
        out = np.zeros_like(u)
        for ax, h in enumerate(self.h):
            lo = [slice(None)] * u.ndim
            hi = [slice(None)] * u.ndim
            lo[ax] = slice(None, -1)
            hi[ax] = slice(1, None)
            lo, hi = (tuple(lo), tuple(hi))
            flux = 0.5 * (a[lo] + a[hi]) * (u[hi] - u[lo]) / h
            out[lo] += flux / h
            out[hi] -= flux / h
        return out

    def chemical_potential(self, u):
        return 4 * self.p.tau * (u ** 3 - u) - self.divergence(u, self.coefficient(u)) / self.p.tau

    def rhs(self, t, y):
        u = np.asarray(y).reshape(self.shape)
        out = self.divergence(self.chemical_potential(u))
        out += self.fidelity * (self.observation - u)
        if self.source is not None:
            out += self.source(t)
        return out.ravel()

    def solve(self, initial, T, rtol=1e-08, atol=1e-10, **kwargs):
        sol = solve_ivp(self.rhs, (0, T), np.asarray(initial).ravel(), method='BDF', rtol=rtol, atol=atol, **kwargs)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol
