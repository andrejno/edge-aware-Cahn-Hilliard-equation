from functools import lru_cache
from time import perf_counter
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

@lru_cache(maxsize=8)
def neumann_laplacian(shape):
    ny, nx = map(int, shape)

    def one(n):
        diagonal = np.full(n, 2.0)
        diagonal[[0, -1]] = 1.0
        if n == 1:
            diagonal[0] = 0.0
        return sparse.diags([-np.ones(n - 1), diagonal, -np.ones(n - 1)], [-1, 0, 1], format='csr')
    return sparse.kron(sparse.eye(ny), one(nx), format='csr') + sparse.kron(one(ny), sparse.eye(nx), format='csr')

def _elliptic(observed, mask, power):
    start = perf_counter()
    f = np.asarray(observed, dtype=float)
    missing = np.asarray(mask, dtype=bool)
    if f.shape != missing.shape or f.ndim != 2:
        raise ValueError('observed and mask must be two-dimensional with identical shape')
    if missing.all():
        raise ValueError('at least one observed pixel is required')
    u = f.copy()
    if not missing.any():
        return (u, {'seconds': perf_counter() - start, 'residual': 0.0, 'method': power})
    L = neumann_laplacian(f.shape)
    A = L if power == 1 else (L @ L).tocsr()
    ids = np.flatnonzero(missing.ravel())
    known = f.ravel().copy()
    known[ids] = 0.0
    rhs = -(A @ known)[ids]
    interior = A[ids][:, ids].tocsc()
    x = spsolve(interior, rhs)
    u.ravel()[ids] = x
    residual = float(np.linalg.norm(interior @ x - rhs, np.inf))
    return (u, {'seconds': perf_counter() - start, 'residual': residual, 'method': 'harmonic' if power == 1 else 'biharmonic'})

def harmonic(observed, mask):
    return _elliptic(observed, mask, 1)

def biharmonic(observed, mask):
    return _elliptic(observed, mask, 2)

def gradient(u):
    gy = np.zeros_like(u)
    gx = np.zeros_like(u)
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    return (gy, gx)

def divergence(py, px):
    out = py + px
    out = out.copy()
    out[1:, :] -= py[:-1, :]
    out[:, 1:] -= px[:, :-1]
    return out

def tv_inpaint(observed, mask, iterations=4000, tolerance=2e-06, min_iterations=300):
    start = perf_counter()
    f = np.asarray(observed, dtype=float)
    missing = np.asarray(mask, dtype=bool)
    if f.ndim != 2 or f.shape != missing.shape:
        raise ValueError('observed and mask must be two-dimensional with identical shape')
    if missing.all():
        raise ValueError('at least one observed pixel is required')
    u = f.copy()
    u[missing] = 0.0
    ubar = u.copy()
    py = np.zeros_like(u)
    px = np.zeros_like(u)
    step = 0.3
    residual = float('inf')
    converged = False
    for k in range(int(iterations)):
        gy, gx = gradient(ubar)
        qy = py + step * gy
        qx = px + step * gx
        norm = np.maximum(1.0, np.sqrt(qy * qy + qx * qx))
        qy /= norm
        qx /= norm
        unew = u + step * divergence(qy, qx)
        unew[~missing] = f[~missing]
        if (k + 1) % 50 == 0:
            primal = float(np.linalg.norm(unew - u) / (step * max(1.0, np.linalg.norm(u))))
            dual = float(np.sqrt(np.sum((qy - py) ** 2 + (qx - px) ** 2)) / (step * max(1.0, np.sqrt(np.sum(py * py + px * px)))))
            residual = max(primal, dual)
            if k + 1 >= min_iterations and residual < tolerance:
                u = unew
                py, px = (qy, qx)
                converged = True
                break
        ubar = 2 * unew - u
        u = unew
        py, px = (qy, qx)
    gy, gx = gradient(u)
    tv_value = float(np.sqrt(gy * gy + gx * gx).sum())
    dual_div = divergence(py, px)
    ids = np.flatnonzero(missing.ravel())
    potential = np.zeros_like(u)
    if ids.size:
        L = neumann_laplacian(u.shape)
        potential.ravel()[ids] = spsolve(L[ids][:, ids].tocsc(), dual_div.ravel()[ids])
    gy_c, gx_c = gradient(potential)
    py_feasible = py + gy_c
    px_feasible = px + gx_c
    scale = max(1.0, float(np.sqrt(py_feasible ** 2 + px_feasible ** 2).max()))
    py_feasible /= scale
    px_feasible /= scale
    divergence_feasible = divergence(py_feasible, px_feasible)
    dual_lower_bound = -float(np.sum(f[~missing] * divergence_feasible[~missing]))
    gap = max(0.0, tv_value - dual_lower_bound)
    dual_feasibility = float(np.max(np.abs(divergence_feasible[missing]))) if missing.any() else 0.0
    return (u, {'seconds': perf_counter() - start, 'iterations': k + 1, 'fixed_point_residual': residual, 'converged': converged, 'tv': tv_value, 'dual_lower_bound': dual_lower_bound, 'primal_dual_gap': gap, 'relative_primal_dual_gap': gap / max(1.0, tv_value), 'dual_feasibility': dual_feasibility, 'method': 'constrained-TV'})
tv = tv_inpaint
