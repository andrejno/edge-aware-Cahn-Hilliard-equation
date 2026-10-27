import numpy as np
import pandas as pd
from dataclasses import replace
from chpm_reference import CHPM, Parameters
from chpm_solver import Params, Operators, solve

P = Parameters(tau=2.0, gamma=0.2, K=0.5, n=1, epsilon=0.08)

def exact_terms(x, t):
    k = np.pi
    b = 0.2 * np.exp(-t)
    u = 0.1 + b * np.cos(k * x)
    p = -b * k * np.sin(k * x)
    p1 = -b * k * k * np.cos(k * x)
    p2 = b * k ** 3 * np.sin(k * x)
    p3 = b * k ** 4 * np.cos(k * x)
    r = np.exp(-0.5 * P.epsilon ** 2 * k * k)
    C = (r / P.K) ** 2
    s = C * p * p
    s1 = 2 * C * p * p1
    s2 = 2 * C * (p1 * p1 + p * p2)
    s3 = 2 * C * (3 * p1 * p2 + p * p3)
    d = 1 + s
    a = P.gamma + (1 - P.gamma) / d
    a1 = -(1 - P.gamma) * s1 / d ** 2
    a2 = (1 - P.gamma) * (2 * s1 * s1 / d ** 3 - s2 / d ** 2)
    a3 = (1 - P.gamma) * (-6 * s1 ** 3 / d ** 4 + 6 * s1 * s2 / d ** 3 - s3 / d ** 2)
    wxx = 4 * P.tau * ((3 * u * u - 1) * p1 + 6 * u * p * p)
    wxx -= (a * p3 + 3 * a1 * p2 + 3 * a2 * p1 + a3 * p) / P.tau
    ut = -b * np.cos(k * x)
    return (u, ut - wxx)

def manufactured():
    rows = []
    for n in [16, 32, 64, 128]:
        x = (np.arange(n) + .5) / n
        model = CHPM((n,), parameters=P, source=lambda t, x=x: exact_terms(x, t)[1])
        solution = model.solve(exact_terms(x, 0)[0], .05, rtol=2e-10, atol=2e-12)
        error = np.sqrt(np.mean((solution.y[:, -1] - exact_terms(x, .05)[0]) ** 2))
        order = np.log2(rows[-1]['L2_error'] / error) if rows else np.nan
        rows.append({'N': n, 'h': 1 / n, 'L2_error': error, 'Order': order})
    tight = model.solve(exact_terms(x, 0)[0], .05, rtol=1e-10, atol=1e-12)
    difference = np.sqrt(np.mean((tight.y[:, -1] - solution.y[:, -1]) ** 2))
    return pd.DataFrame(rows), float(difference)

def temporal():
    n = 12
    y, x = np.mgrid[:n, :n]
    observed = .15 + .25 * np.cos(np.pi * (x + .5) / n) * np.cos(np.pi * (y + .5) / n)
    mask = (x >= 4) & (x < 8) & (y >= 3) & (y < 9)
    observed[mask] = 0
    initial = observed.copy()
    p = Params(tau=.35, gamma=.15, K=.3, n=1, epsilon=1., fidelity=.8, dt=.1, T=2., record_every=0)
    operator = Operators(initial.shape, p)
    reference = CHPM(initial.shape, lengths=(n, n), parameters=Parameters(p.tau, p.gamma, p.K, p.n, p.epsilon), fidelity=p.fidelity * (~mask), observation=observed)
    rhs_difference = np.max(np.abs(operator.rhs(initial, observed, p.fidelity * (~mask)) - reference.rhs(0, initial.ravel()).reshape(initial.shape)))
    solution = reference.solve(initial, p.T, rtol=2e-11, atol=2e-13)
    final_reference = solution.y[:, -1].reshape(initial.shape)
    rows = []
    for dt in [.2, .1, .05, .025, .0125, .00625, .003125, .0015625]:
        u, diagnostics = solve(initial, observed, mask, replace(p, dt=dt))
        error = np.sqrt(np.mean((u - final_reference) ** 2))
        order = np.log2(rows[-1]['RMS_error'] / error) if rows else np.nan
        rows.append({'dt': dt, 'RMS_error': error, 'Order': order})
    rng = np.random.default_rng(20261026)
    u0 = .2 * rng.normal(size=(64, 64))
    u, diagnostics = solve(u0, np.zeros_like(u0), np.ones_like(u0, dtype=bool), replace(p, T=10, dt=.2))
    mean_difference = abs(u.mean() - u0.mean())
    return pd.DataFrame(rows), float(rhs_difference), float(mean_difference)

def mass():
    rng = np.random.default_rng(20261026)
    shape = (13, 17)
    u = rng.normal(scale=.2, size=shape)
    fidelity = 3 * (rng.random(shape) > .35)
    observation = rng.uniform(-1, 1, size=shape)
    model = CHPM(shape, parameters=P, fidelity=fidelity, observation=observation)
    residual = abs(model.rhs(0, u.ravel()).mean() - (fidelity * (observation - u)).mean())
    constant = CHPM((16,), parameters=P, fidelity=2., observation=.3)
    solution = constant.solve(np.full(16, -.4), 1., rtol=2e-10, atol=2e-12, t_eval=np.linspace(0, 1, 51))
    exact = .3 + (-.4 - .3) * np.exp(-2 * solution.t)
    error = np.max(np.abs(solution.y - exact[None, :]))
    return float(residual), float(error)
