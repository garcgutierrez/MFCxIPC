"""E13: initial guesses for the implicit laws when successive closure calls are far apart.

The closure is called along a converged trajectory at every s-th instant of a dense output grid (2001 instants;
s = 1 is close to the spacing of the integrator's own calls).  For large s the previous solution is a poor initial
guess, as with large implicit time steps or after abrupt changes of state; the first call of every sequence has no
previous solution at all.
  python e13_coherence.py traj <heat|osc>      dense converged trajectory (chord, tolerance 1e-10)
  python e13_coherence.py run <heat|osc>       every method and spacing, one after another (timed, one core)
"""
import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
from propagate import intrusive

N_T = 2001
SPACINGS = [1, 5, 25, 125, 500]
CALLS = 400                     # calls per spacing (s > 1): sequences with evenly spread start offsets


def _heat():
    import e8_enthalpy as e8
    from implicit_field import FieldSolver
    B, prob, fs = e8.setup()
    pn = e8.make_closure("pinn", prob, fs)
    methods = {
        "psp_s4": lambda: e8.make_closure("psp_s4", prob, fs),
        "nk_c8": lambda: e8.make_closure("nk_c8", prob, fs),
        "nk_w8": lambda: e8.make_closure("nk_w8", prob, fs),
        "chord_c8": lambda: FieldSolver(fs, method="chord", init="linear", tol=1e-8, m=10, record=True),
        "chord_w8": lambda: e8.make_closure("chord_w8", prob, fs),
        "pinn": lambda: pn,
        "pinn_nk2": lambda: e8.make_closure("pinn_nk2", prob, fs),
        "pinn_nkc8": lambda: FieldSolver(fs, method="nk", init=pn, tol=1e-8, m=10, record=True),
        "pinn_chord8": lambda: FieldSolver(fs, method="chord", init=pn, tol=1e-8, m=10, record=True),
    }
    call = lambda cl, H: cl(H, prob.K3)
    ref = lambda: FieldSolver(fs, method="newton", init="warm", tol=1e-12)    # converged, called in order
    traj = lambda: intrusive(prob, e8.make_closure("chord_w10", prob, fs), np.linspace(0, e8.T_END, N_T),
                             rtol=e8.RTOL, atol=e8.ATOL)
    return methods, call, ref, traj, lambda C: C


def _osc():
    import e7_series_spring as e7
    from implicit import CubicSolver
    B, prob = e7.setup()
    s = prob.sys
    net = e7.make_closure("pinn", prob)
    methods = {
        "psp": lambda: e7.make_closure("psp", prob),
        "newton_c8": lambda: e7.make_closure("newton_c8", prob),
        "newton_w8": lambda: e7.make_closure("newton_w8", prob),
        "chord_w8": lambda: e7.make_closure("chord_w8", prob),
        "nk_w8": lambda: e7.make_closure("nk_w8", prob),
        "pinn": lambda: net,
        "pinn_newton1": lambda: e7.make_closure("pinn_newton1", prob),
        "pinn_newtonc8": lambda: CubicSolver(s, method="newton", init=net, tol=1e-8, m=10, record=True),
        "pinn_chord8": lambda: CubicSolver(s, method="chord", init=net, tol=1e-8, m=10, record=True),
    }
    call = lambda cl, x: cl(x)
    ref = lambda: CubicSolver(s, method="newton", init="warm", tol=1e-12)
    traj = lambda: intrusive(prob, e7.make_closure("chord_w10", prob), np.linspace(0, e7.T_END, N_T))
    return methods, call, ref, traj, lambda C: C[:, 0, :]          # closure input: displacement coefficients


SETUP = dict(heat=_heat, osc=_osc)


def stage_traj(problem):
    _, _, _, traj, _ = SETUP[problem]()
    C, wall, nfev, ok = traj()
    np.savez(os.path.join(RES, f"e13_traj_{problem}.npz"), C=C, wall=wall, nfev=nfev, ok=ok)
    print(f"{problem}: trajectory {C.shape}, {wall:.1f}s, {nfev} RHS evaluations, ok={ok}", flush=True)


def sequences(s):
    """Lists of instants: one sequence for s = 1, otherwise sequences with evenly spread start offsets."""
    if s == 1:
        return [np.arange(N_T)]
    per = len(range(0, N_T, s))
    n_off = min(s, math.ceil(CALLS / per))
    return [np.arange(o, N_T, s) for o in np.unique(np.linspace(0, s - 1, n_off).round().astype(int))]


def rel_err(Y, R):
    Y, R = np.atleast_2d(Y), np.atleast_2d(R)
    return np.linalg.norm(Y - R, axis=1) / np.linalg.norm(R, axis=1)


def stage_run(problem):
    methods, call, ref, _, inputs = SETUP[problem]()
    X = inputs(np.load(os.path.join(RES, f"e13_traj_{problem}.npz"))["C"])
    t0 = time.perf_counter()
    rs = ref()
    R = [call(rs, X[i]) for i in range(N_T)]
    print(f"{problem}: reference closure at {N_T} instants in {time.perf_counter() - t0:.0f}s", flush=True)
    out = dict(spacings=SPACINGS, n_t=N_T, results={})
    for name, make in methods.items():
        for s in SPACINGS:
            t_call, first, iters, e_med, e_max = [], [], [], [], []
            for seq in sequences(s):
                cl = make()
                for j, i in enumerate(seq):
                    a = time.perf_counter()
                    Y = call(cl, X[i])
                    t_call.append(time.perf_counter() - a)
                    first.append(j == 0)
                    its = getattr(cl, "iters", None)
                    iters.append(float(its[-1]) if its else np.nan)
                    e = rel_err(Y, R[i])
                    e_med.append(float(np.median(e)))
                    e_max.append(float(e.max()))
            r = {k: np.array(v) for k, v in dict(t_call=t_call, first=first, iters=iters, e_med=e_med,
                                                 e_max=e_max).items()}
            out["results"][(name, s)] = r
            later = ~r["first"]
            print(f"{name:14s} s={s:4d}: {len(t_call):5d} calls, time/call median {np.median(r['t_call'][later]) * 1e3:8.2f} ms"
                  f" (first calls {np.median(r['t_call'][r['first']]) * 1e3:8.2f} ms), iters {np.nanmean(r['iters'][later]):5.2f},"
                  f" err median {np.median(r['e_med']):.1e} max {r['e_max'].max():.1e}", flush=True)
        with open(os.path.join(RES, f"e13_{problem}.pkl"), "wb") as fh:
            pickle.dump(out, fh)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    dict(traj=stage_traj, run=stage_run)[sys.argv[1]](*sys.argv[2:])
