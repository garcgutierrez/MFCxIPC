"""E3c: statistics of the anchored PINN closure on the pendulum up to the time the Galerkin system fails."""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from closures import SinCos, ExactGalerkin, PINNClosure
from problems import Pendulum
from propagate import monte_carlo, intrusive, compare
from e3_pendulum import D, P_ORDER, T_END, N_MC, closure_error_along

B = PCBasis(D, P_ORDER)
prob = Pendulum(B)
nl = SinCos(B)
t_eval = np.linspace(0, T_END, 121)
xi = np.random.default_rng(2026).normal(size=(N_MC, D))
Ymc = monte_carlo(prob, xi, t_eval)
Psi = B.eval_basis(xi)
out = pickle.load(open(os.path.join(RES, "e3_pendulum.pkl"), "rb"))
for name, f in [("pinn_anchored", "sincos_d2p6_pendulum_sc.npz"), ("pinn_anchored_r0", "sincos_d2p6_pendulum.npz")]:
    pc = PINNClosure.load(nl, os.path.join(MODELS, f))
    C, wall, nfev, ok = intrusive(prob, pc, t_eval)
    n = len(C)
    m = compare(B, C, Ymc[:n], Psi)
    d = dict(ok=ok, partial=not ok, t_fail=float(t_eval[n - 1]), C=C, wall=wall, nfev=nfev, **m,
             closure_err=closure_error_along(prob, out["methods"]["eig"]["C"], pc, ExactGalerkin(nl)))
    out["methods"][name] = d
    i = [k for k in [40, 80, 112] if k < n]
    print(f"{name}: reached t={d['t_fail']:.2f}, L2(theta) at t={t_eval[i]}: {m['l2_err'][i, 0].round(4)}; "
          f"closure err along exact traj median {np.median(d['closure_err']):.2e}, "
          f"at t<=25 max {d['closure_err'][t_eval <= 25].max():.2e}, max {d['closure_err'].max():.2e}", flush=True)
with open(os.path.join(RES, "e3_pendulum.pkl"), "wb") as fh:
    pickle.dump(out, fh)
