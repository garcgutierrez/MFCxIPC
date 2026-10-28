"""Sequential re-timing of the pendulum (E3) and toggle-switch (E4) propagations on an idle core.

The original runs of e3_pendulum.py and e4_toggle.py shared the processor with the training of the
generic closures.  This script repeats every successful propagation (three times for the intrusive
closures, once for X-TFC), checks that the solution is unchanged and stores the median wall time in the
result pickles (field "wall"; the individual times in "wall_runs").
"""
import os
import pickle
import sys

import numpy as np
from scipy.stats import norm, qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from closures import SinCos, Power, ExactGalerkin, PSP, Taylor, Lanczos, PINNClosure
from problems import Pendulum, ToggleSwitch
from propagate import intrusive
from xtfc import SpectralXTFC

REPS = 3
NT = os.environ.get("OPENBLAS_NUM_THREADS", "1")
KEY = "wall" if NT == "1" else f"wall_par{NT}"           # multi-threaded runs are stored next to the 1-core ones


def retime(out, name, prob, cl, t_eval):
    walls = []
    for _ in range(REPS):
        C, wall, _, ok = intrusive(prob, cl, t_eval)
        walls.append(wall)
    dev = float(np.abs(C - out["methods"][name]["C"]).max())
    old = out["methods"][name]["wall"]
    out["methods"][name][KEY], out["methods"][name][KEY + "_runs"] = float(np.median(walls)), walls
    print(f"{name:14s} wall old {old:7.2f}s new {np.median(walls):7.2f}s  runs {np.round(walls, 2)}  "
          f"max|dC| {dev:.1e}", flush=True)


def retime_xtfc(out, name, xt, c0, T, t_eval):
    C = xt.propagate(c0, T, t_eval)
    dev = float(np.abs(C - out["methods"][name]["C"]).max())
    old = out["methods"][name]["wall"]
    out["methods"][name][KEY] = xt.stats["wall"]
    print(f"{name:14s} wall old {old:7.2f}s new {xt.stats['wall']:7.2f}s  max|dC| {dev:.1e}", flush=True)


def pendulum():
    f = os.path.join(RES, "e3_pendulum.pkl")
    out = pickle.load(open(f, "rb"))
    B = PCBasis(2, 6)
    prob, nl, t = Pendulum(B), SinCos(B), out["t"]
    for name, cl in [("eig", ExactGalerkin(nl)), ("psp", PSP(nl, 7)), ("lanczos", Lanczos(nl, 10))]:
        retime(out, name, prob, cl, t)
    X, W = B.quadrature(7)
    retime_xtfc(out, "xtfc", SpectralXTFC(prob, X, W, L=12, m=16, dt=1.0, gn_tol=1e-9), prob.pc_initial(),
                t[-1], t)
    pickle.dump(out, open(f, "wb"))


def toggle():
    f = os.path.join(RES, "e4_toggle.pkl")
    out = pickle.load(open(f, "rb"))
    B = PCBasis(4, 3)
    prob, t = ToggleSwitch(B, y0=(10.0, 0.5)), out["t"]
    mk = lambda g: dict(pow_beta=g(Power(B, prob.beta)), inv=g(Power(B, -1.0)))
    pinn = dict(pow_beta=PINNClosure.load(Power(B, prob.beta), os.path.join(MODELS, "pow2.5_d4p3.npz")),
                inv=PINNClosure.load(Power(B, -1.0), os.path.join(MODELS, "pow-1_d4p3.npz")))
    for name, cl in [("eig", mk(ExactGalerkin)), ("psp", mk(lambda nl: PSP(nl, 4))),
                     ("taylor4", mk(lambda nl: Taylor(nl, 4))), ("lanczos", mk(lambda nl: Lanczos(nl, 10))),
                     ("pinn_generic", pinn)]:
        retime(out, name, prob, cl, t)
    X, W = B.quadrature(4)
    variants = [("xtfc_gh", X, W)]
    for fac in [2, 4]:
        n = fac * B.P
        s = qmc.Sobol(4, scramble=True, seed=11).random(n)
        variants.append((f"xtfc_qmc{fac}P", norm.ppf(s), np.ones(n)))
    for name, Xn, Wn in variants:
        retime_xtfc(out, name, SpectralXTFC(prob, Xn, Wn, L=12, m=16, dt=0.5, gn_tol=1e-9), prob.pc_initial(),
                    t[-1], t)
    pickle.dump(out, open(f, "wb"))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    pendulum()
    toggle()
