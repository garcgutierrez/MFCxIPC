"""E10: network size and training budget of the neural closures (accuracy; the trainings run in parallel).

  python e10_capacity.py explicit <name> <width> <depth> <budget> [R_eff r_max]
  python e10_capacity.py enthalpy <width> <depth> <budget>
  python e10_capacity.py osc <width> <depth> <budget>
  python e10_capacity.py pendulum <model tag>          generic sine/cosine closure in the pendulum propagation

budget b multiplies the Adam iterations, the L-BFGS iterations (explicit closures: by min(b, 2)) and the number of
training samples of the baseline settings (Table S1).  Results: results/e10_<tag>.json, models/<tag>.npz.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from closures import ExactGalerkin, PINNClosure, sample_directions
from e1_closures import CASES


def _r_limit(pc, D, R_eff, r_max):
    """Training radius of the baseline closures, so that every network is tested on the same inputs."""
    old = pc.R_eff, pc.r_max
    pc.R_eff, pc.r_max = R_eff, r_max
    r = pc.r_limit(D)
    pc.R_eff, pc.r_max = old
    return r


def assess(pc, nl, B, R_eff, r_max, n_test=4000, seed=2024):
    rng = np.random.default_rng(seed)
    D = sample_directions(B, n_test, rng)
    r = _r_limit(pc, D, R_eff, r_max) * rng.random(n_test)
    Ge = ExactGalerkin(nl).normalised(r[:, None] * D)
    e = np.linalg.norm(pc.G(r, D) - Ge, axis=1) / np.linalg.norm(Ge, axis=1)
    return float(np.median(e)), float(np.percentile(e, 95))


def explicit(name, width, depth, budget, R_eff=None, r_max=2.0):
    width, depth, budget = int(width), int(depth), float(budget)
    case = [c for c in CASES if c[0] == name][0]
    _, d, p, mk, R0, K = case
    R_eff = float(R_eff) if R_eff else R0
    r_max = float(r_max)
    B = PCBasis(d, p)
    nl = mk(B)
    tag = f"{name}_w{width}d{depth}b{budget:g}" + (f"_R{R_eff:g}" if R_eff != R0 else "")
    pc = PINNClosure(nl, K=K, width=width, depth=depth, r_max=r_max, R_eff=R_eff, seed=0)
    t0 = time.perf_counter()
    pc.train(n_dirs=int(30000 * budget), batch=256, iters=int(8000 * budget), lr=3e-3,
             lbfgs_iters=int(2500 * min(budget, 2)), lbfgs_batch=4096, seed=1, verbose=True)
    t = time.perf_counter() - t0
    pc.save(os.path.join(MODELS, f"{tag}.npz"))
    out = dict(tag=tag, name=name, width=width, depth=depth, budget=budget, R_eff=R_eff, n_params=pc.net.n_params,
               t_train=t, final_val=float(pc.final_val_err))
    out["test_base"] = assess(pc, nl, B, R0, 2.0)                  # the Table-1 test distribution
    if R_eff != R0:
        out["test_own"] = assess(pc, nl, B, R_eff, r_max)           # its own (larger) training distribution
    json.dump(out, open(os.path.join(RES, f"e10_{tag}.json"), "w"))
    print(out, flush=True)


def enthalpy(width, depth, budget):
    import e8_enthalpy as e8
    from implicit_field import FieldPINN, FieldNewton
    width, depth, budget = int(width), int(depth), float(budget)
    B, prob, fs = e8.setup()
    H, K = e8.anchors()
    rng = np.random.default_rng(77)
    Hv, Kv = FieldPINN.sample(H, K, 300, rng)                       # same validation set as e8
    Yv = FieldNewton(fs)(Hv, Kv)
    val = lambda m: float(np.median(np.linalg.norm(m(Hv, Kv) - Yv, axis=1) / np.linalg.norm(Yv, axis=1)))
    tag = f"enthalpy_d8p3_w{width}d{depth}b{budget:g}"
    pn = FieldPINN(fs, width=width, depth=depth, seed=0)
    t0 = time.perf_counter()
    pn.train(H, K, n_pool=int(20000 * budget), batch=256, iters=int(4000 * budget), lr=2e-3,
             lbfgs_iters=int(300 * budget), lbfgs_batch=1024, seed=1, verbose=True, validate=val)
    t = time.perf_counter() - t0
    pn.save(os.path.join(MODELS, f"{tag}.npz"))
    out = dict(tag=tag, width=width, depth=depth, budget=budget, n_params=pn.net.n_params, t_train=t,
               val=float(pn.final_val), history=[list(h) for h in pn.history])
    json.dump(out, open(os.path.join(RES, f"e10_{tag}.json"), "w"))
    print({k: v for k, v in out.items() if k != "history"}, flush=True)


def osc(width, depth, budget):
    import e7_series_spring as e7
    from implicit import PINNImplicit, CardanoPSP
    from propagate import intrusive
    width, depth, budget = int(width), int(depth), float(budget)
    B, prob = e7.setup()
    s = prob.sys
    Cp, _, _, _ = intrusive(prob, CardanoPSP(s, e7.P_ORDER + 1), np.linspace(0, e7.T_END, 401))
    anchors = Cp[:, 0, :]
    Xv, Yv = e7.validation_set(s, anchors)
    val = lambda m: float(np.median(np.linalg.norm(m(Xv) - Yv, axis=1) / np.linalg.norm(Yv, axis=1)))
    tag = f"implicit_d6p3_w{width}d{depth}b{budget:g}"
    pn = PINNImplicit(s, width=width, depth=depth, seed=0)
    t0 = time.perf_counter()
    pn.train(anchors, n_pool=int(20000 * budget), batch=256, iters=int(6000 * budget), lr=2e-3,
             lbfgs_iters=int(2000 * min(budget, 2)), lbfgs_batch=2048, seed=1, verbose=True, validate=val)
    t = time.perf_counter() - t0
    pn.save(os.path.join(MODELS, f"{tag}.npz"))
    out = dict(tag=tag, width=width, depth=depth, budget=budget, n_params=pn.net.n_params, t_train=t,
               val=float(pn.final_val))
    json.dump(out, open(os.path.join(RES, f"e10_{tag}.json"), "w"))
    print(out, flush=True)


def pendulum(tag):
    """Propagate the pendulum with a (generic) sine/cosine closure; reports failure time and errors."""
    import pickle
    from problems import Pendulum
    from closures import SinCos
    from propagate import intrusive, compare, monte_carlo
    B = PCBasis(2, 6)
    prob, nl = Pendulum(B), SinCos(B)
    pc = PINNClosure.load(nl, os.path.join(MODELS, f"{tag}.npz"))
    o = pickle.load(open(os.path.join(RES, "e3_pendulum.pkl"), "rb"))
    t = o["t"]
    C, wall, nfev, ok = intrusive(prob, pc, t)
    xi = np.random.default_rng(2026).normal(size=(100_000, 2))
    Y = monte_carlo(prob, xi, t[:len(C)])
    m = compare(B, C, Y, B.eval_basis(xi))
    l2 = m["l2_err"][:, 0]
    idx = [i for i in (40, 80, 112, 120) if i < len(C)]
    out = dict(tag=tag, ok=bool(ok), t_end=float(t[len(C) - 1]), l2={f"{t[i]:g}": float(l2[i]) for i in idx})
    json.dump(out, open(os.path.join(RES, f"e10_pendulum_{tag}.json"), "w"))
    print(out, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    dict(explicit=explicit, enthalpy=enthalpy, osc=osc, pendulum=pendulum)[cmd](*sys.argv[2:])
