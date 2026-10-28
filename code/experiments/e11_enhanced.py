"""E11: Fourier-feature embeddings and residual-based adaptive sampling (RAD) for the neural closures, at the network
size and training budget of the baselines.

  python e11_enhanced.py explicit <name> <variant> [frac]     generic closures (exp_d2p4, sincos_d2p6)
  python e11_enhanced.py enthalpy <variant> [frac]            heat-conduction network
  python e11_enhanced.py osc <variant> [frac]                 oscillator network
  python e11_enhanced.py best_sigma <problem>                 sigma with the lowest validation error of the sweep

variant: "P" (plain, = baseline settings), "F<sigma>" (64 random Fourier features of scale sigma), "A" (RAD with
k = c = 1, training set redrawn every 1/8 of the Adam iterations from twice as many candidates), or "F<sigma>A".
frac < 1 scales the Adam and L-BFGS iterations (short runs used only to choose sigma).
Results: results/e11_<tag>.json, models/<tag>.npz (implicit networks: enthalpy_d8p3_<variant>.npz,
implicit_d6p3_<variant>.npz, so that e7/e8 can propagate them as method@<variant>).
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from closures import PINNClosure
from e1_closures import CASES
from e10_capacity import assess

M_FOURIER = 64


def parse(variant):
    m = re.fullmatch(r"(P)|(?:F([\d.]+))?(A)?", variant)
    if not m or variant == "":
        raise ValueError(variant)
    sigma = None if m.group(2) is None else float(m.group(2))
    return (None if sigma is None else (M_FOURIER, sigma)), (dict() if m.group(3) else None), sigma


def _tag(base, variant, frac):
    return f"{base}_{variant}" + ("" if float(frac) == 1 else f"_s{float(frac):g}")


def _dump(out):
    json.dump(out, open(os.path.join(RES, f"e11_{out['tag']}.json"), "w"))
    print({k: v for k, v in out.items() if k != "history"}, flush=True)


def explicit(name, variant, frac=1):
    frac = float(frac)
    fourier, adaptive, sigma = parse(variant)
    _, d, p, mk, R_eff, K = [c for c in CASES if c[0] == name][0]
    B = PCBasis(d, p)
    nl = mk(B)
    tag = _tag(name, variant, frac)
    pc = PINNClosure(nl, K=K, width=64, depth=3, r_max=2.0, R_eff=R_eff, seed=0, fourier=fourier)
    t0 = time.perf_counter()
    pc.train(n_dirs=30000, batch=256, iters=int(8000 * frac), lr=3e-3, lbfgs_iters=int(2500 * frac),
             lbfgs_batch=4096, seed=1, verbose=True, adaptive=adaptive)
    t = time.perf_counter() - t0
    pc.save(os.path.join(MODELS, f"{tag}.npz"))
    _dump(dict(tag=tag, problem=name, variant=variant, sigma=sigma, adaptive=adaptive is not None, frac=frac,
               n_params=pc.net.n_params, t_train=t, val=float(pc.final_val_err),
               test_base=assess(pc, nl, B, R_eff, 2.0)))


def enthalpy(variant, frac=1):
    import e8_enthalpy as e8
    from implicit_field import FieldPINN, FieldNewton
    frac = float(frac)
    fourier, adaptive, sigma = parse(variant)
    B, prob, fs = e8.setup()
    H, K = e8.anchors()
    rng = np.random.default_rng(77)
    Hv, Kv = FieldPINN.sample(H, K, 300, rng)                       # validation set of e8
    Yv = FieldNewton(fs)(Hv, Kv)
    val = lambda m: float(np.median(np.linalg.norm(m(Hv, Kv) - Yv, axis=1) / np.linalg.norm(Yv, axis=1)))
    tag = _tag("enthalpy_d8p3", variant, frac)
    pn = FieldPINN(fs, width=128, depth=3, seed=0, fourier=fourier)
    t0 = time.perf_counter()
    pn.train(H, K, n_pool=20000, batch=256, iters=int(4000 * frac), lr=2e-3, lbfgs_iters=int(300 * frac),
             lbfgs_batch=1024, seed=1, verbose=True, validate=val, adaptive=adaptive)
    t = time.perf_counter() - t0
    pn.save(os.path.join(MODELS, f"{tag}.npz"))
    _dump(dict(tag=tag, problem="enthalpy", variant=variant, sigma=sigma, adaptive=adaptive is not None, frac=frac,
               n_params=pn.net.n_params, t_train=t, val=float(pn.final_val)))


def osc(variant, frac=1):
    import e7_series_spring as e7
    from implicit import PINNImplicit, CardanoPSP
    from propagate import intrusive
    frac = float(frac)
    fourier, adaptive, sigma = parse(variant)
    B, prob = e7.setup()
    s = prob.sys
    Cp, _, _, _ = intrusive(prob, CardanoPSP(s, e7.P_ORDER + 1), np.linspace(0, e7.T_END, 401))
    anchors = Cp[:, 0, :]
    Xv, Yv = e7.validation_set(s, anchors)
    val = lambda m: float(np.median(np.linalg.norm(m(Xv) - Yv, axis=1) / np.linalg.norm(Yv, axis=1)))
    tag = _tag("implicit_d6p3", variant, frac)
    pn = PINNImplicit(s, width=64, depth=3, seed=0, fourier=fourier)
    t0 = time.perf_counter()
    pn.train(anchors, n_pool=20000, batch=256, iters=int(6000 * frac), lr=2e-3, lbfgs_iters=int(2000 * frac),
             lbfgs_batch=2048, seed=1, verbose=True, validate=val, adaptive=adaptive)
    t = time.perf_counter() - t0
    pn.save(os.path.join(MODELS, f"{tag}.npz"))
    _dump(dict(tag=tag, problem="osc", variant=variant, sigma=sigma, adaptive=adaptive is not None, frac=frac,
               n_params=pn.net.n_params, t_train=t, val=float(pn.final_val)))


def best_sigma(problem):
    """Prints the sigma of the short-run sweep with the lowest validation error (used by run_v7_enhanced.sh)."""
    base = {"exp_d2p4": "exp_d2p4", "sincos_d2p6": "sincos_d2p6", "enthalpy": "enthalpy_d8p3",
            "osc": "implicit_d6p3"}[problem]
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(RES, f"e11_{base}_F*_s*.json"))]
    rows = [r for r in rows if not r["adaptive"]]
    best = min(rows, key=lambda r: r["val"])
    print(f"{best['sigma']:g}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    dict(explicit=explicit, enthalpy=enthalpy, osc=osc, best_sigma=best_sigma)[cmd](*sys.argv[2:])
