"""E1: train generic (problem-independent) PINN closures and assess them at closure level."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from closures import Exp, SinCos, Power, ExactGalerkin, PSP, PINNClosure, sample_directions

CASES = [  # name, d, p, nonlinearity, R_eff, K
    ("exp_d2p4", 2, 4, lambda B: Exp(B), 4.0, 3),
    ("exp_d5p3", 5, 3, lambda B: Exp(B), 4.0, 3),
    ("sincos_d2p6", 2, 6, lambda B: SinCos(B), 6.0, 3),
    ("pow2.5_d4p3", 4, 3, lambda B: Power(B, 2.5), 3.0, 3),
    ("pow-1_d4p3", 4, 3, lambda B: Power(B, -1.0), 3.0, 3),
]


def spectral_radius(B, X):
    lam = np.concatenate([np.linalg.eigvalsh(B.mult_matrix(X[i:i + 1000]))[:, [0, -1]]
                          for i in range(0, len(X), 1000)])
    return np.abs(lam).max(1), lam[:, 0]


def assess(pc, nl, B, n_test=4000, seed=2024):
    rng = np.random.default_rng(seed)
    D = sample_directions(B, n_test, rng)
    r = pc.r_limit(D) * rng.random(n_test)
    X = r[:, None] * D
    rho, lmin = spectral_radius(B, X)
    Ge = ExactGalerkin(nl).normalised(X)
    nrm = np.linalg.norm(Ge, axis=1)
    out = dict(rho=rho.tolist(), lmin=lmin.tolist())
    out["pinn"] = (np.linalg.norm(pc.G(r, D) - Ge, axis=1) / nrm).tolist()
    for K in [2, 4, 8]:
        T = sum(nl.taylor_terms(X, K - 1)[:K + 1])
        out[f"taylor{K}"] = (np.linalg.norm(T - Ge, axis=1) / nrm).tolist()
    # pseudo-spectral projection of the same normalised input (different closure definition)
    psp = PSP(nl, B.p + 1)
    u = X.copy(); u[:, 0] = 0.0 if nl.c == 0 else 1.0
    if nl.c != 0:
        u = X + B.e0
    Gp = psp(u)
    out["psp_vs_galerkin"] = (np.linalg.norm(Gp - Ge, axis=1) / nrm).tolist()
    return out


def main(which=None):
    summary = {}
    for name, d, p, mk, R_eff, K in CASES:
        if which and name not in which:
            continue
        B = PCBasis(d, p)
        nl = mk(B)
        print(f"=== {name}: P={B.P} nnz={B.nnz}", flush=True)
        pc = PINNClosure(nl, K=K, width=64, depth=3, r_max=2.0, R_eff=R_eff, seed=0)
        t0 = time.perf_counter()
        pc.train(n_dirs=30000, batch=256, iters=8000, lr=3e-3, lbfgs_iters=2500, lbfgs_batch=4096,
                 seed=1, verbose=True)
        t_train = time.perf_counter() - t0
        pc.save(os.path.join(MODELS, f"{name}.npz"))
        res = assess(pc, nl, B)
        res.update(name=name, d=d, p=p, P=B.P, K=K, R_eff=R_eff, t_train=t_train,
                   n_params=pc.net.n_params, history=[list(h) for h in pc.history])
        with open(os.path.join(RES, f"e1_{name}.json"), "w") as fh:
            json.dump(res, fh)
        e = np.array(res["pinn"])
        summary[name] = dict(t_train=t_train, med=float(np.median(e)), p95=float(np.percentile(e, 95)))
        print(f"  train {t_train:.0f}s  PINN median {np.median(e):.2e}  p95 {np.percentile(e, 95):.2e}", flush=True)
        for k in ["taylor2", "taylor4", "taylor8", "psp_vs_galerkin"]:
            print(f"  {k:16s} median {np.median(res[k]):.2e}  p95 {np.percentile(res[k], 95):.2e}", flush=True)
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
