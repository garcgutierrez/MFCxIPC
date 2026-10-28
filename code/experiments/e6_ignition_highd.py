"""E6: ignition PDE in higher stochastic dimension (d = 8: seven KL modes + diffusivity, p = 3, P = 165).

Stages (run as separate processes to use all cores):
  python e6_ignition_highd.py mc            Monte Carlo reference (process pool)
  python e6_ignition_highd.py train         pilot (Taylor-4) + trajectory-anchored PINN closure
  python e6_ignition_highd.py run <method>  intrusive propagation with one closure
  python e6_ignition_highd.py collect       merge everything into e6_ignition_highd.pkl
"""
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from closures import Exp, ExactGalerkin, PSP, Taylor, Lanczos, PINNClosure
from problems import Ignition
from propagate import monte_carlo, intrusive, compare

D, P_ORDER, T_END, N_MC, DELTA = 8, 3, 1.0, 20_000, 1.5
T_EVAL = np.linspace(0, T_END, 21)
TAG = f"d{D}p{P_ORDER}"
METHODS = ["eig", "psp", "psp_s3", "psp_s4", "taylor4", "lanczos", "pinn"]


def setup():
    B = PCBasis(D, P_ORDER)
    return B, Ignition(B, delta=DELTA)


def xi_samples():
    return np.random.default_rng(3000 + D).normal(size=(N_MC, D))


def _mc_chunk(args):
    i0, i1 = args
    os.environ["OMP_NUM_THREADS"] = "1"
    _, prob = setup()
    return monte_carlo(prob, xi_samples()[i0:i1], T_EVAL, rtol=1e-8, atol=1e-10)


def stage_mc(workers=4):
    t0 = time.perf_counter()
    edges = np.linspace(0, N_MC, workers + 1).astype(int)
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_mc_chunk, list(zip(edges[:-1], edges[1:]))))
    Y = np.concatenate(parts, axis=2)
    t_mc = time.perf_counter() - t0
    np.savez(os.path.join(RES, f"e6_mc_{TAG}.npz"), Y=Y.astype(np.float32), t_mc=t_mc, workers=workers)
    print(f"MC done: {t_mc:.0f}s wall on {workers} workers; u_mid(T) mean {Y[-1, 15].mean():.4f} "
          f"sd {Y[-1, 15].std():.4f} max {Y[-1].max():.3f}", flush=True)


def stage_train():
    B, prob = setup()
    nl = Exp(B)
    tp = np.linspace(0, T_END, 41)
    t0 = time.perf_counter()
    pilot = os.path.join(RES, f"e6_pilot_{TAG}.npz")
    if os.path.exists(pilot):
        z = np.load(pilot); Cp, wall = z["C"], float(z["wall"])
    else:
        Cp, wall, _, ok = intrusive(prob, Taylor(nl, 4), tp, rtol=1e-6, atol=1e-8)
        np.savez(pilot, C=Cp, wall=wall)
    print(f"pilot (Taylor-4) {wall:.0f}s", flush=True)
    t0 = time.perf_counter()
    anchors = Cp.reshape(-1, B.P)
    pc = PINNClosure(nl, K=3, width=64, depth=3, r_max=0.1, R_eff=2.0, seed=0)
    pc.train(n_dirs=20000, batch=256, iters=5000, lr=2e-3, lbfgs_iters=1500, lbfgs_batch=2048,
             anchors=anchors, generic_frac=0.2, seed=3, verbose=True)
    pc.save(os.path.join(MODELS, f"exp_{TAG}_ignition.npz"))
    with open(os.path.join(RES, f"e6_train_{TAG}.pkl"), "wb") as fh:
        pickle.dump(dict(t_pilot=wall, t_train=time.perf_counter() - t0,
                         val_err=pc.final_val_err, history=pc.history, n_params=pc.net.n_params), fh)
    print(f"training done, val err {pc.final_val_err:.2e}", flush=True)


def make_closure(method, B):
    nl = Exp(B)
    if method == "eig":
        return ExactGalerkin(nl)
    if method == "psp":
        return PSP(nl, P_ORDER + 1)
    if method.startswith("psp_s"):                      # Smolyak sparse rule of level k
        return PSP(nl, method[4:])
    if method == "taylor4":
        return Taylor(nl, 4)
    if method == "lanczos":
        return Lanczos(nl, 10)
    if method == "pinn":
        return PINNClosure.load(nl, os.path.join(MODELS, f"exp_{TAG}_ignition.npz"))
    raise ValueError(method)


def stage_run(method):
    B, prob = setup()
    cl = make_closure(method, B)
    W = int(os.environ.get("PCE_WORKERS", "1"))
    if W > 1:                                           # multi-core run: nodes distributed over W processes
        from parallel import NodeParallel
        par = NodeParallel(cl, prob.nx, W)
        C, wall, nfev, ok = intrusive(prob, par, T_EVAL, rtol=1e-7, atol=1e-9)
        par.close()
        np.savez(os.path.join(RES, f"e6_par_{TAG}_{method}_{W}.npz"), C=C, wall=wall, nfev=nfev, ok=ok, workers=W)
        print(f"{method} [{W} workers]: wall {wall:.1f}s nfev {nfev} ok={ok}", flush=True)
        return
    C, wall, nfev, ok = intrusive(prob, cl, T_EVAL, rtol=1e-7, atol=1e-9)
    np.savez(os.path.join(RES, f"e6_run_{TAG}_{method}.npz"), C=C, wall=wall, nfev=nfev, ok=ok)
    print(f"{method}: wall {wall:.1f}s nfev {nfev} ok={ok}", flush=True)


def stage_collect():
    B, prob = setup()
    z = np.load(os.path.join(RES, f"e6_mc_{TAG}.npz"))
    Ymc = z["Y"].astype(float)
    Psi = B.eval_basis(xi_samples())
    out = dict(t=T_EVAL, x=prob.x, t_mc=float(z["t_mc"]), mc_workers=int(z["workers"]), P=B.P, nnz=B.nnz,
               mc_mean=Ymc.mean(2), mc_std=Ymc.std(2), methods={})
    for m in METHODS:
        f = os.path.join(RES, f"e6_run_{TAG}_{m}.npz")
        if not os.path.exists(f):
            continue
        r = np.load(f)
        d = dict(ok=bool(r["ok"]), wall=float(r["wall"]), nfev=int(r["nfev"]))
        for Wp in (2, 4, 8):
            fp = os.path.join(RES, f"e6_par_{TAG}_{m}_{Wp}.npz")
            if os.path.exists(fp):
                d[f"wall_par{Wp}"] = float(np.load(fp)["wall"])
        if d["ok"]:
            d["C"] = r["C"]
            d.update(compare(B, r["C"], Ymc, Psi))
            print(f"{m:8s} wall {d['wall']:8.1f}s  max_x L2(T) {d['l2_err'][-1].max():.3e}", flush=True)
        out["methods"][m] = d
    if out["methods"].get("eig", {}).get("ok") and out["methods"].get("pinn", {}).get("ok"):
        U = out["methods"]["eig"]["C"].reshape(-1, B.P)
        ex, pc = make_closure("eig", B), make_closure("pinn", B)
        e = np.linalg.norm(pc(U) - ex(U), axis=1) / np.linalg.norm(ex(U), axis=1)
        X = U.copy(); X[:, 0] = 0
        lam = np.linalg.eigvalsh(B.mult_matrix(X[::5]))
        for m in ["psp", "psp_s3", "psp_s4"]:
            if m in out["methods"]:
                es = np.linalg.norm(make_closure(m, B)(U) - ex(U), axis=1) / np.linalg.norm(ex(U), axis=1)
                out["methods"][m]["closure_err"] = es
                print(f"{m} closure err median {np.median(es):.2e} max {es.max():.2e}", flush=True)
        out.update(closure_err=e, input_rho_max=float(np.abs(lam).max()),
                   input_std_max=float(np.linalg.norm(X, axis=1).max()))
        print(f"inputs: std max {out['input_std_max']:.3f}, spectral radius max {out['input_rho_max']:.2f}; "
              f"PINN closure err median {np.median(e):.2e} max {e.max():.2e}", flush=True)
    tr = os.path.join(RES, f"e6_train_{TAG}.pkl")
    if os.path.exists(tr):
        out["train"] = pickle.load(open(tr, "rb"))
    with open(os.path.join(RES, f"e6_ignition_highd.pkl"), "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "mc":
        stage_mc(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    elif stage == "train":
        stage_train()
    elif stage == "run":
        stage_run(sys.argv[2])
    elif stage == "collect":
        stage_collect()
