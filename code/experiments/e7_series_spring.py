"""E7: implicitly defined, bivariate closure (oscillator with a massless node and a hardening spring).

Stages:
  python e7_series_spring.py mc [workers]
  python e7_series_spring.py train          PSP pilot (inputs only) + label-free PINN closure
  python e7_series_spring.py run <method>   newton | nk | psp | psp_s3 | psp_s4 | pinn | pinn_newton1 | ...
                                            (append _seed<k> to a pinn* method for the network of seed k)
  python e7_series_spring.py collect
"""
import os
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from problems import SeriesSpringOscillator
from implicit import NewtonClosure, CardanoPSP, PINNImplicit, NewtonKrylovClosure, CubicSolver
from propagate import monte_carlo, intrusive, compare

D, P_ORDER, T_END, N_MC = 6, 3, 20.0, 100_000
T_EVAL = np.linspace(0, T_END, 81)
TAG = f"d{D}p{P_ORDER}"
SEED_METHODS = [f"{m}_seed{k}" for k in (1, 2) for m in ("pinn", "pinn_newton1")]
METHODS = ["newton", "nk", "psp", "psp_s3", "psp_s4", "pinn", "pinn_newton1", "pinn_nk1", "pinn_newtonconv", "pinn_nkconv", "linear",
           "newton_c8", "nk_c8", "newton_w8", "nk_w8", "chord_w8", "chord_w10", "pinn_nkc8", "pinn_newton1rec",
           "psp5", "psp_s5"]


def setup():
    B = PCBasis(D, P_ORDER)
    return B, SeriesSpringOscillator(B)


def xi_samples():
    return np.random.default_rng(4000 + D).normal(size=(N_MC, D))


def _mc_chunk(args):
    i0, i1 = args
    _, prob = setup()
    return monte_carlo(prob, xi_samples()[i0:i1], T_EVAL)


def stage_mc(workers=2):
    t0 = time.perf_counter()
    edges = np.linspace(0, N_MC, workers + 1).astype(int)
    with ProcessPoolExecutor(workers) as ex:
        Y = np.concatenate(list(ex.map(_mc_chunk, list(zip(edges[:-1], edges[1:])))), axis=2)
    t_mc = time.perf_counter() - t0
    np.savez(os.path.join(RES, f"e7_mc_{TAG}.npz"), Y=Y.astype(np.float32), t_mc=t_mc, workers=workers)
    print(f"MC done {t_mc:.0f}s on {workers} workers; x(T) mean {Y[-1, 0].mean():.3f} sd {Y[-1, 0].std():.3f}",
          flush=True)


def validation_set(sysm, anchors, n=500, seed=77):
    rng = np.random.default_rng(seed)
    pn = PINNImplicit(sysm)
    X = pn.sample(anchors, n, rng)
    return X, NewtonClosure(sysm)(X)


def _suffix(seed):
    return "" if int(seed) == 0 else f"_seed{int(seed)}"


def stage_train(seed=0):
    B, prob = setup()
    s = prob.sys
    tp = np.linspace(0, T_END, 401)
    Cp, wall, _, ok = intrusive(prob, CardanoPSP(s, P_ORDER + 1), tp)
    print(f"pilot (PSP) {wall:.0f}s ok={ok}", flush=True)
    anchors = Cp[:, 0, :]
    Xv, Yv = validation_set(s, anchors)
    val = lambda m: float(np.median(np.linalg.norm(m(Xv) - Yv, axis=1) / np.linalg.norm(Yv, axis=1)))
    pn = PINNImplicit(s, width=64, depth=3, seed=seed)
    t0 = time.perf_counter()
    pn.train(anchors, n_pool=20000, batch=256, iters=6000, lr=2e-3, lbfgs_iters=2000, lbfgs_batch=2048,
             seed=1 + 100 * seed, verbose=True, validate=val)
    t_train = time.perf_counter() - t0
    pn.save(os.path.join(MODELS, f"implicit_{TAG}{_suffix(seed)}.npz"))
    lin = float(np.median(np.linalg.norm(s.linear(Xv) - Yv, axis=1) / np.linalg.norm(Yv, axis=1)))
    with open(os.path.join(RES, f"e7_train_{TAG}{_suffix(seed)}.pkl"), "wb") as fh:
        pickle.dump(dict(t_pilot=wall, t_train=t_train, val=pn.final_val, val_linear=lin,
                         history=pn.history, n_params=pn.net.n_params), fh)
    print(f"train {t_train:.0f}s  val median rel err {pn.final_val:.2e} (linearised closure {lin:.2e})", flush=True)


class Linear:
    label = "linearised"

    def __init__(self, s):
        self.s = s

    def __call__(self, x):
        return self.s.linear(np.atleast_2d(x)).reshape(np.shape(x))


STRONG = {"newton_c8": ("newton", "linear", 1e-8), "nk_c8": ("nk", "linear", 1e-8),
          "newton_w8": ("newton", "warm", 1e-8), "nk_w8": ("nk", "warm", 1e-8),
          "chord_w8": ("chord", "warm", 1e-8), "chord_w10": ("chord", "warm", 1e-10)}


def make_closure(method, prob):
    s = prob.sys
    if method in STRONG:
        meth, init, tol = STRONG[method]
        return CubicSolver(s, method=meth, init=init, tol=tol, m=10, record=True)
    seed = 0
    if "@" in method:                                   # networks of e10/e11, e.g. pinn@w256d4b4, pinn_newton1@F0.5A
        method, ctag = method.split("@")
        net = PINNImplicit.from_file(s, os.path.join(MODELS, f"implicit_{TAG}_{ctag}.npz"))
        if method == "pinn":
            return net
        if method == "pinn_newton1":
            return NewtonClosure(s, init=net, fixed_steps=1)
        raise ValueError(method)
    if "_seed" in method:                               # networks trained with other seeds
        method, seed = method.split("_seed")
    model = os.path.join(MODELS, f"implicit_{TAG}{_suffix(seed)}.npz")
    if method == "newton":
        return NewtonClosure(s, tol=1e-12)
    if method == "psp":
        return CardanoPSP(s, P_ORDER + 1)
    if method == "psp5":                                # tensor rule refined (quadrature-error check)
        return CardanoPSP(s, P_ORDER + 2)
    if method.startswith("psp_s"):                      # Smolyak sparse rule of level k
        return CardanoPSP(s, method[4:])
    if method == "pinn":
        return PINNImplicit(s).load_weights(model)
    if method == "pinn_newton1":
        pn = PINNImplicit(s).load_weights(model)
        return NewtonClosure(s, init=pn, fixed_steps=1)
    if method == "nk":
        return NewtonKrylovClosure(s, m=10, tol=1e-10)
    if method == "pinn_nk1":
        pn = PINNImplicit(s).load_weights(model)
        return NewtonKrylovClosure(s, m=5, init=pn, fixed_steps=1)
    if method == "pinn_newtonconv":
        pn = PINNImplicit(s).load_weights(model)
        return NewtonClosure(s, init=pn, tol=1e-12)
    if method == "pinn_nkconv":
        pn = PINNImplicit(s).load_weights(model)
        return NewtonKrylovClosure(s, m=10, init=pn, tol=1e-10)
    if method == "pinn_nkc8":
        return CubicSolver(s, method="nk", init=PINNImplicit(s).load_weights(model), tol=1e-8, m=10, record=True)
    if method == "pinn_newton1rec":
        return CubicSolver(s, method="newton", init=PINNImplicit(s).load_weights(model), fixed_steps=1, record=True)
    if method == "linear":
        return Linear(s)
    raise ValueError(method)


def stage_run(method):
    B, prob = setup()
    cl = make_closure(method, prob)
    C, wall, nfev, ok = intrusive(prob, cl, T_EVAL)
    extra = dict(newton_iters=np.mean(cl.iters)) if getattr(cl, "iters", None) else {}
    if isinstance(cl, CubicSolver):
        extra.update(counters=np.array([cl.n[k] for k in ("calls", "jv", "jac", "res")]),
                     final_res=np.array(cl.final_res), init_res=np.array(cl.init_res), iters_all=np.array(cl.iters))
    nt = os.environ.get("OPENBLAS_NUM_THREADS", "1")
    if nt != "1":                                       # multi-threaded BLAS run: keep the single-core result
        np.savez(os.path.join(RES, f"e7_par_{TAG}_{method}_{nt}.npz"), C=C, wall=wall, nfev=nfev, ok=ok)
        print(f"{method} [{nt} BLAS threads]: wall {wall:.1f}s nfev {nfev} ok={ok}", flush=True)
        return
    np.savez(os.path.join(RES, f"e7_run_{TAG}_{method}.npz"), C=C, wall=wall, nfev=nfev, ok=ok, **extra)
    print(f"{method}: wall {wall:.1f}s nfev {nfev} ok={ok} {extra}", flush=True)


def stage_collect():
    B, prob = setup()
    z = np.load(os.path.join(RES, f"e7_mc_{TAG}.npz"))
    Ymc = z["Y"].astype(float)
    xi = xi_samples()
    Psi = B.eval_basis(xi)
    out = dict(t=T_EVAL, t_mc=float(z["t_mc"]), mc_workers=int(z["workers"]), P=B.P, nnz=B.nnz, methods={},
               pdf=dict(idx=[20, 40, 80], mc=Ymc[[20, 40, 80]][:, :, :20000]), Psi_sub=Psi[:20000])
    import glob
    CAP = sorted(os.path.basename(f)[len(f"e7_run_{TAG}_"):-4] for f in glob.glob(os.path.join(RES, f"e7_run_{TAG}_*@*.npz")))
    for m in METHODS + SEED_METHODS + CAP:
        f = os.path.join(RES, f"e7_run_{TAG}_{m}.npz")
        if not os.path.exists(f):
            continue
        r = np.load(f)
        d = dict(ok=bool(r["ok"]), wall=float(r["wall"]), nfev=int(r["nfev"]))
        if "newton_iters" in r:
            d["newton_iters"] = float(r["newton_iters"])
        for Wp in (2, 4, 8):
            fp = os.path.join(RES, f"e7_par_{TAG}_{m}_{Wp}.npz")
            if os.path.exists(fp):
                d[f"wall_par{Wp}"] = float(np.load(fp)["wall"])
        for key in ("counters", "final_res", "init_res", "iters_all"):
            if key in r:
                d[key] = r[key]
        if d["ok"]:
            d["C"] = r["C"]
            d.update(compare(B, r["C"], Ymc, Psi))
            print(f"{m:7s} wall {d['wall']:8.1f}s  L2(x) t=5,10,20: {d['l2_err'][[20, 40, 80], 0].round(4)} "
                  f"BC {d['bc'][[20, 40, 80], 0].round(4)}", flush=True)
        out["methods"][m] = d
    if out["methods"].get("newton", {}).get("ok"):
        X = out["methods"]["newton"]["C"][:, 0, :]
        ref = make_closure("newton", prob)(X)
        for m in ["pinn", "pinn_newton1", "pinn_nk1", "nk", "psp", "psp_s3", "psp_s4", "linear"] + SEED_METHODS + CAP:
            if m in out["methods"]:
                e = np.linalg.norm(make_closure(m, prob)(X) - ref, axis=1) / np.linalg.norm(ref, axis=1)
                out["methods"][m]["closure_err"] = e
                print(f"closure err along Newton trajectory, {m}: median {np.median(e):.2e} max {e.max():.2e}")
    tr = os.path.join(RES, f"e7_train_{TAG}.pkl")
    if os.path.exists(tr):
        out["train"] = pickle.load(open(tr, "rb"))
    for k in (1, 2):
        tr = os.path.join(RES, f"e7_train_{TAG}_seed{k}.pkl")
        if os.path.exists(tr):
            out[f"train_seed{k}"] = pickle.load(open(tr, "rb"))
    with open(os.path.join(RES, "e7_series_spring.pkl"), "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "mc":
        stage_mc(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    elif st == "train":
        stage_train(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    elif st == "run":
        stage_run(sys.argv[2])
    elif st == "collect":
        stage_collect()
