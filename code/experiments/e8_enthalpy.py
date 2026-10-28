"""E8: PDE with an implicit constitutive law at every node (nonlinear heat conduction in enthalpy form).

    h_t = kappa T_xx + Q sin(pi x),   h = T + k3(x, xi) T^3     (d = 8: kappa, Q, six KL modes of log k3)

Stages:  mc [workers] | pilot | train [seed] | run <method> | collect
Methods: newton, nk, psp, psp_s3, psp_s4 (Smolyak sparse PSP), pinn, pinn_nk1, pinn_newton1, pinn_nk2,
         pinn_nkconv, pinn_newtonconv, linear; append _seed<k> to a pinn* method for the network of seed k
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
from problems import EnthalpyHeat
from implicit_field import CubicField, FieldNewton, FieldNewtonKrylov, FieldPSP, FieldPINN, FieldSolver
from propagate import monte_carlo, intrusive, compare

D, P_ORDER, T_END, N_MC, NX = 8, 3, 1.0, 20_000, 15
T_EVAL = np.linspace(0, T_END, 21)
TAG = f"d{D}p{P_ORDER}"
METHODS = ["newton", "nk", "psp", "psp_s3", "psp_s4", "psp_s5", "pinn", "pinn_nk1", "pinn_newton1", "pinn_nk2", "pinn_nkconv",
           "pinn_newtonconv", "linear", "newton_c8", "nk_c8", "newton_w8", "nk_w8", "chord_w8", "chord_w10",
           "chord_w6", "nk_w10", "pinn_nkc8", "pinn_nk2rec"]
SEED_METHODS = [f"{m}_seed{k}" for k in (1, 2) for m in ("pinn", "pinn_nk2")]
RTOL, ATOL = 1e-7, 1e-9


def setup():
    B = PCBasis(D, P_ORDER)
    return B, EnthalpyHeat(B, nx=NX), CubicField(B)


def xi_samples():
    return np.random.default_rng(5000 + D).normal(size=(N_MC, D))


def _mc_chunk(args):
    i0, i1 = args
    _, prob, _ = setup()
    return monte_carlo(prob, xi_samples()[i0:i1], T_EVAL, rtol=1e-8, atol=1e-10)


def stage_mc(workers=4):
    t0 = time.perf_counter()
    edges = np.linspace(0, N_MC, workers + 1).astype(int)
    with ProcessPoolExecutor(workers) as ex:
        Y = np.concatenate(list(ex.map(_mc_chunk, list(zip(edges[:-1], edges[1:])))), axis=2)
    t_mc = time.perf_counter() - t0
    np.savez(os.path.join(RES, f"e8_mc_{TAG}.npz"), Y=Y.astype(np.float32), t_mc=t_mc, workers=workers)
    print(f"MC done {t_mc:.0f}s on {workers} workers; h_mid(T) mean {Y[-1, NX // 2].mean():.3f} "
          f"sd {Y[-1, NX // 2].std():.3f}", flush=True)


def stage_pilot():
    B, prob, fs = setup()
    tp = np.linspace(0, T_END, 101)
    C, wall, _, ok = intrusive(prob, FieldPSP(fs, P_ORDER + 1, prob.K3), tp, rtol=1e-5, atol=1e-7)
    np.savez(os.path.join(RES, f"e8_pilot_{TAG}.npz"), C=C, wall=wall)
    print(f"pilot (PSP, loose tolerance) {wall:.0f}s ok={ok}", flush=True)


def anchors():
    B, prob, fs = setup()
    C = np.load(os.path.join(RES, f"e8_pilot_{TAG}.npz"))["C"]              # (nt, nx, P)
    H = C.reshape(-1, B.P)
    K = np.tile(prob.K3, (C.shape[0], 1))
    keep = np.linalg.norm(H, axis=1) > 1e-8
    return H[keep], K[keep]


def _suffix(seed):
    return "" if int(seed) == 0 else f"_seed{int(seed)}"


def stage_train(seed=0):
    B, prob, fs = setup()
    H, K = anchors()
    rng = np.random.default_rng(77)
    Hv, Kv = FieldPINN.sample(H, K, 300, rng)
    Yv = FieldNewton(fs)(Hv, Kv)
    val = lambda m: float(np.median(np.linalg.norm(m(Hv, Kv) - Yv, axis=1) / np.linalg.norm(Yv, axis=1)))
    pn = FieldPINN(fs, width=128, depth=3, seed=seed)
    t0 = time.perf_counter()
    pn.train(H, K, n_pool=20000, batch=256, iters=4000, lr=2e-3, lbfgs_iters=300, lbfgs_batch=1024,
             seed=1 + 100 * seed, verbose=True, validate=val)
    t_train = time.perf_counter() - t0
    pn.save(os.path.join(MODELS, f"enthalpy_{TAG}{_suffix(seed)}.npz"))
    lin = float(np.median(np.linalg.norm(Hv - Yv, axis=1) / np.linalg.norm(Yv, axis=1)))
    with open(os.path.join(RES, f"e8_train_{TAG}{_suffix(seed)}.pkl"), "wb") as fh:
        pickle.dump(dict(t_train=t_train, val=pn.final_val, val_linear=lin, history=pn.history,
                         n_params=pn.net.n_params,
                         t_pilot=float(np.load(os.path.join(RES, f"e8_pilot_{TAG}.npz"))["wall"])), fh)
    print(f"train {t_train:.0f}s  val median rel err {pn.final_val:.2e} (linearised law {lin:.2e})", flush=True)


class Linear:
    label = "linearised law"
    iters = []

    def __call__(self, h, k3):
        return h.copy()


# Revision baselines (FieldSolver): <method>_<init><tol exponent>, e.g. nk_w8 = matrix-free Newton-Krylov,
# warm-started from the previous call, converged to ||R|| <= 1e-8 ||h||; c = cold (linearised-law) start.
STRONG = {"newton_c8": ("newton", "linear", 1e-8), "nk_c8": ("nk", "linear", 1e-8),
          "newton_w8": ("newton", "warm", 1e-8), "nk_w8": ("nk", "warm", 1e-8),
          "chord_w8": ("chord", "warm", 1e-8), "chord_w10": ("chord", "warm", 1e-10),
          "chord_w6": ("chord", "warm", 1e-6), "nk_w10": ("nk", "warm", 1e-10)}


def make_closure(method, prob, fs):
    if method in STRONG:
        meth, init, tol = STRONG[method]
        return FieldSolver(fs, method=meth, init=init, tol=tol, m=10, record=True)
    if method == "newton":
        return FieldNewton(fs)
    if method == "nk":
        return FieldNewtonKrylov(fs, m=10)
    if method == "psp":
        return FieldPSP(fs, P_ORDER + 1, prob.K3)
    if method.startswith("psp_s"):                      # Smolyak sparse rule of level k
        return FieldPSP(fs, method[4:], prob.K3)
    if method == "linear":
        return Linear()
    seed = 0
    if "@" in method:                                   # networks of e10/e11, e.g. pinn_nk2@w512d4b4, pinn@F0.5A
        method, ctag = method.split("@")
        pn = FieldPINN.from_file(fs, os.path.join(MODELS, f"enthalpy_{TAG}_{ctag}.npz"))
    else:
        if "_seed" in method:                           # networks trained with other seeds
            method, seed = method.split("_seed")
        pn = FieldPINN(fs, width=128, depth=3).load_weights(os.path.join(MODELS, f"enthalpy_{TAG}{_suffix(seed)}.npz"))
    if method == "pinn":
        return pn
    if method == "pinn_nk1":
        return FieldNewtonKrylov(fs, m=5, init=pn, fixed_steps=1)
    if method == "pinn_newton1":
        return FieldNewton(fs, init=pn, fixed_steps=1)
    if method == "pinn_nk2":
        return FieldNewtonKrylov(fs, m=5, init=pn, fixed_steps=2)
    if method == "pinn_nkconv":
        return FieldNewtonKrylov(fs, m=10, init=pn)
    if method == "pinn_newtonconv":
        return FieldNewton(fs, init=pn)
    if method == "pinn_nkc8":                           # network guess, converged to the baselines' tolerance
        return FieldSolver(fs, method="nk", init=pn, tol=1e-8, m=10, record=True)
    if method == "pinn_nk2rec":                         # = pinn_nk2, recording the achieved residual (not timed)
        return FieldSolver(fs, method="nk", init=pn, fixed_steps=2, m=5, record=True)
    raise ValueError(method)


def stage_run(method):
    B, prob, fs = setup()
    cl = make_closure(method, prob, fs)
    W = int(os.environ.get("PCE_WORKERS", "1"))
    if W > 1:                                           # multi-core run: nodes distributed over W processes
        from parallel import NodeParallel
        par = NodeParallel(cl, prob.nx, W, static=prob.K3)
        C, wall, nfev, ok = intrusive(prob, par, T_EVAL, rtol=RTOL, atol=ATOL)
        par.close()
        np.savez(os.path.join(RES, f"e8_par_{TAG}_{method}_{W}.npz"), C=C, wall=wall, nfev=nfev, ok=ok, workers=W)
        print(f"{method} [{W} workers]: wall {wall:.1f}s nfev {nfev} ok={ok}", flush=True)
        return
    C, wall, nfev, ok = intrusive(prob, cl, T_EVAL, rtol=RTOL, atol=ATOL)
    its = float(np.mean(cl.iters)) if getattr(cl, "iters", None) else -1.0
    extra = {}
    if isinstance(cl, FieldSolver):
        extra = dict(counters=np.array([cl.n[k] for k in ("calls", "jv", "jac", "res")]),
                     final_res=np.array(cl.final_res), init_res=np.array(cl.init_res),
                     iters_all=np.array(cl.iters))
    np.savez(os.path.join(RES, f"e8_run_{TAG}_{method}.npz"), C=C, wall=wall, nfev=nfev, ok=ok, iters=its, **extra)
    print(f"{method}: wall {wall:.1f}s nfev {nfev} ok={ok} iters {its:.2f}", flush=True)


def stage_collect():
    B, prob, fs = setup()
    z = np.load(os.path.join(RES, f"e8_mc_{TAG}.npz"))
    Ymc = z["Y"].astype(float)
    xi = xi_samples()
    Psi = B.eval_basis(xi)
    # temperature (the physically relevant output) from the MC enthalpy
    from implicit import cubic_root
    k3 = prob.sample_inputs(xi)[1]["k3"]
    Tmc = cubic_root(k3[None], 1.0, Ymc)
    out = dict(t=T_EVAL, x=prob.x, t_mc=float(z["t_mc"]), mc_workers=int(z["workers"]), P=B.P, nnz=B.nnz,
               mc_mean=Ymc.mean(2), mc_std=Ymc.std(2), mcT_mean=Tmc.mean(2), mcT_std=Tmc.std(2), methods={})
    newton = FieldNewton(fs)
    import glob
    CAP = sorted(os.path.basename(f)[len(f"e8_run_{TAG}_"):-4] for f in glob.glob(os.path.join(RES, f"e8_run_{TAG}_*@*.npz")))
    for m in METHODS + SEED_METHODS + CAP:
        f = os.path.join(RES, f"e8_run_{TAG}_{m}.npz")
        if not os.path.exists(f):
            continue
        r = np.load(f)
        d = dict(ok=bool(r["ok"]), wall=float(r["wall"]), nfev=int(r["nfev"]), iters=float(r["iters"]))
        for key in ("counters", "final_res", "init_res", "iters_all"):
            if key in r:
                d[key] = r[key]
        for Wp in (2, 4, 8):
            fp = os.path.join(RES, f"e8_par_{TAG}_{m}_{Wp}.npz")
            if os.path.exists(fp):
                rp = np.load(fp)
                d[f"wall_par{Wp}"] = float(rp["wall"])
                if bool(r["ok"]) and bool(rp["ok"]) and rp["C"].shape == r["C"].shape:
                    d[f"dev_par{Wp}"] = float(np.abs(rp["C"] - r["C"]).max() / np.abs(r["C"]).max())
        if d["ok"]:
            C = r["C"]
            d["C"] = C
            d.update(compare(B, C, Ymc, Psi))
            # temperature coefficients via the converged Galerkin solve, at T_END
            Tc = newton(C[-1], prob.K3)
            d["T_end"] = Tc
            d["T_mean"], d["T_std"] = Tc[:, 0], np.sqrt(np.sum(Tc[:, 1:] ** 2, 1))
            print(f"{m:13s} wall {d['wall']:8.1f}s  its {d['iters']:.2f}  max_x L2(h) t=0.5,1: "
                  f"{d['l2_err'][[10, 20]].max(1).round(4)}", flush=True)
        out["methods"][m] = d
    if out["methods"].get("newton", {}).get("ok"):
        H = out["methods"]["newton"]["C"].reshape(-1, B.P)
        K = np.tile(prob.K3, (len(T_EVAL), 1))
        keep = np.linalg.norm(H, axis=1) > 1e-8
        H, K = H[keep], K[keep]
        ref = newton(H, K)
        for m in ["pinn", "pinn_nk1", "pinn_newton1", "pinn_nk2", "pinn_nkconv", "psp", "psp_s3", "psp_s4",
                  "linear"] + SEED_METHODS + CAP:
            if m in out["methods"]:
                cl = make_closure(m, prob, fs)
                Y = cl(H, K) if not m.startswith("psp") else np.concatenate([cl(H[i:i + NX]) for i in range(0, len(H), NX)])
                e = np.linalg.norm(Y - ref, axis=1) / np.linalg.norm(ref, axis=1)
                out["methods"][m]["closure_err"] = e
                print(f"closure err along Newton trajectory, {m}: median {np.median(e):.2e} max {e.max():.2e}")
    tr = os.path.join(RES, f"e8_train_{TAG}.pkl")
    if os.path.exists(tr):
        out["train"] = pickle.load(open(tr, "rb"))
    for k in (1, 2):
        tr = os.path.join(RES, f"e8_train_{TAG}_seed{k}.pkl")
        if os.path.exists(tr):
            out[f"train_seed{k}"] = pickle.load(open(tr, "rb"))
    with open(os.path.join(RES, "e8_enthalpy.pkl"), "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    st = sys.argv[1]
    {"mc": lambda: stage_mc(int(sys.argv[2]) if len(sys.argv) > 2 else 4), "pilot": stage_pilot,
     "train": lambda: stage_train(int(sys.argv[2]) if len(sys.argv) > 2 else 0),
     "run": lambda: stage_run(sys.argv[2]), "collect": stage_collect}[st]()
