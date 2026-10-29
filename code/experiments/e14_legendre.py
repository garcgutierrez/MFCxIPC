"""E14: Legendre chaos (uniform inputs) and multi-element PC.

  python e14_legendre.py checks      (P3) domain and (P4) consistency for a Legendre basis, against Hermite
  python e14_legendre.py train       generic neural closure of exp for the Legendre basis (d = 2, p = 4, settings of E1)
  python e14_legendre.py ignition    ignition problem with uniform inputs: global basis and 2 x 2, 4 x 4 elements
"""
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from closures import Exp, ExactGalerkin, PSP, Taylor, Lanczos, PINNClosure, sample_directions
from problems import Ignition
from propagate import monte_carlo, intrusive, bhattacharyya

D, P_ORDER, T_END, N_MC, DELTA = 2, 4, 1.0, 20_000, 1.5
MODEL = os.path.join(MODELS, "exp_d2p4_legendre.npz")
SQ3 = np.sqrt(3.0)


class IgnitionUniform(Ignition):
    """Ignition with uniform inputs of unit variance, xi_k = sqrt(3) eta_k with eta_k ~ U(-1, 1), in a Legendre basis.
    On the element [lo, hi] of eta-space the basis variable is eta' ~ U(-1, 1), eta = c + h eta'; the random fields
    are projected on the (local) basis with a Gauss-Legendre rule."""

    def __init__(self, basis, lo=None, hi=None, **kw):
        super().__init__(basis, **kw)
        d = basis.d
        self.lo = -np.ones(d) if lo is None else np.asarray(lo, float)
        self.hi = np.ones(d) if hi is None else np.asarray(hi, float)
        c, h = (self.hi + self.lo) / 2, (self.hi - self.lo) / 2
        xi = lambda E: SQ3 * (c + h * E)
        q = 2 * basis.p + 8
        self.delta = basis.project(lambda E: np.exp(self.mu_log[:, None] + self.A @ xi(E)[:, :self.n_kl].T), q)
        self.kappa = basis.project(
            lambda E: self.mean_kappa * np.exp(self.sk * xi(E)[:, self.kappa_dim] - 0.5 * self.sk ** 2), q)


def stage_checks():
    out = {}
    # (P3) smallest eigenvalue of M_u for u = 1 + 0.5 psi_1: bounded (Legendre, min 1 - 0.5 sqrt3) or Gaussian input
    ps = [2, 4, 6, 8, 10, 12]
    for fam in ["legendre", "hermite"]:
        lam = []
        for p in ps:
            B = PCBasis(1, p, fam)
            u = np.zeros(B.P); u[0], u[1] = 1.0, 0.5
            lam.append(float(np.linalg.eigvalsh(B.mult_matrix(u)).min()))
        out[f"lmin_{fam}"] = lam
    out["lmin_p"], out["umin_legendre"] = ps, 1 - 0.5 * SQ3
    # (P4) G_exp against the exact projection on the modes of degree <= 2, fixed degree-two input, d = 2
    ps = [2, 4, 6, 8, 10, 12]
    coef = {(1, 0): 0.4, (0, 1): 0.3, (2, 0): 0.15, (1, 1): 0.1}
    for fam in ["legendre", "hermite"]:
        e_mf, e_psp = [], []
        for p in ps:
            B = PCBasis(2, p, fam)
            u = np.zeros(B.P)
            for a, v in coef.items():
                u[[i for i, al in enumerate(B.alphas) if tuple(al) == a][0]] = v
            X, W = B.quadrature(40)
            Psi = B.eval_basis(X)
            H = (np.exp(Psi @ u)) @ (Psi * W[:, None])
            low = B.degree <= 2
            nrm = np.linalg.norm(H[low])
            e_mf.append(float(np.linalg.norm(ExactGalerkin(Exp(B))(u)[low] - H[low]) / nrm))
            e_psp.append(float(np.linalg.norm(PSP(Exp(B), p + 1)(u)[low] - H[low]) / nrm))
        out[f"p4_{fam}"] = dict(p=ps, mf=e_mf, psp=e_psp)
    with open(os.path.join(RES, "e14_checks.pkl"), "wb") as fh:
        pickle.dump(out, fh)
    for k, v in out.items():
        print(k, v, flush=True)


def stage_train():
    from e1_closures import assess
    B = PCBasis(D, P_ORDER, "legendre")
    nl = Exp(B)
    pc = PINNClosure(nl, K=3, width=64, depth=3, r_max=2.0, R_eff=4.0, seed=0)
    t0 = time.perf_counter()
    pc.train(n_dirs=30000, batch=256, iters=8000, lr=3e-3, lbfgs_iters=2500, lbfgs_batch=4096, seed=1, verbose=True)
    t_train = time.perf_counter() - t0
    pc.save(MODEL)
    res = assess(pc, nl, B)                       # test inputs of Table 1 (same construction and seed)
    rng = np.random.default_rng(2024)
    Dt = sample_directions(B, 4000, rng)
    X = (pc.r_limit(Dt) * rng.random(4000))[:, None] * Dt
    Ge = ExactGalerkin(nl)(X)
    for m in [6, 8, 10]:
        res[f"lanczos{m}"] = (np.linalg.norm(Lanczos(nl, m)(X) - Ge, axis=1) / np.linalg.norm(Ge, axis=1)).tolist()
    res.update(t_train=t_train, n_params=pc.net.n_params, val=float(pc.final_val_err))
    json.dump(res, open(os.path.join(RES, "e14_train.json"), "w"))
    for k in ["pinn", "taylor4", "taylor8", "lanczos6", "lanczos8", "lanczos10", "psp_vs_galerkin"]:
        print(f"{k:16s} median {np.median(res[k]):.2e}  p95 {np.percentile(res[k], 95):.2e}", flush=True)
    print(f"train {t_train:.0f}s", flush=True)


def metrics(Y_pc, Y_mc):
    sd = np.maximum(Y_mc.std(axis=2), 1e-300)
    bc = np.array([[bhattacharyya(Y_pc[i, j], Y_mc[i, j]) for j in range(Y_mc.shape[1])] for i in range(Y_mc.shape[0])])
    return dict(l2_err=np.sqrt(np.mean((Y_pc - Y_mc) ** 2, axis=2)) / sd,
                mean_err_p=np.abs(Y_pc.mean(axis=2) - Y_mc.mean(axis=2)) / sd,
                std_err_p=np.abs(Y_pc.std(axis=2) - Y_mc.std(axis=2)) / sd, bc=bc)


def elements(ne):
    edges = np.linspace(-1, 1, ne + 1)
    return [(np.array([edges[i], edges[j]]), np.array([edges[i + 1], edges[j + 1]]))
            for i in range(ne) for j in range(ne)]


def stage_ignition():
    B = PCBasis(D, P_ORDER, "legendre")
    nl = Exp(B)
    t_eval = np.linspace(0, T_END, 21)
    eta = np.random.default_rng(2030).uniform(-1, 1, size=(N_MC, D))
    glob = IgnitionUniform(B, delta=DELTA)
    t0 = time.perf_counter()
    Ymc = monte_carlo(glob, SQ3 * eta, t_eval, rtol=1e-8, atol=1e-10)
    t_mc = time.perf_counter() - t0
    print(f"MC {t_mc:.0f}s; u_mid(T) mean {Ymc[-1, glob.mid].mean():.4f} sd {Ymc[-1, glob.mid].std():.4f}", flush=True)
    meths = [("eig", ExactGalerkin(nl)), ("psp", PSP(nl, P_ORDER + 1)), ("taylor4", Taylor(nl, 4)),
             ("lanczos", Lanczos(nl, 10)), ("pinn", PINNClosure.load(nl, MODEL))]
    exact = ExactGalerkin(nl)
    out = dict(t=t_eval, x=glob.x, t_mc=t_mc, mc_mean=Ymc.mean(2), mc_std=Ymc.std(2), runs={})
    for ne in [1, 2, 4]:
        els = elements(ne)
        inputs = []
        for name, cl in meths:
            Y_pc = np.zeros_like(Ymc)
            wall, nfev, ok = 0.0, 0, True
            for lo, hi in els:
                prob = IgnitionUniform(B, lo, hi, delta=DELTA)
                C, w, nf, okk = intrusive(prob, cl, t_eval, rtol=1e-7, atol=1e-9)
                wall, nfev, ok = wall + w, nfev + nf, ok and okk
                if not okk:
                    break
                inside = np.all((eta >= lo) & (eta <= hi), axis=1)
                c, h = (hi + lo) / 2, (hi - lo) / 2
                Y_pc[:, :, inside] = C @ B.eval_basis((eta[inside] - c) / h).T
                if name == "eig":
                    inputs.append(C.reshape(-1, B.P))
            if not ok:
                out["runs"][(ne, name)] = dict(ok=False)
                print(f"ne={ne} {name}: FAILED", flush=True)
                continue
            m = metrics(Y_pc, Ymc)
            out["runs"][(ne, name)] = dict(ok=True, wall=wall, nfev=nfev, **m)
            print(f"ne={ne} {name:8s} wall {wall:6.2f}s nfev {nfev:5d}  max_x L2(T) {m['l2_err'][-1].max():.2e}  "
                  f"mean {m['mean_err_p'][-1].max():.1e} sd {m['std_err_p'][-1].max():.1e}", flush=True)
        U = np.concatenate(inputs)
        X = U.copy(); X[:, 0] = 0.0
        rho = np.abs(np.linalg.eigvalsh(B.mult_matrix(X[::5]))).max(axis=1)
        Ge = exact(U)
        err = {k: np.linalg.norm(cl(U) - Ge, axis=1) / np.linalg.norm(Ge, axis=1)
               for k, cl in meths if k in ("taylor4", "lanczos", "pinn")}
        out["runs"][(ne, "inputs")] = dict(rho_max=float(rho.max()), rho_med=float(np.median(rho)),
                                           **{f"cerr_{k}": v for k, v in err.items()})
        print(f"ne={ne}: closure inputs spectral radius median {np.median(rho):.2f} max {rho.max():.2f}; closure error "
              + ", ".join(f"{k} {np.median(v):.1e}/{v.max():.1e}" for k, v in err.items()), flush=True)
    with open(os.path.join(RES, "e14_ignition.pkl"), "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    dict(checks=stage_checks, train=stage_train, ignition=stage_ignition)[sys.argv[1]]()
