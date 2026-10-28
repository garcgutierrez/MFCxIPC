"""E5: solid-fuel ignition (Frank-Kamenetskii) reaction-diffusion with a random reaction field."""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from closures import Exp, ExactGalerkin, PSP, Taylor, PINNClosure, Lanczos
from problems import Ignition
from propagate import monte_carlo, intrusive, compare

T_END, N_MC, DELTA = 1.0, 20_000, 1.5
CASES = [(2, 4, "exp_d2p4"), (5, 3, "exp_d5p3")]


def main():
    out = {}
    t_eval = np.linspace(0, T_END, 21)
    for d, p, model in CASES:
        B = PCBasis(d, p)
        prob = Ignition(B, delta=DELTA)
        xi = np.random.default_rng(2028 + d).normal(size=(N_MC, d))
        cache = os.path.join(RES, f"e5_mc_d{d}p{p}.npz")
        if os.path.exists(cache):
            z = np.load(cache); Ymc, t_mc = z["Y"], float(z["t_mc"])
        else:
            t0 = time.perf_counter()
            Ymc = monte_carlo(prob, xi, t_eval, rtol=1e-8, atol=1e-10)
            t_mc = time.perf_counter() - t0
            np.savez(cache, Y=Ymc.astype(np.float32), t_mc=t_mc)
        Psi = B.eval_basis(xi)
        res = dict(t=t_eval, x=prob.x, t_mc=t_mc, P=B.P, methods={},
                   mc_mean=Ymc.mean(2), mc_std=Ymc.std(2), mc_mid=Ymc[[5, 20], prob.mid, :])
        print(f"d={d} p={p} P={B.P}: MC {t_mc:.1f}s, u_mid(T) mean {Ymc[-1, prob.mid].mean():.4f} "
              f"sd {Ymc[-1, prob.mid].std():.4f} max {Ymc[-1].max():.3f}", flush=True)
        nl = Exp(B)
        meths = [("eig", ExactGalerkin(nl)), ("psp", PSP(nl, p + 1)), ("psp_sparse", PSP(nl, f"s{p}")),
                 ("taylor4", Taylor(nl, 4)),
                 ("lanczos", Lanczos(nl, 10)), ("pinn_generic", PINNClosure.load(nl, os.path.join(MODELS, f"{model}.npz")))]
        for name, cl in meths:
            C, wall, nfev, ok = intrusive(prob, cl, t_eval, rtol=1e-7, atol=1e-9)
            if not ok:
                res["methods"][name] = dict(ok=False)
                print(f"  {name}: FAILED", flush=True)
                continue
            m = compare(B, C, Ymc, Psi)
            res["methods"][name] = dict(ok=True, C=C, wall=wall, nfev=nfev, **m)
            print(f"  {name:13s} wall {wall:7.2f}s nfev {nfev}  L2(mid) t=.25,.5,1: "
                  f"{m['l2_err'][[5, 10, 20], prob.mid].round(5)}  max_x L2(T): {m['l2_err'][-1].max():.4f}",
                  flush=True)
        if res["methods"]["eig"]["ok"]:
            U = res["methods"]["eig"]["C"].reshape(-1, B.P)
            X = U.copy(); X[:, 0] = 0
            lam = np.linalg.eigvalsh(B.mult_matrix(X[::7]))
            res["input_rho_max"] = float(np.abs(lam).max())
            res["input_std_max"] = float(np.linalg.norm(X, axis=1).max())
            pc = meths[-1][1]
            ex = meths[0][1]
            e = np.linalg.norm(pc(U) - ex(U), axis=1) / np.linalg.norm(ex(U), axis=1)
            res["closure_err"] = e
            print(f"  inputs: std max {res['input_std_max']:.3f}, spectral radius max {res['input_rho_max']:.2f}; "
                  f"PINN closure err median {np.median(e):.2e} max {e.max():.2e}", flush=True)
        out[f"d{d}p{p}"] = res
    with open(os.path.join(RES, "e5_ignition.pkl"), "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main()
