"""E4: genetic toggle switch with four log-normal parameters (d = 4)."""
import os
import pickle
import sys
import time

import numpy as np
from scipy.stats import norm, qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from closures import Power, ExactGalerkin, PSP, Taylor, PINNClosure, Lanczos
from problems import ToggleSwitch
from propagate import monte_carlo, intrusive, compare
from xtfc import SpectralXTFC

D, P_ORDER, T_END, N_MC = 4, 3, 10.0, 100_000


def main():
    B = PCBasis(D, P_ORDER)
    prob = ToggleSwitch(B, y0=(10.0, 0.5))
    t_eval = np.linspace(0, T_END, 101)
    xi = np.random.default_rng(2027).normal(size=(N_MC, D))
    t0 = time.perf_counter()
    Ymc = monte_carlo(prob, xi, t_eval)
    t_mc = time.perf_counter() - t0
    print(f"MC: {t_mc:.1f}s", flush=True)
    Psi = B.eval_basis(xi)
    out = dict(t=t_eval, t_mc=t_mc, methods={})
    sel = [10, 30, 100]

    def run(name, cl):
        C, wall, nfev, ok = intrusive(prob, cl, t_eval)
        if not ok:
            print(f"{name}: FAILED", flush=True)
            out["methods"][name] = dict(ok=False)
            return
        m = compare(B, C, Ymc, Psi)
        out["methods"][name] = dict(ok=True, C=C, wall=wall, nfev=nfev, **m)
        print(f"{name:14s} wall {wall:6.2f}s nfev {nfev:5d}  L2 u: {m['l2_err'][sel, 0].round(4)} "
              f"v: {m['l2_err'][sel, 1].round(4)}", flush=True)

    mk = lambda f: dict(pow_beta=f(Power(B, prob.beta)), inv=f(Power(B, -1.0)))
    run("eig", mk(ExactGalerkin))
    run("psp", mk(lambda nl: PSP(nl, P_ORDER + 1)))
    run("taylor4", mk(lambda nl: Taylor(nl, 4)))
    run("lanczos", mk(lambda nl: Lanczos(nl, 10)))
    pinn = dict(pow_beta=PINNClosure.load(Power(B, prob.beta), os.path.join(MODELS, "pow2.5_d4p3.npz")),
                inv=PINNClosure.load(Power(B, -1.0), os.path.join(MODELS, "pow-1_d4p3.npz")))
    run("pinn_generic", pinn)
    # closure error of the PINN closures along the exact-closure trajectory
    if out["methods"]["eig"]["ok"]:
        C = out["methods"]["eig"]["C"]
        V = C[:, 1, :]
        ex_b, ex_i = ExactGalerkin(Power(B, prob.beta)), ExactGalerkin(Power(B, -1.0))
        e1 = np.linalg.norm(pinn["pow_beta"](V) - ex_b(V), axis=1) / np.linalg.norm(ex_b(V), axis=1)
        den = B.e0 + ex_b(V)
        e2 = np.linalg.norm(pinn["inv"](den) - ex_i(den), axis=1) / np.linalg.norm(ex_i(den), axis=1)
        out["closure_err"] = dict(pow_beta=e1, inv=e2)
        print(f"closure err along trajectory: pow {np.median(e1):.2e}/{e1.max():.2e}  inv {np.median(e2):.2e}/{e2.max():.2e}")

    # solution-level PINN: Gauss-Hermite tensor nodes and (scrambled Sobol) random nodes
    X, W = B.quadrature(P_ORDER + 1)
    variants = [("xtfc_gh", X, W)]
    for fac in [2, 4]:
        n = fac * B.P
        s = qmc.Sobol(D, scramble=True, seed=11).random(n)
        variants.append((f"xtfc_qmc{fac}P", norm.ppf(s), np.ones(n)))
    for name, Xn, Wn in variants:
        xt = SpectralXTFC(prob, Xn, Wn, L=12, m=16, dt=0.5, gn_tol=1e-9)
        C = xt.propagate(prob.pc_initial(), T_END, t_eval)
        m = compare(B, C, Ymc, Psi)
        out["methods"][name] = dict(ok=True, C=C, wall=xt.stats["wall"], Q=len(Wn),
                                    gn=np.mean(xt.stats["gn_iters"]), **m)
        print(f"{name:14s} Q={len(Wn)} wall {xt.stats['wall']:.1f}s  L2 u: {m['l2_err'][sel, 0].round(4)} "
              f"v: {m['l2_err'][sel, 1].round(4)}", flush=True)
    out["pdf"] = dict(idx=sel, mc=Ymc[sel][:, :, :20000])
    out["Psi_sub"] = Psi[:20000]
    with open(os.path.join(RES, "e4_toggle.pkl"), "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main()
