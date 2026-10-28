"""E3: damped nonlinear pendulum with random initial angle and natural frequency (d = 2)."""
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from closures import SinCos, ExactGalerkin, PSP, Taylor, PINNClosure, Lanczos
from problems import Pendulum
from propagate import monte_carlo, intrusive, compare
from xtfc import SpectralXTFC

D, P_ORDER, T_END, N_MC = 2, 6, 30.0, 100_000


def closure_error_along(prob, C, closure, exact):
    th = C[:, 0, :]
    ge, gp = exact(th), closure(th)
    return np.linalg.norm(gp - ge, axis=1) / np.linalg.norm(ge, axis=1)


def main():
    B = PCBasis(D, P_ORDER)
    prob = Pendulum(B)
    nl = SinCos(B)
    t_eval = np.linspace(0, T_END, 121)
    rng = np.random.default_rng(2026)
    xi = rng.normal(size=(N_MC, D))
    t0 = time.perf_counter()
    Ymc = monte_carlo(prob, xi, t_eval)
    t_mc = time.perf_counter() - t0
    print(f"MC: {t_mc:.1f}s", flush=True)
    Psi = B.eval_basis(xi)
    exact = ExactGalerkin(nl)

    out = dict(t=t_eval, t_mc=t_mc, methods={})

    def run(name, cl):
        C, wall, nfev, ok = intrusive(prob, cl, t_eval)
        if not ok:
            print(f"{name}: FAILED", flush=True)
            out["methods"][name] = dict(ok=False, wall=wall, nfev=nfev)
            return None
        m = compare(B, C, Ymc, Psi)
        out["methods"][name] = dict(ok=True, C=C, wall=wall, nfev=nfev, **m)
        print(f"{name:14s} wall {wall:6.2f}s nfev {nfev:5d}  L2(theta) t=10,20,30: "
              f"{m['l2_err'][[40, 80, 120], 0].round(4)}  BC: {m['bc'][[40, 80, 120], 0].round(3)}", flush=True)
        return C

    run("eig", exact)
    run("psp", PSP(nl, P_ORDER + 1))
    run("taylor8", Taylor(nl, 8))
    run("lanczos", Lanczos(nl, 10))

    # generic PINN closure (trained once in E1, problem independent)
    pc = PINNClosure.load(nl, os.path.join(MODELS, "sincos_d2p6.npz"))
    pc.label = "PINN (generic)"
    Cg = run("pinn_generic", pc)

    # trajectory-anchored fine-tuning.  The generic closure is not reliable on this problem, so the
    # anchor inputs are taken from a cheap pilot run (PSP with q = p+1).  Only the *inputs* visited by
    # the pilot are used to place collocation points; no closure values (labels) are used.
    tp = np.linspace(0, T_END, 601)
    Cp, _, _, _ = intrusive(prob, PSP(nl, P_ORDER + 1), tp)
    anchors = Cp[:, 0, :]
    t1 = time.perf_counter()
    pc.train(n_dirs=20000, batch=256, iters=6000, lr=1e-3, lbfgs_iters=2500, lbfgs_batch=2048,
             anchors=anchors, warm_start=True, seed=5, verbose=False)
    out["t_finetune"] = time.perf_counter() - t1
    out["finetune_val_err"] = pc.final_val_err
    pc.save(os.path.join(MODELS, "sincos_d2p6_pendulum.npz"))
    print(f"fine-tune {out['t_finetune']:.0f}s, val err {pc.final_val_err:.2e}", flush=True)
    run("pinn_anchored", pc)
    if out["methods"]["pinn_anchored"].get("ok"):
        out["methods"]["pinn_anchored"]["closure_err"] = closure_error_along(
            prob, out["methods"]["eig"]["C"], pc, exact)
    pcg = PINNClosure.load(nl, os.path.join(MODELS, "sincos_d2p6.npz"))
    out["closure_err_generic_on_exact_traj"] = closure_error_along(prob, out["methods"]["eig"]["C"], pcg, exact)

    # solution-level PINN (spectral X-TFC) with Gauss-Hermite nodes in xi
    X, W = B.quadrature(P_ORDER + 1)
    for L, m, dt in [(12, 16, 1.0)]:
        xt = SpectralXTFC(prob, X, W, L=L, m=m, dt=dt, gn_tol=1e-9)
        C = xt.propagate(prob.pc_initial(), T_END, t_eval)
        mm = compare(B, C, Ymc, Psi)
        out["methods"]["xtfc"] = dict(ok=True, C=C, wall=xt.stats["wall"], gn=np.mean(xt.stats["gn_iters"]),
                                      windows=xt.stats["windows"], L=L, m=m, dt=dt, **mm)
        print(f"xtfc L={L} wall {xt.stats['wall']:.1f}s  L2: {mm['l2_err'][[40, 80, 120], 0].round(4)}", flush=True)

    # samples for PDF figures (theta at t = 10, 20, 30)
    sel = [40, 80, 120]
    out["pdf"] = dict(idx=sel, mc=Ymc[sel][:, :, :20000])
    out["Psi_sub"] = Psi[:20000]
    with open(os.path.join(RES, "e3_pendulum.pkl"), "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main()
