"""E3b: self-consistent anchored fine-tuning of the sine/cosine closure for the pendulum.

Round k: run the Galerkin system with the current neural closure (until it ends or fails), add the
inputs it visited to the anchor set, fine-tune (warm start, label-free) and repeat."""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from closures import SinCos, ExactGalerkin, PSP, PINNClosure
from problems import Pendulum
from propagate import monte_carlo, intrusive, compare
from e3_pendulum import D, P_ORDER, T_END, N_MC, closure_error_along

MAX_ROUNDS = 4


def main():
    B = PCBasis(D, P_ORDER)
    prob = Pendulum(B)
    nl = SinCos(B)
    exact = ExactGalerkin(nl)
    t_eval = np.linspace(0, T_END, 121)
    rng = np.random.default_rng(2026)
    xi = rng.normal(size=(N_MC, D))
    Ymc = monte_carlo(prob, xi, t_eval)
    Psi = B.eval_basis(xi)
    out = pickle.load(open(os.path.join(RES, "e3_pendulum.pkl"), "rb"))

    pc = PINNClosure.load(nl, os.path.join(MODELS, "sincos_d2p6_pendulum.npz"))
    tp = np.linspace(0, T_END, 601)
    Cp, _, _, _ = intrusive(prob, PSP(nl, P_ORDER + 1), tp)
    anchors = [Cp[:, 0, :]]
    rounds = []
    for k in range(1, MAX_ROUNDS + 1):
        C, wall, nfev, ok = intrusive(prob, pc, tp)
        t_reached = tp[len(C) - 1]
        anchors.append(C[:, 0, :])
        rounds.append(dict(round=k, ok=ok, t_reached=float(t_reached)))
        print(f"round {k}: run {'completed' if ok else 'failed'} (t reached {t_reached:.2f})", flush=True)
        if ok:
            break
        t1 = time.perf_counter()
        pc.train(n_dirs=20000, batch=256, iters=3000, lr=5e-4, lbfgs_iters=1500, lbfgs_batch=2048,
                 anchors=np.concatenate(anchors), warm_start=True, seed=10 + k, verbose=False)
        rounds[-1]["t_finetune"] = time.perf_counter() - t1
        rounds[-1]["val_err"] = pc.final_val_err
        print(f"  fine-tune {rounds[-1]['t_finetune']:.0f}s, val err {pc.final_val_err:.2e}", flush=True)
    pc.save(os.path.join(MODELS, "sincos_d2p6_pendulum_sc.npz"))

    C, wall, nfev, ok = intrusive(prob, pc, t_eval)
    out["selfconsistent_rounds"] = rounds
    if ok:
        m = compare(B, C, Ymc, Psi)
        out["methods"]["pinn_anchored"] = dict(ok=True, C=C, wall=wall, nfev=nfev, **m,
                                               closure_err=closure_error_along(prob, out["methods"]["eig"]["C"], pc, exact))
        print(f"pinn_anchored  wall {wall:6.2f}s nfev {nfev:5d}  L2(theta) t=10,20,30: "
              f"{m['l2_err'][[40, 80, 120], 0].round(4)}  BC: {m['bc'][[40, 80, 120], 0].round(3)}", flush=True)
        ce = out["methods"]["pinn_anchored"]["closure_err"]
        print(f"closure err along exact trajectory: median {np.median(ce):.2e} max {ce.max():.2e}")
    else:
        out["methods"]["pinn_anchored"] = dict(ok=False, wall=wall, nfev=nfev)
        print("pinn_anchored still FAILED")
    with open(os.path.join(RES, "e3_pendulum.pkl"), "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main()
