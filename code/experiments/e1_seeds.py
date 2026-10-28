"""E1s: sensitivity of a generic closure (exp, d = 2, p = 4) to the network initialisation and training samples.
Usage: python e1_seeds.py <seed>   (seed 0 is the network of Table 1)"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from closures import Exp, PINNClosure
from e1_closures import assess

if __name__ == "__main__":
    k = int(sys.argv[1])
    B = PCBasis(2, 4)
    nl = Exp(B)
    pc = PINNClosure(nl, K=3, width=64, depth=3, r_max=2.0, R_eff=4.0, seed=k)
    t0 = time.perf_counter()
    pc.train(n_dirs=30000, batch=256, iters=8000, lr=3e-3, lbfgs_iters=2500, lbfgs_batch=4096, seed=1 + 100 * k,
             verbose=False)
    t = time.perf_counter() - t0
    pc.save(os.path.join(MODELS, f"exp_d2p4_seed{k}.npz"))
    res = assess(pc, nl, B)                      # same test set as Table 1 (fixed seed inside assess)
    e = np.array(res["pinn"])
    out = dict(seed=k, t_train=t, med=float(np.median(e)), p95=float(np.percentile(e, 95)))
    json.dump(out, open(os.path.join(RES, f"e1s_exp_d2p4_seed{k}.json"), "w"))
    print(out, flush=True)
