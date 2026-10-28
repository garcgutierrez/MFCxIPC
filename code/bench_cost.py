"""Cost of one batched closure evaluation vs stochastic dimension d and order p."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pce import PCBasis
from closures import Exp, ExactGalerkin, PSP, Taylor, PINNClosure, Lanczos, sample_directions

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "bench_cost.json")


def timeit(f, u, reps):
    """Median wall time of `reps` calls after one warm-up call."""
    f(u)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(u)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main(batch=31):
    rows = []
    rng = np.random.default_rng(0)
    for d, p in [(1, 4), (2, 4), (3, 4), (4, 3), (4, 4), (5, 3), (6, 3), (6, 4), (8, 3)]:
        B = PCBasis(d, p)
        nl = Exp(B)
        x = 0.2 * sample_directions(B, batch, rng)
        u = x + 0.5 * B.e0
        row = dict(d=d, p=p, P=B.P, nnz=B.nnz, batch=batch)
        meths = {"eig": ExactGalerkin(nl), "taylor4": Taylor(nl, 4), "taylor8": Taylor(nl, 8),
                 "lanczos10": Lanczos(nl, 10),
                 "pinn": PINNClosure(nl, K=3, width=64, depth=3, r_max=1.0)}
        Q = (p + 1) ** d
        if Q <= 70000:
            meths["psp"] = PSP(nl, p + 1)
        row["Q"] = Q
        for k in (p, p + 1):                               # Smolyak sparse rules of level k
            meths[f"psp_s{k - p}"] = PSP(nl, f"s{k}")
            row[f"Q_s{k - p}"] = meths[f"psp_s{k - p}"].Psi.shape[0]
        for k, f in meths.items():
            reps = 3 if k == "eig" and B.P > 150 else 20
            row[k] = timeit(f, u, reps)
        rows.append(row)
        print({k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
