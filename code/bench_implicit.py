"""Cost of one batched evaluation of the implicit (cubic-constraint) closures vs stochastic dimension."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pce import PCBasis, lognormal_coeffs, linear_coeffs
from implicit import CubicSystem, NewtonClosure, NewtonKrylovClosure, CardanoPSP, PINNImplicit

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "bench_implicit.json")


def timeit(f, x, reps):
    """Median wall time of `reps` calls after one warm-up call."""
    f(x)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(x)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main(batch=8):
    rows = []
    rng = np.random.default_rng(0)
    for d in [2, 3, 4, 5, 6, 7, 8]:
        B = PCBasis(d, 3)
        k1 = lognormal_coeffs(B, -0.005, 0.1, 1)
        k3 = lognormal_coeffs(B, np.log(2.0) - 0.02, 0.2, min(2, d - 1))
        s = CubicSystem(B, k1, 0.5 * B.e0, k3)
        x = np.zeros((batch, B.P))
        x[:, 0] = rng.uniform(-2, 2, batch)
        x[:, B.lin_index[0]] = 0.3
        x[:, 1:] += 0.02 * rng.normal(size=(batch, B.P - 1))
        pn = PINNImplicit(s, width=64, depth=3)            # untrained: timing only
        meths = {"newton": NewtonClosure(s), "nk": NewtonKrylovClosure(s, m=10),
                 "pinn": pn, "pinn_newton1": NewtonClosure(s, init=pn, fixed_steps=1),
                 "pinn_nk1": NewtonKrylovClosure(s, m=5, init=pn, fixed_steps=1),
                 "pinn_nk2": NewtonKrylovClosure(s, m=5, init=pn, fixed_steps=2),
                 "psp_s0": CardanoPSP(s, "s3"), "psp_s1": CardanoPSP(s, "s4")}
        if 4 ** d <= 70000:
            meths["psp"] = CardanoPSP(s, 4)
        row = dict(d=d, p=3, P=B.P, nnz=B.nnz, batch=batch, Q=4 ** d,
                   Q_s0=meths["psp_s0"].Psi.shape[0], Q_s1=meths["psp_s1"].Psi.shape[0])
        for k, f in meths.items():
            row[k] = timeit(f, x, 3 if k == "newton" and B.P > 100 else 10)
        rows.append(row)
        print({k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
