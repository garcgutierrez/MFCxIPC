"""E1b: re-assess the trained generic closures, adding the Lanczos closure (no retraining)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from closures import PINNClosure, Lanczos
from e1_closures import CASES, assess


def main():
    for name, d, p, mk, R_eff, K in CASES:
        B = PCBasis(d, p)
        nl = mk(B)
        pc = PINNClosure.load(nl, os.path.join(MODELS, f"{name}.npz"))
        path = os.path.join(RES, f"e1_{name}.json")
        res = json.load(open(path))
        new = assess(pc, nl, B)
        # same test set (same seed) -> add Lanczos errors
        rng = np.random.default_rng(2024)
        from closures import sample_directions, ExactGalerkin
        D = sample_directions(B, 4000, rng)
        r = pc.r_limit(D) * rng.random(4000)
        X = r[:, None] * D
        u = X + B.e0 if nl.c != 0 else X
        Ge = ExactGalerkin(nl)(u)
        nrm = np.linalg.norm(Ge, axis=1)
        for m in [6, 8, 10]:
            res[f"lanczos{m}"] = (np.linalg.norm(Lanczos(nl, m)(u) - Ge, axis=1) / nrm).tolist()
        assert np.allclose(new["pinn"], res["pinn"]), "test set mismatch"
        with open(path, "w") as fh:
            json.dump(res, fh)
        line = f"{name:12s} PINN {np.median(res['pinn']):.1e}/{np.percentile(res['pinn'], 95):.1e}"
        for k in ["taylor4", "taylor8", "lanczos6", "lanczos8", "lanczos10"]:
            line += f" | {k} {np.median(res[k]):.1e}/{np.percentile(res[k], 95):.1e}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
