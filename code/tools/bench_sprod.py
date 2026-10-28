"""Cost of one batched Galerkin product (PCBasis.sprod, d = 8, p = 3) against the number of inputs in the batch,
single-threaded.  Explains why distributing the 15 nodes of the heat-conduction problem over processes gains little
for closures built from Galerkin products (Supplementary Section S6)."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except ImportError:
    pass
from pce import PCBasis

B = PCBasis(8, 3)
rng = np.random.default_rng(0)
for nb in (1, 4, 8, 15, 60):
    u, v = rng.normal(size=(2, nb, B.P))
    B.sprod(u, v)
    t0 = time.perf_counter()
    for _ in range(200):
        B.sprod(u, v)
    print(f"batch {nb:3d}: {(time.perf_counter() - t0) / 200 * 1e3:.3f} ms per product")
