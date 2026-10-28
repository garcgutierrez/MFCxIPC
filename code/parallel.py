"""Multi-core execution of node-batched closures.

NodeParallel(closure, n_nodes, workers, static=None) forks `workers` persistent processes that inherit the (already
built) closure and each own a fixed contiguous block of nodes, so that per-node state (warm starts, frozen Jacobians,
cached multiplication matrices) stays consistent across calls.  A call closure(H) or closure(H, K) sends each worker
its rows of H through a pipe (raw bytes, no pickling) and concatenates the results; a static second argument (e.g. the
k3 field of the heat-conduction problem) is sliced once and kept by the workers.  Closures with per-node cached data
(FieldPSP) are restricted to the block through restrict(block).

Workers run single-threaded BLAS; the parent keeps the rest of the right-hand side.  Fork start method (Linux).
Iterative closures stop per block instead of per batch, so results agree with the serial ones to the solver
tolerance (non-iterative closures: up to floating-point reordering of the batched products).
"""
import multiprocessing as mp
import os

import numpy as np


def _restrict(closure, block):
    if hasattr(closure, "k3q"):                         # FieldPSP: k3 at the quadrature nodes, cached per node
        closure.k3q = closure.k3q[block]
    return closure


def _worker(conn, closure, block, static, out_cols):
    os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    closure = _restrict(closure, block)
    K = None if static is None else np.ascontiguousarray(static[block])
    nb = len(block)
    while True:
        msg = conn.recv_bytes()
        if not msg:
            break
        H = np.frombuffer(msg, dtype=np.float64).reshape(nb, -1)
        Y = closure(H) if K is None else closure(H, K)
        conn.send_bytes(np.ascontiguousarray(Y, dtype=np.float64).tobytes())
    conn.close()


class NodeParallel:
    def __init__(self, closure, n_nodes, workers, static=None, out_cols=None):
        ctx = mp.get_context("fork")
        self.blocks = [b for b in np.array_split(np.arange(n_nodes), workers) if len(b)]
        self.conns, self.procs = [], []
        for b in self.blocks:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, closure, b, static, out_cols), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self.label = getattr(closure, "label", "closure") + f" [{len(self.blocks)} workers]"
        self.iters = getattr(closure, "iters", None)

    def __call__(self, H, K=None):
        H = np.ascontiguousarray(H, dtype=np.float64)
        for c, b in zip(self.conns, self.blocks):
            c.send_bytes(H[b].tobytes())
        parts = [np.frombuffer(c.recv_bytes(), dtype=np.float64).reshape(len(b), -1)
                 for c, b in zip(self.conns, self.blocks)]
        return np.concatenate(parts, axis=0)

    def close(self):
        for c in self.conns:
            try:
                c.send_bytes(b"")
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
