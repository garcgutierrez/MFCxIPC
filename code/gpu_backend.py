"""
Node-batched closures on an array module (NumPy or CuPy) for the node-scaling / GPU benchmark (experiments/e12_scaling.py).

Nothing here replaces the library: pce.py, closures.py, implicit.py and implicit_field.py are untouched.  With xp = numpy the
Galerkin products are the library's own; with xp = cupy they are re-implemented (gather u[:, I] * v[:, J], then a sparse
product with the stored triple-product values).  Every method keeps the algorithm, tolerances and stopping tests of the
library class it mirrors:

  FieldSolverX    <-  implicit_field.FieldSolver  (newton | nk | chord; linear | warm | callable start; fixed steps),
                      plus an active-set variant that keeps iterating only on the unconverged nodes
  gmres_vec       <-  implicit.gmres_batched      (the least-squares step by Givens rotations on the whole batch
                                                   instead of a Python loop over the nodes of np.linalg.lstsq)
  FieldPSPX       <-  implicit_field.FieldPSP     (chunked over nodes; tensor rules do not fit in memory at large N)
  FieldPINNX      <-  implicit_field.FieldPINN    (both forward passes in one stacked batch)
  ExactGalerkinX, LanczosX, TaylorX, PSPX, PINNClosureX  <-  closures.py (exponential nonlinearity only)

Chunking (reported with the results): products, Jacobian assemblies, inversions and quadrature evaluations are processed in
chunks of nodes whose temporaries stay below `Ops.work` bytes.  The multiplication matrix of k3 is cached only when it is
smaller than `Ops.cache_bytes`, otherwise it is rebuilt per chunk at every Jacobian refresh (same numbers, a little more work).
"""
import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
except Exception:                                      # CuPy absent: only xp = numpy is available
    cp = csp = None


def get_xp(device):
    if device.startswith("gpu"):
        if cp is None:
            raise RuntimeError("CuPy is not installed")
        return cp
    return np


def sync(xp):
    if xp is not np:
        xp.cuda.Device().synchronize()


def to_host(a):
    return a if isinstance(a, np.ndarray) else a.get()


def pool_bytes():
    return cp.get_default_memory_pool().total_bytes() if cp is not None else 0


# =============================================================================
# Galerkin algebra
# =============================================================================
class Ops:
    """Galerkin products and multiplication matrices of a PCBasis on one array module."""

    def __init__(self, basis, xp=np, dtype=np.float64, work_bytes=256e6, cache_bytes=1.0e9):
        self.b, self.xp, self.P, self.dtype = basis, xp, basis.P, np.dtype(dtype)
        self.work, self.cache_bytes = float(work_bytes), float(cache_bytes)
        self.nnz = basis.nnz
        self.gpu = xp is not np
        if self.gpu:
            self._I, self._J = xp.asarray(basis._I), xp.asarray(basis._J)
            self._ST = csp.csr_matrix(basis._S.T.tocsr().astype(self.dtype))       # (P, nnz)
            self._MT = csp.csr_matrix(basis._MT.astype(self.dtype))                 # (P^2, P)
            self._chunk_sprod = max(1, int(self.work // (self.nnz * self.dtype.itemsize * 3)))
        self.eye = xp.eye(self.P, dtype=self.dtype)
        self.e0 = xp.zeros(self.P, dtype=self.dtype)
        self.e0[0] = 1.0

    # ---- helpers ------------------------------------------------------------
    def asarray(self, a):
        return self.xp.asarray(a, dtype=self.dtype)

    def chunk_nodes(self, per_node_bytes):
        return max(1, int(self.work // per_node_bytes))

    def norm(self, a):
        return self.xp.linalg.norm(a, axis=1)

    # ---- products -----------------------------------------------------------
    def sprod(self, u, v):
        """Galerkin product (u * v) for arrays (B, P) (broadcasting over leading axes)."""
        if not self.gpu:
            return self.b.sprod(u, v)
        xp = self.xp
        u, v = xp.broadcast_arrays(u, v)
        sh = u.shape
        u2, v2 = u.reshape(-1, self.P), v.reshape(-1, self.P)
        nb = u2.shape[0]
        if nb <= self._chunk_sprod:
            return xp.ascontiguousarray((self._ST @ (u2[:, self._I] * v2[:, self._J]).T).T).reshape(sh)
        out = xp.empty((nb, self.P), dtype=u2.dtype)
        c = self._chunk_sprod
        for i in range(0, nb, c):
            out[i:i + c] = (self._ST @ (u2[i:i + c][:, self._I] * v2[i:i + c][:, self._J]).T).T
        return out.reshape(sh)

    def mult_matrix(self, u):
        """M_u (B, P, P) from u (B, P); the caller chunks the batch (see chunk_nodes)."""
        if not self.gpu:
            return self.b.mult_matrix(u)
        B = u.shape[0]
        return self.xp.ascontiguousarray((self._MT @ u.T).T).reshape(B, self.P, self.P)

    def eigh(self, M):
        """Eigendecomposition of a stack of symmetric matrices (B, n, n).  GPU: cuSOLVER syevd matrix by matrix for
        n > 32 (measured faster than the batched Jacobi routine at n = 165), batched Jacobi (syevjBatched) otherwise."""
        if not self.gpu or M.shape[-1] <= 32:
            return self.xp.linalg.eigh(M)
        lam = self.xp.empty(M.shape[:2], dtype=M.dtype)
        V = self.xp.empty_like(M)
        for i in range(M.shape[0]):
            lam[i], V[i] = self.xp.linalg.eigh(M[i])
        return lam, V


def cubic_root(xp, a3, a1, b):
    """Real root of a3 y^3 + a1 y = b with a3, a1 > 0 (Cardano), same formula as implicit.cubic_root."""
    p = a1 / a3
    q = -b / a3
    disc = xp.sqrt(q * q / 4 + p ** 3 / 27)
    return xp.cbrt(-q / 2 + disc) + xp.cbrt(-q / 2 - disc)


# =============================================================================
# Vectorised GMRES
# =============================================================================
def gmres_vec(xp, matvec, rhs, m):
    """m steps of GMRES (zero initial guess, modified Gram-Schmidt) vectorised over the batch, as
    implicit.gmres_batched, but the small least-squares problem min ||H z - beta e1|| is solved for all nodes at once
    by Givens rotations and back substitution instead of one np.linalg.lstsq call per node."""
    Bsz, P = rhs.shape
    beta = xp.linalg.norm(rhs, axis=1)
    V = xp.zeros((m + 1, Bsz, P), dtype=rhs.dtype)
    H = xp.zeros((Bsz, m + 1, m), dtype=rhs.dtype)
    V[0] = rhs / xp.maximum(beta, 1e-300)[:, None]
    k = m
    for j in range(m):
        w = matvec(V[j])
        for i in range(j + 1):
            H[:, i, j] = xp.sum(w * V[i], axis=1)
            w = w - H[:, i, j, None] * V[i]
        H[:, j + 1, j] = xp.linalg.norm(w, axis=1)
        V[j + 1] = w / xp.maximum(H[:, j + 1, j], 1e-300)[:, None]
        if bool(xp.all(H[:, j + 1, j] < 1e-14 * beta)):
            k = j + 1
            break
    g = xp.zeros((Bsz, k + 1), dtype=rhs.dtype)
    g[:, 0] = beta
    cs = xp.ones((Bsz, k), dtype=rhs.dtype)
    sn = xp.zeros((Bsz, k), dtype=rhs.dtype)
    Hk = H[:, :k + 1, :k].copy()
    for j in range(k):
        for i in range(j):                                     # earlier rotations applied to column j
            a, b = Hk[:, i, j].copy(), Hk[:, i + 1, j].copy()
            Hk[:, i, j] = cs[:, i] * a + sn[:, i] * b
            Hk[:, i + 1, j] = -sn[:, i] * a + cs[:, i] * b
        a, b = Hk[:, j, j].copy(), Hk[:, j + 1, j].copy()
        r = xp.hypot(a, b)
        ok = r > 0
        rs = xp.where(ok, r, 1.0)
        cs[:, j] = xp.where(ok, a / rs, 1.0)
        sn[:, j] = xp.where(ok, b / rs, 0.0)
        Hk[:, j, j] = r
        Hk[:, j + 1, j] = 0.0
        g[:, j + 1] = -sn[:, j] * g[:, j]
        g[:, j] = cs[:, j] * g[:, j]
    z = xp.zeros((Bsz, k), dtype=rhs.dtype)
    for i in range(k - 1, -1, -1):
        s = g[:, i] - xp.sum(Hk[:, i, i + 1:k] * z[:, i + 1:k], axis=1)
        d = Hk[:, i, i]
        z[:, i] = xp.where(xp.abs(d) > 1e-300, s / xp.where(xp.abs(d) > 1e-300, d, 1.0), 0.0)
    return xp.einsum('kbp,bk->bp', V[:k], z)


# =============================================================================
# Heat conduction: implicit law k3 T^3 + T = h
# =============================================================================
class CubicFieldX:
    """Galerkin algebra of R(T; h, k3) = k3 * (T * (T * T)) + T - h  (implicit_field.CubicField on an array module)."""

    def __init__(self, ops):
        self.o, self.P = ops, ops.P

    def residual(self, h, k3, y):
        o = self.o
        return o.sprod(k3, o.sprod(y, o.sprod(y, y))) + y - h

    def jv(self, k3, y, v):
        o = self.o
        return o.sprod(k3, o.sprod(v, o.sprod(y, y)) + 2 * o.sprod(y, o.sprod(y, v))) + v

    def jac_from(self, Mk3, y):
        o = self.o
        My = o.mult_matrix(y)
        return Mk3 @ (o.mult_matrix(o.sprod(y, y)) + 2 * My @ My) + o.eye


class FieldSolverX:
    """implicit_field.FieldSolver on an array module, with the options

    method     'newton' | 'nk' | 'chord'  (as in the library; chord keeps one inverse Jacobian per node across calls)
    init       'linear' | 'warm' | callable(h, k3)
    fixed_steps  no convergence test; the residual after the last step is not evaluated (the library evaluates it,
                 which costs one extra residual per call, unless it is needed for `record`)
    active     False: the library semantics (all nodes iterate until the slowest node converges; the chord refresh
               rule acts on the whole batch).  True: iterate only on the nodes that have not converged; the chord
               inverse is refreshed per node (a node refreshes when its own contraction rate exceeds chord_rate or
               after chord_max iterations); same tolerance, same warm start.
    gmres      'vec' (gmres_vec) | 'orig' (implicit.gmres_batched, NumPy only)
    Counters: n = {calls, jv, jac, res} as in the library (batch level), plus node-level counters in n_nodes
    (node_iters: updates summed over nodes and calls; jac_rows, res_rows).  iters: max over nodes per call;
    iters_mean: mean over nodes per call (exact only for active=True, where each node stops by itself)."""

    def __init__(self, ops, method="nk", init="linear", tol=1e-10, maxit=40, m=10, fixed_steps=None, chord_max=6,
                 chord_rate=0.5, active=False, gmres="vec", record=False):
        self.o, self.xp, self.fs = ops, ops.xp, CubicFieldX(ops)
        self.method, self.init, self.tol, self.maxit, self.m = method, init, tol, maxit, m
        self.fixed, self.chord_max, self.chord_rate, self.active, self.gmres, self.record = (
            fixed_steps, chord_max, chord_rate, active, gmres, record)
        self.iters, self.iters_mean, self.final_res = [], [], []
        self.n = dict(calls=0, jv=0, jac=0, res=0)
        self.n_nodes = dict(node_iters=0, jac_rows=0, res_rows=0)
        self._prev = self._k3 = self._Mk3 = self._Jinv = None
        self.chunk = ops.chunk_nodes(ops.P * ops.P * ops.dtype.itemsize * 6)
        self.label = f"{method} ({init if isinstance(init, str) else 'net'}){' active' if active else ''}"

    # ---- building blocks ----------------------------------------------------
    def _mk3(self, k3):
        """Cached multiplication matrices of k3 (None when too large: rebuilt per chunk)."""
        if self._k3 is not k3:
            self._k3, self._Mk3 = k3, None
            N, P = k3.shape[0], self.o.P
            if N * P * P * self.o.dtype.itemsize <= self.o.cache_bytes:
                self._Mk3 = self.xp.empty((N, P, P), dtype=self.o.dtype)
                for i in range(0, N, self.chunk):
                    self._Mk3[i:i + self.chunk] = self.o.mult_matrix(k3[i:i + self.chunk])
        return self._Mk3

    def _jac_chunk(self, k3, y, i, j, rows=None):
        """Jacobians of the nodes i:j (or of the nodes `rows[i:j]`)."""
        Mk3 = self._mk3(k3)
        sel = slice(i, j) if rows is None else rows[i:j]
        Mk = Mk3[sel] if Mk3 is not None else self.o.mult_matrix(k3[sel])
        return self.fs.jac_from(Mk, y[sel])

    def _newton_step(self, k3, y, R, rows=None):
        """y - J^{-1} R for all nodes (rows=None) or for the nodes `rows` (y, R already restricted to rows)."""
        xp = self.xp
        n = y.shape[0]
        out = xp.empty_like(y)
        full = rows is None
        for i in range(0, n, self.chunk):
            j = min(n, i + self.chunk)
            Mk = self._mk3(k3)
            sel = slice(i, j) if full else rows[i:j]
            Mk = Mk[sel] if Mk is not None else self.o.mult_matrix(k3[sel])
            J = self.fs.jac_from(Mk, y[i:j])
            out[i:j] = y[i:j] - xp.linalg.solve(J, R[i:j, :, None])[..., 0]
        self.n["jac"] += 1
        self.n_nodes["jac_rows"] += n
        return out

    def _refresh(self, k3, y, rows=None):
        """Recompute the stored inverse Jacobians (all nodes, or the nodes `rows` with states y[rows])."""
        xp = self.xp
        N = k3.shape[0]
        if self._Jinv is None or self._Jinv.shape[0] != N:
            self._Jinv = xp.empty((N, self.o.P, self.o.P), dtype=self.o.dtype)
        if rows is None:
            for i in range(0, N, self.chunk):
                j = min(N, i + self.chunk)
                self._Jinv[i:j] = xp.linalg.inv(self._jac_chunk(k3, y, i, j))
            self.n_nodes["jac_rows"] += N
        else:
            for i in range(0, len(rows), self.chunk):
                j = min(len(rows), i + self.chunk)
                J = self._jac_chunk(k3, y, i, j, rows)
                self._Jinv[rows[i:j]] = xp.linalg.inv(J)
            self.n_nodes["jac_rows"] += len(rows)
        self.n["jac"] += 1

    def _gmres(self, matvec, R):
        if self.gmres == "orig":
            from implicit import gmres_batched
            return gmres_batched(matvec, R, self.m)
        return gmres_vec(self.xp, matvec, R, self.m)

    def _guess(self, h, k3):
        if callable(self.init):
            return self.init(h, k3)
        if self.init == "warm" and self._prev is not None and self._prev.shape == h.shape:
            return self._prev.copy()
        return h.copy()

    # ---- the solve ----------------------------------------------------------
    def __call__(self, h, k3):
        self.n["calls"] += 1
        if self.active:
            return self._call_active(h, k3)
        xp, fs, o = self.xp, self.fs, self.o
        N = h.shape[0]
        y = self._guess(h, k3)
        ref = o.norm(h) + 1e-300
        nmax = self.fixed if self.fixed is not None else self.maxit
        rn_prev, it, rn = None, 0, None
        refresh = self._Jinv is None or self._Jinv.shape[0] != N
        need_last = self.fixed is None or self.record
        for it in range(nmax + 1):
            if it == nmax and not need_last:
                break
            R = fs.residual(h, k3, y)
            self.n["res"] += 1
            self.n_nodes["res_rows"] += N
            rn = o.norm(R) / ref
            if self.fixed is None and bool(xp.all(rn <= self.tol)):
                break
            if it == nmax:
                break
            if self.method == "newton":
                y = self._newton_step(k3, y, R)
            elif self.method == "nk":
                y = y - self._gmres(lambda v: fs.jv(k3, y, v), R)
                self.n["jv"] += self.m
            elif self.method == "chord":
                if (refresh or it >= self.chord_max
                        or (rn_prev is not None and float(xp.max(rn / xp.maximum(rn_prev, 1e-300))) > self.chord_rate)):
                    self._refresh(k3, y)
                    refresh = False
                y = y - (self._Jinv @ R[:, :, None])[:, :, 0]
            else:
                raise ValueError(self.method)
            self.n_nodes["node_iters"] += N
            rn_prev = rn
        self.iters.append(it)
        self.iters_mean.append(float(it))
        if self.record and rn is not None:
            self.final_res.append(float(rn.max()))
        self._prev = y
        return y

    def _call_active(self, h, k3):
        """Iterate only on the unconverged nodes.  Residuals of finished nodes are not recomputed."""
        xp, fs, o = self.xp, self.fs, self.o
        N = h.shape[0]
        y = self._guess(h, k3)
        ref = o.norm(h) + 1e-300
        nmax = self.fixed if self.fixed is not None else self.maxit
        act = xp.arange(N)
        its = xp.zeros(N, dtype=xp.int64)
        rn_prev = None
        first = self.method == "chord" and (self._Jinv is None or self._Jinv.shape[0] != N)
        if first:
            self._refresh(k3, y)
            first = False
        full = True                                         # act is still all nodes (avoids gathers)
        for it in range(nmax + 1):
            if full:
                ha, ka, ya, refa = h, k3, y, ref
            else:
                ha, ka, ya, refa = h[act], k3[act], y[act], ref[act]
            if it == nmax and self.fixed is not None and not self.record:
                break
            R = fs.residual(ha, ka, ya)
            self.n["res"] += 1
            self.n_nodes["res_rows"] += len(act)
            rn = o.norm(R) / refa
            if self.fixed is None:
                keep = rn > self.tol
                if not bool(keep.all()):
                    if not bool(keep.any()):
                        break
                    idx = xp.nonzero(keep)[0]
                    act, R, rn, ha, ka, ya = act[idx], R[idx], rn[idx], ha[idx], ka[idx], ya[idx]
                    if rn_prev is not None:
                        rn_prev = rn_prev[idx]
                    full = False
            if it == nmax:
                break
            if self.method == "newton":
                yn = self._newton_step(ka, ya, R) if full else self._newton_step_rows(k3, ya, R, act)
            elif self.method == "nk":
                yn = ya - self._gmres(lambda v: fs.jv(ka, ya, v), R)
                self.n["jv"] += self.m
            elif self.method == "chord":
                if it >= self.chord_max:
                    need = xp.ones(len(act), dtype=bool)
                elif rn_prev is not None:
                    need = rn / xp.maximum(rn_prev, 1e-300) > self.chord_rate
                else:
                    need = None
                if need is not None and bool(need.any()):
                    sel = xp.nonzero(need)[0]
                    self._refresh_rows(k3, y, ya, act, sel)
                yn = ya - self._apply_jinv(act, R, full)
            else:
                raise ValueError(self.method)
            if full:
                y = yn
            else:
                y[act] = yn
            its[act] += 1
            self.n_nodes["node_iters"] += len(act)
            rn_prev = rn
        self.iters.append(int(its.max()))
        self.iters_mean.append(float(its.mean()))
        self._prev = y
        return y

    def _apply_jinv(self, act, R, full):
        """J^{-1} R for the active nodes; the stored inverses are gathered chunk by chunk (a gather of all of them would
        copy up to N P^2 values per iteration)."""
        if full:
            return (self._Jinv @ R[:, :, None])[:, :, 0]
        out = self.xp.empty_like(R)
        for i in range(0, len(act), self.chunk):
            j = min(len(act), i + self.chunk)
            out[i:j] = (self._Jinv[act[i:j]] @ R[i:j, :, None])[:, :, 0]
        return out

    def _newton_step_rows(self, k3, ya, R, act):
        return self._newton_step(k3, ya, R, rows=act)

    def _refresh_rows(self, k3, y, ya, act, sel):
        """Refresh the inverse Jacobians of the active nodes `sel` (positions in act) at their current states."""
        xp = self.xp
        rows = act[sel]
        ysub = ya[sel]
        for i in range(0, len(rows), self.chunk):
            j = min(len(rows), i + self.chunk)
            Mk3 = self._mk3(k3)
            Mk = Mk3[rows[i:j]] if Mk3 is not None else self.o.mult_matrix(k3[rows[i:j]])
            self._Jinv[rows[i:j]] = xp.linalg.inv(self.fs.jac_from(Mk, ysub[i:j]))
        self.n["jac"] += 1
        self.n_nodes["jac_rows"] += len(rows)


class FieldPSPX:
    """Pseudo-spectral projection of the pointwise root, chunked over nodes.  The quadrature values of k3 are cached when
    they fit in `cache_bytes`; otherwise they are recomputed per chunk at every call (one more matrix product)."""

    def __init__(self, ops, q, K3, cache_bytes=None):
        self.o, self.xp = ops, ops.xp
        Psi, Wpsi = ops.b.psp_operator(q)
        self.Psi, self.Wpsi = ops.asarray(Psi), ops.asarray(Wpsi)
        self.K3 = ops.asarray(K3)
        Q, N = Psi.shape[0], K3.shape[0]
        self.Q = Q
        cache_bytes = ops.cache_bytes * 3 if cache_bytes is None else cache_bytes
        self.cached = N * Q * ops.dtype.itemsize <= cache_bytes
        if self.cached:                                    # (absent otherwise: parallel.NodeParallel restricts k3q)
            self.k3q = self.K3 @ self.Psi.T
        self.chunk = ops.chunk_nodes(Q * ops.dtype.itemsize * 8)
        self.label = f"PSP (q={q})"
        self.iters = []

    def __call__(self, h, k3=None):
        xp = self.xp
        out = xp.empty_like(h)
        K3 = self.K3 if k3 is None else k3
        for i in range(0, h.shape[0], self.chunk):
            j = min(h.shape[0], i + self.chunk)
            hq = h[i:j] @ self.Psi.T
            k3q = self.k3q[i:j] if self.cached else K3[i:j] @ self.Psi.T
            out[i:j] = cubic_root(xp, k3q, 1.0, hq) @ self.Wpsi
        return out


class MLPX:
    """Forward pass of mlp.MLP / mlp.FourierMLP (tanh hidden layers, linear output) on an array module."""

    def __init__(self, net, ops):
        xp, dt = ops.xp, ops.dtype
        inner = net.inner if hasattr(net, "inner") else net
        self.B2pi = ops.asarray(net.B2pi) if hasattr(net, "B2pi") else None
        self.W = [ops.asarray(W) for W in inner.W]
        self.b = [ops.asarray(b) for b in inner.b]
        self.xp = xp
        self.n_params = int(sum(W.size + b.size for W, b in zip(inner.W, inner.b)))

    def __call__(self, x):
        xp = self.xp
        if self.B2pi is not None:
            u = x @ self.B2pi
            x = xp.concatenate([x, xp.sin(u), xp.cos(u)], 1)
        a = x
        L = len(self.W)
        for l in range(L - 1):
            a = xp.tanh(a @ self.W[l] + self.b[l])
        return a @ self.W[-1] + self.b[-1]


class FieldPINNX:
    """implicit_field.FieldPINN: T = h + s_y [N(h/s_h, k3/s_k) - N(0, k3/s_k)]; the two forward passes run as one batch."""

    label = "PINN closure"

    def __init__(self, ops, lib_net, sh, sk, sy):
        self.o, self.xp, self.net = ops, ops.xp, MLPX(lib_net, ops)
        self.sh, self.sk, self.sy = sh, sk, sy
        self.iters = []

    def __call__(self, h, k3):
        xp = self.xp
        n = h.shape[0]
        ks = k3 / self.sk
        z = xp.concatenate([xp.concatenate([h / self.sh, ks], 1), xp.concatenate([xp.zeros_like(h), ks], 1)], 0)
        y = self.net(z)
        return h + self.sy * (y[:n] - y[n:])


def load_field_pinn(ops, fs_lib, path):
    """Library FieldPINN read from a model file, then moved to the array module."""
    from implicit_field import FieldPINN
    pn = FieldPINN.from_file(fs_lib, path)
    return FieldPINNX(ops, pn.net, pn.sh, pn.sk, pn.sy)


# =============================================================================
# Ignition: explicit exponential closure
# =============================================================================
class _ExpBase:
    def __init__(self, ops):
        self.o, self.xp, self.P = ops, ops.xp, ops.P
        self.iters = []

    @staticmethod
    def split(u):
        u0 = u[:, 0].copy()
        x = u.copy()
        x[:, 0] = 0.0
        return u0, x

    def taylor_terms(self, x, K):
        """[g_0, ..., g_{K+1}] for the exponential (closures.Nonlinearity.taylor_terms with A_n v = n * v)."""
        g = [self.xp.tile(self.o.e0, (x.shape[0], 1))]
        for m_ in range(K + 1):
            g.append(self.o.sprod(x, g[-1]) / (m_ + 1))
        return g


class ExactGalerkinX(_ExpBase):
    """f(M_u) e_0 by symmetric eigendecomposition, in chunks of nodes."""
    label = "Galerkin (eig)"

    def __init__(self, ops):
        super().__init__(ops)
        self.chunk = ops.chunk_nodes(ops.P * ops.P * ops.dtype.itemsize * 4)

    def __call__(self, u):
        xp, o = self.xp, self.o
        out = xp.empty_like(u)
        for i in range(0, u.shape[0], self.chunk):
            lam, V = o.eigh(o.mult_matrix(u[i:i + self.chunk]))
            out[i:i + self.chunk] = xp.einsum('bkq,bq->bk', V, xp.exp(lam) * V[:, 0, :])
        return out


class LanczosX(_ExpBase):
    """closures.Lanczos (m steps, full reorthogonalisation) for the exponential."""

    def __init__(self, ops, m):
        super().__init__(ops)
        self.m = m
        self.label = f"Lanczos (m={m})"

    def __call__(self, u):
        xp, o, m = self.xp, self.o, self.m
        Bsz, P = u.shape
        dt = u.dtype
        Q = xp.zeros((m, Bsz, P), dtype=dt)
        alpha = xp.zeros((Bsz, m), dtype=dt)
        beta = xp.zeros((Bsz, m), dtype=dt)
        Q[0, :, 0] = 1.0
        alive = xp.ones(Bsz, dtype=bool)
        for j in range(m):
            w = o.sprod(u, Q[j])
            alpha[:, j] = xp.where(alive, xp.sum(w * Q[j], axis=1), u[:, 0])
            w -= alpha[:, j, None] * Q[j]
            if j > 0:
                w -= beta[:, j - 1, None] * Q[j - 1]
            for i in range(j + 1):
                w -= xp.sum(w * Q[i], axis=1)[:, None] * Q[i]
            if j < m - 1:
                nb = xp.linalg.norm(w, axis=1)
                ok = nb > 1e-12 * (1 + xp.abs(alpha[:, j]))
                alive &= ok
                beta[:, j] = xp.where(alive, nb, 0.0)
                Q[j + 1] = xp.where(alive[:, None], w / xp.maximum(nb, 1e-300)[:, None], 0.0)
        T = xp.zeros((Bsz, m, m), dtype=dt)
        idx = xp.arange(m)
        T[:, idx, idx] = alpha
        T[:, idx[:-1], idx[1:]] = beta[:, :-1]
        T[:, idx[1:], idx[:-1]] = beta[:, :-1]
        th, V = o.eigh(T)
        v0 = V[:, 0, :]
        fv = xp.where(xp.abs(v0) > 1e-14, xp.exp(th) * v0, 0.0)
        y = xp.einsum('bkq,bq->bk', V, fv)
        return xp.einsum('kbp,bk->bp', Q, y)


class TaylorX(_ExpBase):
    def __init__(self, ops, K):
        super().__init__(ops)
        self.K = K
        self.label = f"Taylor (K={K})"

    def __call__(self, u):
        u0, x = self.split(u)
        G = sum(self.taylor_terms(x, self.K - 1)[:self.K + 1])
        return self.xp.exp(u0)[:, None] * G


class PSPX(_ExpBase):
    """Pseudo-spectral projection of exp(u), chunked over nodes."""

    def __init__(self, ops, q):
        super().__init__(ops)
        Psi, Wpsi = ops.b.psp_operator(q)
        self.Psi, self.Wpsi = ops.asarray(Psi), ops.asarray(Wpsi)
        self.chunk = ops.chunk_nodes(Psi.shape[0] * ops.dtype.itemsize * 3)
        self.label = f"PSP (q={q})"

    def __call__(self, u):
        out = self.xp.empty_like(u)
        for i in range(0, u.shape[0], self.chunk):
            out[i:i + self.chunk] = self.xp.exp(u[i:i + self.chunk] @ self.Psi.T) @ self.Wpsi
        return out


class PINNClosureX(_ExpBase):
    """closures.PINNClosure for the exponential: G = sum_{m<=K} g_m + r^{K+1} N_theta(r, n)."""
    label = "PINN"

    def __init__(self, ops, lib):
        super().__init__(ops)
        self.K, self.r_max, self.n_scale, self.y_scale = lib.K, lib.r_max, lib.n_scale, lib.y_scale
        self.net = MLPX(lib.net, ops)

    def __call__(self, u):
        xp = self.xp
        u0, x = self.split(u)
        r = xp.linalg.norm(x, axis=1)
        n = x / xp.maximum(r, 1e-300)[:, None]
        z = xp.concatenate([(r / self.r_max)[:, None], self.n_scale * n[:, 1:]], 1)
        G = sum(self.taylor_terms(x, self.K - 1)[:self.K + 1]) + (r ** (self.K + 1))[:, None] * (self.y_scale * self.net(z))
        return xp.exp(u0)[:, None] * G


# =============================================================================
# Method tables (names as in experiments/e8_enthalpy.py and e6_ignition_highd.py)
# =============================================================================
HEAT_SOLVERS = {"newton_w8": ("newton", "warm", 1e-8), "nk_w8": ("nk", "warm", 1e-8),
                "chord_w8": ("chord", "warm", 1e-8), "chord_w10": ("chord", "warm", 1e-10)}
HEAT_ACTIVE = {"chord_w8_as": "chord_w8", "chord_w10_as": "chord_w10", "nk_w8_as": "nk_w8"}   # active-set variants
HEAT_FIXED = {"chordfix1": 1, "chordfix2": 2}                                                   # fixed-step controls
HEAT_METHODS = (["chord_w8", "chord_w10", "nk_w8", "newton_w8", "chord_w8_as", "chord_w10_as", "nk_w8_as",
                 "chordfix1", "chordfix2", "psp_s3", "psp_s4", "psp", "pinn", "pinn_nk1", "pinn_nk2"])
IGN_METHODS = ["eig", "lanczos", "taylor4", "psp_s3", "psp_s4", "psp", "pinn"]


def build_heat(method, ops, K3, model_path, lib_fs, gmres="vec"):
    """Heat-conduction closure `method` on ops; K3: (N, P) coefficients of k3 at the nodes (NumPy);
    lib_fs: library CubicField (only its P is used, to rebuild the saved network)."""
    P_ORDER = 3
    if method in HEAT_SOLVERS or method in HEAT_ACTIVE:
        base = HEAT_ACTIVE.get(method, method)
        meth, init, tol = HEAT_SOLVERS[base]
        return FieldSolverX(ops, method=meth, init=init, tol=tol, m=10, active=method in HEAT_ACTIVE, gmres=gmres)
    if method in HEAT_FIXED:
        return FieldSolverX(ops, method="chord", init="warm", fixed_steps=HEAT_FIXED[method], m=10, gmres=gmres)
    if method == "psp":
        return FieldPSPX(ops, P_ORDER + 1, K3)
    if method.startswith("psp_s"):
        return FieldPSPX(ops, method[4:], K3)
    pn = load_field_pinn(ops, lib_fs, model_path)
    if method == "pinn":
        return pn
    if method == "pinn_nk1":
        return FieldSolverX(ops, method="nk", init=pn, fixed_steps=1, m=5, gmres=gmres)
    if method == "pinn_nk2":
        return FieldSolverX(ops, method="nk", init=pn, fixed_steps=2, m=5, gmres=gmres)
    raise ValueError(method)


def build_ignition(method, ops, model_path, nl_lib):
    P_ORDER = 3
    if method == "eig":
        return ExactGalerkinX(ops)
    if method == "lanczos":
        return LanczosX(ops, 10)
    if method == "taylor4":
        return TaylorX(ops, 4)
    if method == "psp":
        return PSPX(ops, P_ORDER + 1)
    if method.startswith("psp_s"):
        return PSPX(ops, method[4:])
    if method == "pinn":
        from closures import PINNClosure
        return PINNClosureX(ops, PINNClosure.load(nl_lib, model_path))
    raise ValueError(method)
