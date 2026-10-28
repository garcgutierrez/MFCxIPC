"""
Implicit constitutive law evaluated at every node of a PDE:

    temperature T from enthalpy h:      k3 T^3 + T - h = 0       (pointwise, k3 = k3(x, xi) > 0)

Intrusive closure: solve the Galerkin system  R(T; h, k3) = k3 * (T * (T * T)) + T - h = 0  for every node,
where h and k3 are both random fields (two non-commuting multiplication matrices) and k3 differs from node
to node.  All classes act on batches: h, k3 of shape (B, P).
"""
import numpy as np
from scipy.optimize import minimize

from mlp import MLP, Adam, make_net, net_spec, spec_from_file, rad_indices
from implicit import cubic_root, gmres_batched


class CubicField:
    def __init__(self, basis):
        self.b = basis
        self.P = basis.P

    def residual(self, h, k3, y):
        b = self.b
        return b.sprod(k3, b.sprod(y, b.sprod(y, y))) + y - h

    def jv(self, k3, y, v):
        b = self.b
        return b.sprod(k3, b.sprod(v, b.sprod(y, y)) + 2 * b.sprod(y, b.sprod(y, v))) + v

    def jtv(self, k3, y, w):
        b = self.b
        kw = b.sprod(k3, w)
        return b.sprod(b.sprod(y, y), kw) + 2 * b.sprod(y, b.sprod(y, kw)) + w

    def jacobian(self, k3, y):
        b = self.b
        My = b.mult_matrix(y)
        return b.mult_matrix(k3) @ (b.mult_matrix(b.sprod(y, y)) + 2 * My @ My) + np.eye(self.P)


class FieldNewton:
    label = "Newton"

    def __init__(self, fs, tol=1e-12, maxit=40, init=None, fixed_steps=None):
        self.fs, self.tol, self.maxit, self.init, self.fixed = fs, tol, maxit, init, fixed_steps
        self.iters = []

    def __call__(self, h, k3):
        y = h.copy() if self.init is None else self.init(h, k3)
        ref = np.linalg.norm(h, axis=1) + 1e-300
        n = self.fixed if self.fixed is not None else self.maxit
        it = 0
        for it in range(n):
            R = self.fs.residual(h, k3, y)
            if self.fixed is None and np.all(np.linalg.norm(R, axis=1) <= self.tol * ref):
                break
            y = y - np.linalg.solve(self.fs.jacobian(k3, y), R[..., None])[..., 0]
        else:
            it = n
        self.iters.append(it)
        return y


class FieldNewtonKrylov:
    label = "Newton-Krylov"

    def __init__(self, fs, m=10, tol=1e-10, maxit=40, init=None, fixed_steps=None):
        self.fs, self.m, self.tol, self.maxit, self.init, self.fixed = fs, m, tol, maxit, init, fixed_steps
        self.iters = []

    def __call__(self, h, k3):
        y = h.copy() if self.init is None else self.init(h, k3)
        ref = np.linalg.norm(h, axis=1) + 1e-300
        n = self.fixed if self.fixed is not None else self.maxit
        it = 0
        for it in range(n):
            R = self.fs.residual(h, k3, y)
            if self.fixed is None and np.all(np.linalg.norm(R, axis=1) <= self.tol * ref):
                break
            y = y - gmres_batched(lambda v: self.fs.jv(k3, y, v), R, self.m)
        else:
            it = n
        self.iters.append(it)
        return y


class FieldPSP:
    """Pseudo-spectral projection of the pointwise root; k3 at the nodes is cached per spatial node."""

    def __init__(self, fs, q, k3_nodes_fields):
        self.fs = fs
        self.Psi, self.Wpsi = fs.b.psp_operator(q)
        self.k3q = k3_nodes_fields @ self.Psi.T          # (nx, Q)
        self.label = f"PSP (q={q})"

    def __call__(self, h, k3=None):
        hq = h @ self.Psi.T
        return cubic_root(self.k3q, 1.0, hq) @ self.Wpsi


class FieldPINN:
    """T = h + s_y [N(h/s_h, k3/s_k) - N(0, k3/s_k)], trained on the Galerkin residual (no labels)."""

    label = "PINN closure"

    def __init__(self, fs, width=128, depth=3, seed=0, fourier=None):
        self.fs = fs
        P = fs.P
        self.net = make_net([2 * P] + [width] * depth + [P], seed=seed, fourier=fourier)
        self.sh = self.sk = self.sy = 1.0
        self.history = []

    def _forward(self, h, k3):
        z1 = np.concatenate([h / self.sh, k3 / self.sk], 1)
        z0 = np.concatenate([np.zeros_like(h), k3 / self.sk], 1)
        y1, _, c1 = self.net.forward(z1)
        y0, _, c0 = self.net.forward(z0)
        return h + self.sy * (y1 - y0), c1, c0

    def __call__(self, h, k3):
        return self._forward(h, k3)[0]

    def loss_and_grad(self, h, k3, w):
        y, c1, c0 = self._forward(h, k3)
        R = self.fs.residual(h, k3, y)
        Bsz = h.shape[0]
        loss = np.sum(w[:, None] * R ** 2) / Bsz
        gy = self.fs.jtv(k3, y, 2 * w[:, None] * R / Bsz) * self.sy
        gW1, gb1 = self.net.backward(c1, gy)
        gW0, gb0 = self.net.backward(c0, -gy)
        return loss, MLP.flat_grads([a + b for a, b in zip(gW1, gW0)], [a + b for a, b in zip(gb1, gb0)])

    @staticmethod
    def sample(H, K, n, rng, jitter=0.1):
        idx = rng.integers(0, len(H), n)
        h, k = H[idx], K[idx]
        sc = np.linalg.norm(h, axis=1, keepdims=True)
        h = h * (1 + jitter * rng.normal(size=h.shape)) + 0.3 * jitter * sc * rng.normal(size=h.shape) / np.sqrt(h.shape[1])
        return h, k

    def train(self, H, K, n_pool=20000, batch=256, iters=6000, lr=2e-3, lbfgs_iters=2000, lbfgs_batch=2048,
              seed=0, verbose=True, validate=None, adaptive=None):
        """adaptive: None (fixed pool, the paper's baseline) or a dict for residual-based adaptive sampling (RAD):
        every `every` Adam iterations the pool is redrawn from `n_cand` fresh samples around the anchors with
        probability ~ eps^k/mean(eps^k) + c (eps: weighted residual norm)."""
        rng = np.random.default_rng(seed)
        Hs, Ks = self.sample(H, K, n_pool, rng)
        self.sh = float(np.sqrt(np.mean(Hs ** 2)) + 1e-12)
        self.sk = float(np.sqrt(np.mean(Ks ** 2)) + 1e-12)
        self.sy = 0.3 * self.sh
        h2m = np.mean(np.sum(Hs ** 2, 1))
        W = 1.0 / (np.sum(Hs ** 2, 1) + 1e-3 * h2m)
        wm = W.mean()
        W /= wm
        wfun = lambda h: 1.0 / (np.sum(h ** 2, 1) + 1e-3 * h2m) / wm
        ad = None if adaptive is None else dict(dict(every=max(1, iters // 8), k=1.0, c=1.0, n_cand=2 * n_pool),
                                                **adaptive)
        theta = self.net.get_flat()
        opt = Adam(theta.size, lr=lr)
        for it in range(iters):
            opt.lr = lr * 0.5 * (1 + np.cos(np.pi * it / iters)) + 1e-5
            if ad is not None and it > 0 and it % ad["every"] == 0:
                self.net.set_flat(theta)
                Hs, Ks, W = self._rad(H, K, n_pool, ad, wfun, rng)
            idx = rng.integers(0, n_pool, batch)
            self.net.set_flat(theta)
            loss, g = self.loss_and_grad(Hs[idx], Ks[idx], W[idx])
            theta = opt.step(theta, g)
            if it % 500 == 0 or it == iters - 1:
                self.net.set_flat(theta)
                ve = validate(self) if validate else np.nan
                self.history.append(("adam", it, loss, ve))
                if verbose:
                    print(f"  adam {it:5d} loss {loss:.3e} val {ve:.3e}", flush=True)
        self.net.set_flat(theta)
        if lbfgs_iters:
            if ad is not None:
                Hs, Ks, W = self._rad(H, K, n_pool, ad, wfun, rng)
            idx = rng.choice(n_pool, lbfgs_batch, replace=False)

            def fg(th):
                self.net.set_flat(th)
                return self.loss_and_grad(Hs[idx], Ks[idx], W[idx])
            cnt = [0]

            def cb(th):
                cnt[0] += 1
                if cnt[0] % 250 == 0:
                    self.net.set_flat(th)
                    ve = validate(self) if validate else np.nan
                    self.history.append(("lbfgs", cnt[0], fg(th)[0], ve))
                    if verbose:
                        print(f"  lbfgs {cnt[0]:5d} val {ve:.3e}", flush=True)
            res = minimize(fg, theta, jac=True, method="L-BFGS-B", callback=cb,
                           options=dict(maxiter=lbfgs_iters, maxcor=50, gtol=1e-14, ftol=1e-16))
            self.net.set_flat(res.x)
        self.final_val = validate(self) if validate else np.nan
        return self

    def _rad(self, H, K, n_pool, ad, wfun, rng):
        hc, kc = self.sample(H, K, ad["n_cand"], rng)
        wc = wfun(hc)
        eps = np.concatenate([np.sqrt(wc[j:j + 2048]) * np.linalg.norm(
            self.fs.residual(hc[j:j + 2048], kc[j:j + 2048], self(hc[j:j + 2048], kc[j:j + 2048])), axis=1)
            for j in range(0, len(hc), 2048)])
        sel = rad_indices(eps, n_pool, rng, ad["k"], ad["c"])
        return hc[sel], kc[sel], wc[sel]

    def save(self, path):
        np.savez(path, theta=self.net.get_flat(), sizes=np.array(self.net.sizes), sh=self.sh, sk=self.sk, sy=self.sy,
                 fourier=net_spec(self.net))

    def load_weights(self, path):
        z = np.load(path)
        self.net.set_flat(z["theta"])
        self.sh, self.sk, self.sy = float(z["sh"]), float(z["sk"]), float(z["sy"])
        return self

    @classmethod
    def from_file(cls, fs, path):
        """Rebuild a saved network (size and Fourier embedding read from the file)."""
        z = np.load(path)
        sizes = z["sizes"]
        return cls(fs, width=int(sizes[1]), depth=len(sizes) - 2, fourier=spec_from_file(z)).load_weights(path)


# =============================================================================
# Stronger classical baselines (revision 2026-10): warm starts, matched tolerances, Jacobian reuse
# =============================================================================
class FieldSolver:
    """Galerkin solve of R(T; h, k3) = k3*(T*(T*T)) + T - h = 0 at every node, with options

    method : 'newton' (dense Jacobian, fresh at every iteration), 'nk' (matrix-free inexact Newton: one
             GMRES cycle of m Arnoldi steps, zero initial guess, no restart) or 'chord' (simplified
             Newton: the inverse Jacobian is frozen across iterations *and calls*, and refreshed only when
             the iteration contracts poorly or needs more than `chord_max` iterations)
    init   : 'linear' (T = h, the linearised law), 'warm' (solution of the previous call, which is a
             valid guess whatever stage or rejected step the integrator evaluates next) or a callable
             (e.g. a trained network)
    tol    : stopping criterion ||R||_2 <= tol ||h||_2 at every node; fixed_steps disables the test.
    The multiplication matrix of k3 is constant per node and is assembled once.
    Counters: calls, iters (per call, max over nodes), jv (Jacobian-vector products per node),
    jac (Jacobian assemblies/inversions), res (residual evaluations); record=True also stores the final
    relative residual of every call (max over nodes; costs one extra residual for fixed-step variants)."""

    def __init__(self, fs, method="nk", init="linear", tol=1e-10, maxit=40, m=10, fixed_steps=None,
                 chord_max=6, chord_rate=0.5, record=False):
        self.fs, self.method, self.init, self.tol, self.maxit, self.m = fs, method, init, tol, maxit, m
        self.fixed, self.chord_max, self.chord_rate, self.record = fixed_steps, chord_max, chord_rate, record
        self.iters, self.final_res, self.init_res = [], [], []
        self.n = dict(calls=0, jv=0, jac=0, res=0)
        self._prev = None
        self._k3 = None
        self._Mk3 = None
        self._Jinv = None
        self.label = f"{method} ({init})"

    def _mk3(self, k3):
        if self._k3 is not k3:
            self._k3, self._Mk3 = k3, self.fs.b.mult_matrix(k3)
        return self._Mk3

    def _jac(self, k3, y):
        b = self.fs.b
        My = b.mult_matrix(y)
        self.n["jac"] += 1
        return self._mk3(k3) @ (b.mult_matrix(b.sprod(y, y)) + 2 * My @ My) + np.eye(self.fs.P)

    def _guess(self, h, k3):
        if callable(self.init):
            return self.init(h, k3)
        if self.init == "warm" and self._prev is not None and self._prev.shape == h.shape:
            return self._prev.copy()
        return h.copy()

    def __call__(self, h, k3):
        fs = self.fs
        self.n["calls"] += 1
        y = self._guess(h, k3)
        ref = np.linalg.norm(h, axis=1) + 1e-300
        nmax = self.fixed if self.fixed is not None else self.maxit
        rn_prev, it = None, 0
        refresh = self._Jinv is None or self._Jinv.shape[0] != h.shape[0]
        for it in range(nmax + 1):
            R = fs.residual(h, k3, y)
            self.n["res"] += 1
            rn = np.linalg.norm(R, axis=1) / ref
            if it == 0 and self.record:
                self.init_res.append(float(rn.max()))
            if self.fixed is None and np.all(rn <= self.tol):
                break
            if it == nmax:
                break
            if self.method == "newton":
                y = y - np.linalg.solve(self._jac(k3, y), R[..., None])[..., 0]
            elif self.method == "nk":
                y = y - gmres_batched(lambda v: fs.jv(k3, y, v), R, self.m)
                self.n["jv"] += self.m
            elif self.method == "chord":
                if (refresh or it >= self.chord_max
                        or (rn_prev is not None and np.max(rn / np.maximum(rn_prev, 1e-300)) > self.chord_rate)):
                    self._Jinv = np.linalg.inv(self._jac(k3, y))
                    refresh = False
                y = y - np.einsum('bij,bj->bi', self._Jinv, R)
            else:
                raise ValueError(self.method)
            rn_prev = rn
        self.iters.append(it)
        if self.record:
            if self.fixed is not None:
                R = fs.residual(h, k3, y)
                rn = np.linalg.norm(R, axis=1) / ref
            self.final_res.append(float(rn.max()))
        self._prev = y
        return y
