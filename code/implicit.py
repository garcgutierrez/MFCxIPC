"""
Closures for an implicitly defined, multivariate nonlinearity:

    y = Y(x; k)   such that   k3 y^3 + (k1 + k2) y - k1 x = 0          (pointwise)

with random fields x (state) and k1, k3 (parameters).  The intrusive (Galerkin) closure is the
solution of the projected algebraic system

    R(y; x) = k3 * (y * (y * y)) + k12 * y - k1 * x = 0 ,      (* = Galerkin product)

which is not a matrix function of a single multiplication matrix (M_x and M_k3 do not commute), so
the Krylov/Lanczos closure does not apply.  Classical options: Newton on R (dense P x P solves) or a
pseudo-spectral projection of the pointwise root.  The PINN closure is trained on ||R||^2 (label-free).
"""
import numpy as np
from scipy.optimize import minimize

from mlp import MLP, Adam, make_net, net_spec, spec_from_file, rad_indices


def cubic_root(a3, a1, b):
    """Real root of a3 y^3 + a1 y = b with a3, a1 > 0 (Cardano, unique real root)."""
    p = a1 / a3
    q = -b / a3
    disc = np.sqrt(q * q / 4 + p ** 3 / 27)
    return np.cbrt(-q / 2 + disc) + np.cbrt(-q / 2 - disc)


class CubicSystem:
    """Galerkin algebra of the implicit relation (parameters as PC vectors)."""

    def __init__(self, basis, k1, k2, k3):
        self.b = basis
        self.k1, self.k3 = k1, k3
        self.k12 = k1 + k2
        self.P = basis.P
        # linearised (small-x) Galerkin solution: y = A x with A = M_k12^{-1} M_k1
        self.A = np.linalg.solve(basis.mult_matrix(self.k12), basis.mult_matrix(self.k1))

    def residual(self, x, y):
        b = self.b
        yy = b.sprod(y, y)
        return b.sprod(self.k3, b.sprod(y, yy)) + b.sprod(self.k12, y) - b.sprod(self.k1, x)

    def jT_apply(self, y, w):
        """J(y)^T w with J v = k3*(v*(y*y) + 2 y*(y*v)) + k12*v."""
        b = self.b
        kw = b.sprod(self.k3, w)
        return b.sprod(b.sprod(y, y), kw) + 2 * b.sprod(y, b.sprod(y, kw)) + b.sprod(self.k12, w)

    def jacobian(self, y):
        b = self.b
        My = b.mult_matrix(y)
        Myy = b.mult_matrix(b.sprod(y, y))
        Mk3 = b.mult_matrix(np.broadcast_to(self.k3, y.shape))
        return Mk3 @ (Myy + 2 * My @ My) + b.mult_matrix(np.broadcast_to(self.k12, y.shape))

    def linear(self, x):
        return x @ self.A.T


class NewtonClosure:
    label = "Galerkin-Newton"

    def __init__(self, sysm, tol=1e-12, maxit=30, init=None, fixed_steps=None):
        """init: callable giving the initial guess (default: linearised closure);
        fixed_steps: if set, perform exactly this many Newton steps (no convergence test)."""
        self.s, self.tol, self.maxit = sysm, tol, maxit
        self.init, self.fixed = init, fixed_steps
        self.iters = []

    def __call__(self, x):
        x2 = np.atleast_2d(x)
        y = self.s.linear(x2) if self.init is None else self.init(x2)
        if self.fixed is not None:
            for _ in range(self.fixed):
                R = self.s.residual(x2, y)
                y = y - np.linalg.solve(self.s.jacobian(y), R[..., None])[..., 0]
            self.iters.append(self.fixed)
            return y.reshape(np.shape(x))
        ref = np.linalg.norm(self.s.b.sprod(self.s.k1, x2), axis=1) + 1e-300
        for it in range(self.maxit):
            R = self.s.residual(x2, y)
            if np.all(np.linalg.norm(R, axis=1) <= self.tol * ref):
                break
            y = y - np.linalg.solve(self.s.jacobian(y), R[..., None])[..., 0]
        self.iters.append(it)
        return y.reshape(np.shape(x))


class CardanoPSP:
    """Pseudo-spectral projection of the pointwise root (tensor Gauss-Hermite, q points per dim)."""

    def __init__(self, sysm, q):
        self.s = sysm
        self.Psi, self.Wpsi = sysm.b.psp_operator(q)
        self.k1q = sysm.k1 @ self.Psi.T
        self.k3q = sysm.k3 @ self.Psi.T
        self.k12q = sysm.k12 @ self.Psi.T
        self.label = f"PSP (q={q})"

    def __call__(self, x):
        x2 = np.atleast_2d(x)
        xq = x2 @ self.Psi.T
        yq = cubic_root(self.k3q, self.k12q, self.k1q * xq)
        return (yq @ self.Wpsi).reshape(np.shape(x))


class PINNImplicit:
    """y = A x + s_y [N(x/s_x) - N(0)],  trained on the Galerkin residual (no labels)."""

    label = "PINN closure"

    def __init__(self, sysm, width=64, depth=3, seed=0, fourier=None):
        self.s = sysm
        P = sysm.P
        self.net = make_net([P] + [width] * depth + [P], seed=seed, fourier=fourier)
        self.sx, self.sy = 1.0, 1.0
        self.history = []

    def _y(self, x, keep=False):
        z = x / self.sx
        y1, _, c1 = self.net.forward(z)
        y0, _, c0 = self.net.forward(np.zeros((1, x.shape[1])))
        y = self.s.linear(x) + self.sy * (y1 - y0)
        return (y, c1, c0) if keep else y

    def __call__(self, x):
        x2 = np.atleast_2d(x)
        return self._y(x2).reshape(np.shape(x))

    def loss_and_grad(self, x, w):
        y, c1, c0 = self._y(x, keep=True)
        R = self.s.residual(x, y)
        Bsz = x.shape[0]
        loss = np.sum(w[:, None] * R ** 2) / Bsz
        gy = self.s.jT_apply(y, 2 * w[:, None] * R / Bsz) * self.sy
        gW1, gb1 = self.net.backward(c1, gy)
        gW0, gb0 = self.net.backward(c0, -gy.sum(0, keepdims=True))
        g = MLP.flat_grads([a + b for a, b in zip(gW1, gW0)], [a + b for a, b in zip(gb1, gb0)])
        return loss, g

    def sample(self, anchors, n, rng, jitter=0.15):
        idx = rng.integers(0, len(anchors), n)
        xa = anchors[idx]
        sc = np.linalg.norm(xa, axis=1, keepdims=True)
        x = xa * (1 + jitter * rng.normal(size=xa.shape)) + 0.3 * jitter * sc * rng.normal(size=xa.shape) / np.sqrt(self.s.P)
        return x

    def train(self, anchors, n_pool=20000, batch=256, iters=5000, lr=2e-3, lbfgs_iters=1500,
              lbfgs_batch=2048, seed=0, verbose=True, validate=None, adaptive=None):
        """adaptive: None (fixed pool, the paper's baseline) or a dict for residual-based adaptive sampling (RAD),
        as in FieldPINN.train."""
        rng = np.random.default_rng(seed)
        X = self.sample(anchors, n_pool, rng)
        self.sx = float(np.sqrt(np.mean(X ** 2)) + 1e-12)
        ylin = self.s.linear(X)
        self.sy = float(np.sqrt(np.mean(ylin ** 2)) * 0.1 + 1e-12)
        ref = np.linalg.norm(self.s.b.sprod(self.s.k1, X), axis=1) ** 2
        rm = ref.mean()
        W = 1.0 / (ref + 1e-3 * rm)
        wm = W.mean()
        W /= wm
        wfun = lambda x: 1.0 / (np.linalg.norm(self.s.b.sprod(self.s.k1, x), axis=1) ** 2 + 1e-3 * rm) / wm
        ad = None if adaptive is None else dict(dict(every=max(1, iters // 8), k=1.0, c=1.0, n_cand=2 * n_pool),
                                                **adaptive)

        def rad():
            xc = self.sample(anchors, ad["n_cand"], rng)
            wc = wfun(xc)
            eps = np.concatenate([np.sqrt(wc[j:j + 4096]) * np.linalg.norm(
                self.s.residual(xc[j:j + 4096], self._y(xc[j:j + 4096])), axis=1) for j in range(0, len(xc), 4096)])
            sel = rad_indices(eps, n_pool, rng, ad["k"], ad["c"])
            return xc[sel], wc[sel]
        theta = self.net.get_flat()
        opt = Adam(theta.size, lr=lr)
        for it in range(iters):
            opt.lr = lr * 0.5 * (1 + np.cos(np.pi * it / iters)) + 1e-5
            if ad is not None and it > 0 and it % ad["every"] == 0:
                self.net.set_flat(theta)
                X, W = rad()
            idx = rng.integers(0, n_pool, batch)
            self.net.set_flat(theta)
            loss, g = self.loss_and_grad(X[idx], W[idx])
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
                X, W = rad()
            idx = rng.choice(n_pool, lbfgs_batch, replace=False)
            Xb, Wb = X[idx], W[idx]

            def fg(th):
                self.net.set_flat(th)
                return self.loss_and_grad(Xb, Wb)
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

    def save(self, path):
        np.savez(path, theta=self.net.get_flat(), sizes=np.array(self.net.sizes), sx=self.sx, sy=self.sy,
                 fourier=net_spec(self.net))

    def load_weights(self, path):
        z = np.load(path)
        self.net.set_flat(z["theta"])
        self.sx, self.sy = float(z["sx"]), float(z["sy"])
        return self

    @classmethod
    def from_file(cls, sysm, path):
        """Rebuild a saved network (size and Fourier embedding read from the file)."""
        z = np.load(path)
        sizes = z["sizes"]
        return cls(sysm, width=int(sizes[1]), depth=len(sizes) - 2, fourier=spec_from_file(z)).load_weights(path)


def _jv(s, y, v):
    """Matrix-free Jacobian-vector product J(y) v (Galerkin products only)."""
    b = s.b
    return b.sprod(s.k3, b.sprod(v, b.sprod(y, y)) + 2 * b.sprod(y, b.sprod(y, v))) + b.sprod(s.k12, v)


def gmres_batched(matvec, rhs, m):
    """m steps of GMRES (zero initial guess, modified Gram-Schmidt), vectorised over the batch."""
    Bsz, P = rhs.shape
    beta = np.linalg.norm(rhs, axis=1)
    V = np.zeros((m + 1, Bsz, P))
    H = np.zeros((Bsz, m + 1, m))
    V[0] = rhs / np.maximum(beta, 1e-300)[:, None]
    k_used = m
    for j in range(m):
        w = matvec(V[j])
        for i in range(j + 1):
            H[:, i, j] = np.sum(w * V[i], axis=1)
            w = w - H[:, i, j, None] * V[i]
        H[:, j + 1, j] = np.linalg.norm(w, axis=1)
        V[j + 1] = w / np.maximum(H[:, j + 1, j], 1e-300)[:, None]
        if np.all(H[:, j + 1, j] < 1e-14 * beta):
            k_used = j + 1
            break
    e1 = np.zeros((Bsz, k_used + 1)); e1[:, 0] = beta
    z = np.stack([np.linalg.lstsq(H[b, :k_used + 1, :k_used], e1[b], rcond=None)[0] for b in range(Bsz)])
    return np.einsum('kbp,bk->bp', V[:k_used], z)


class NewtonKrylovClosure:
    """Inexact Newton with matrix-free GMRES(m) inner solves.  With init=PINN and fixed_steps=1 it is the
    'PINN + one Newton-Krylov step' closure; with the linearised guess and a tolerance it is the classical
    matrix-free Newton-Krylov closure."""

    def __init__(self, sysm, m=10, init=None, fixed_steps=None, tol=1e-10, maxit=30):
        self.s, self.m, self.init, self.fixed, self.tol, self.maxit = sysm, m, init, fixed_steps, tol, maxit
        self.iters = []
        self.label = "Newton-Krylov" if init is None else "PINN + Newton-Krylov"

    def __call__(self, x):
        s = self.s
        x2 = np.atleast_2d(x)
        y = s.linear(x2) if self.init is None else self.init(x2)
        ref = np.linalg.norm(s.b.sprod(s.k1, x2), axis=1) + 1e-300
        nsteps = self.fixed if self.fixed is not None else self.maxit
        for it in range(nsteps):
            R = s.residual(x2, y)
            if self.fixed is None and np.all(np.linalg.norm(R, axis=1) <= self.tol * ref):
                break
            y = y - gmres_batched(lambda v: _jv(s, y, v), R, self.m)
        else:
            it = nsteps
        self.iters.append(it)
        return y.reshape(np.shape(x))


# =============================================================================
# Stronger classical baselines (revision 2026-10): warm starts, matched tolerances, Jacobian reuse
# =============================================================================
class CubicSolver:
    """Galerkin solve of R(y; x) = k3*(y*(y*y)) + k12*y - k1*x = 0 with the options of
    implicit_field.FieldSolver: method 'newton' | 'nk' (one GMRES cycle of m Arnoldi steps) | 'chord'
    (inverse Jacobian frozen across iterations and calls, refreshed on poor contraction); init 'linear'
    (y = A x) | 'warm' (previous solution) | callable; tol on ||R|| <= tol ||k1*x||."""

    def __init__(self, sysm, method="nk", init="linear", tol=1e-10, maxit=40, m=10, fixed_steps=None,
                 chord_max=6, chord_rate=0.5, record=False):
        self.s, self.method, self.init, self.tol, self.maxit, self.m = sysm, method, init, tol, maxit, m
        self.fixed, self.chord_max, self.chord_rate, self.record = fixed_steps, chord_max, chord_rate, record
        self.iters, self.final_res, self.init_res = [], [], []
        self.n = dict(calls=0, jv=0, jac=0, res=0)
        b = sysm.b
        self._Mk3, self._Mk12 = b.mult_matrix(sysm.k3), b.mult_matrix(sysm.k12)
        self._prev, self._Jinv = None, None
        self.label = f"{method} ({init})"

    def _jac(self, y):
        b = self.s.b
        My = b.mult_matrix(y)
        self.n["jac"] += 1
        return self._Mk3 @ (b.mult_matrix(b.sprod(y, y)) + 2 * My @ My) + self._Mk12

    def __call__(self, x):
        s = self.s
        x2 = np.atleast_2d(x)
        self.n["calls"] += 1
        if callable(self.init):
            y = self.init(x2)
        elif self.init == "warm" and self._prev is not None and self._prev.shape == x2.shape:
            y = self._prev.copy()
        else:
            y = s.linear(x2)
        ref = np.linalg.norm(s.b.sprod(s.k1, x2), axis=1) + 1e-300
        nmax = self.fixed if self.fixed is not None else self.maxit
        rn_prev, it = None, 0
        refresh = self._Jinv is None or self._Jinv.shape[0] != x2.shape[0]
        for it in range(nmax + 1):
            R = s.residual(x2, y)
            self.n["res"] += 1
            rn = np.linalg.norm(R, axis=1) / ref
            if it == 0 and self.record:
                self.init_res.append(float(rn.max()))
            if self.fixed is None and np.all(rn <= self.tol):
                break
            if it == nmax:
                break
            if self.method == "newton":
                y = y - np.linalg.solve(self._jac(y), R[..., None])[..., 0]
            elif self.method == "nk":
                y = y - gmres_batched(lambda v: _jv(s, y, v), R, self.m)
                self.n["jv"] += self.m
            elif self.method == "chord":
                if (refresh or it >= self.chord_max
                        or (rn_prev is not None and np.max(rn / np.maximum(rn_prev, 1e-300)) > self.chord_rate)):
                    self._Jinv = np.linalg.inv(self._jac(y))
                    refresh = False
                y = y - np.einsum('bij,bj->bi', self._Jinv, R)
            else:
                raise ValueError(self.method)
            rn_prev = rn
        self.iters.append(it)
        if self.record:
            if self.fixed is not None:
                rn = np.linalg.norm(s.residual(x2, y), axis=1) / ref
            self.final_res.append(float(rn.max()))
        self._prev = y
        return y.reshape(np.shape(x))
