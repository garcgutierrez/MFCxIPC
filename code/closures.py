"""
Closures for non-polynomial nonlinearities in intrusive (stochastic Galerkin) PC.

Given the PC coefficients u of a random input, a closure returns PC coefficients of f(u).

  ExactGalerkin : Galerkin functional calculus  f(M_u) e_0   (eigendecomposition, O(P^3))
  PSP           : pseudo-spectral projection with tensor Gauss-Hermite quadrature
  Taylor        : truncated series of the ray ODE (Galerkin algebra only)
  PINN          : Taylor_K + r^{K+1} N_theta(r, n), N_theta trained on the ray-ODE residual

Ray ODE.  Write u = u_0 e_0 + x (exp, sin/cos) or u = u_0 (e_0 + x) (powers), r = ||x||,
n = x / r.  Along the ray x(s) = s n the Galerkin closure G(s) = f(M_{s n}) e_0 obeys
      B(s) dG/ds = A_n G,      G(0) = g_0,
  exp      : B = I,          A_n v = n*v                     (f' = f)
  sin/cos  : B = I,          A_n [S;C] = [n*C; -n*S]          (S' = C, C' = -S)
  power a  : B = I + s M_n,  A_n v = a n*v                   ((1+z) f' = a f)
where * is the Galerkin product.  The truncated series G = sum_m s^m g_m follows from
  g_{m+1} = (A_n g_m - c m M_n g_m) / (m+1),   c = 1 for powers and 0 otherwise.
"""
import numpy as np
from scipy.optimize import minimize

from mlp import MLP, Adam, make_net, net_spec, spec_from_file, rad_indices


# =============================================================================
# Nonlinearity specifications
# =============================================================================
class Nonlinearity:
    m = 1          # number of output fields
    c = 0.0        # B(s) = I + c s M_n

    def __init__(self, basis):
        self.basis = basis
        self.P = basis.P

    def g0(self):
        g = np.zeros(self.m * self.P)
        g[(self.m - 1) * self.P] = 1.0     # e_0 in the last block (C for sin/cos)
        return g

    def mul(self, n, v):
        """n*v blockwise for v of shape (B, m*P)."""
        if self.m == 1:
            return self.basis.sprod(n, v)
        P = self.P
        return np.concatenate([self.basis.sprod(n, v[:, k * P:(k + 1) * P]) for k in range(self.m)], 1)

    def B_apply(self, n, s, v):
        return v if self.c == 0 else v + self.c * s[:, None] * self.mul(n, v)

    BT_apply = B_apply          # M_n is symmetric

    def taylor_terms(self, x, K):
        """Returns [g_0, ..., g_{K+1}] for the (unnormalised) direction x, i.e. s^m g_m(n) at s=1."""
        Bsz = x.shape[0]
        g = [np.tile(self.g0(), (Bsz, 1))]
        for m_ in range(K + 1):
            nxt = self.A_apply(x, g[-1])
            if self.c != 0:
                nxt = nxt - self.c * m_ * self.mul(x, g[-1])
            g.append(nxt / (m_ + 1))
        return g


class Exp(Nonlinearity):
    name = "exp"

    def split(self, u):
        u0 = u[:, 0].copy()
        x = u.copy(); x[:, 0] = 0.0
        return u0, x

    def combine(self, u0, G):
        return np.exp(u0)[:, None] * G

    def A_apply(self, n, v):
        return self.mul(n, v)

    def AT_apply(self, n, v):
        return self.mul(n, v)

    @staticmethod
    def f(z):
        return [np.exp(z)]

    def f_normalised(self, lam):
        return [np.exp(lam)]


class SinCos(Nonlinearity):
    """Output blocks [sin; cos]."""
    name = "sincos"
    m = 2

    def split(self, u):
        u0 = u[:, 0].copy()
        x = u.copy(); x[:, 0] = 0.0
        return u0, x

    def combine(self, u0, G):
        P = self.P
        S, C = G[:, :P], G[:, P:]
        s0, c0 = np.sin(u0)[:, None], np.cos(u0)[:, None]
        return np.concatenate([s0 * C + c0 * S, c0 * C - s0 * S], 1)

    def A_apply(self, n, v):
        P = self.P
        return np.concatenate([self.basis.sprod(n, v[:, P:]), -self.basis.sprod(n, v[:, :P])], 1)

    def AT_apply(self, n, v):
        return -self.A_apply(n, v)

    @staticmethod
    def f(z):
        return [np.sin(z), np.cos(z)]

    def f_normalised(self, lam):
        return [np.sin(lam), np.cos(lam)]


class Power(Nonlinearity):
    """u^a for a random input with positive mean; normalised input x = u/u_0 - e_0."""
    c = 1.0

    def __init__(self, basis, a):
        super().__init__(basis)
        self.a = float(a)
        self.name = f"pow{a:g}"

    def split(self, u):
        u0 = u[:, 0].copy()
        x = u / u0[:, None]; x[:, 0] = 0.0
        return u0, x

    def combine(self, u0, G):
        return (u0 ** self.a)[:, None] * G

    def A_apply(self, n, v):
        return self.a * self.mul(n, v)

    def AT_apply(self, n, v):
        return self.a * self.mul(n, v)

    def f(self, z):
        return [np.abs(z) ** self.a]

    def f_normalised(self, lam):
        return [np.maximum(1.0 + lam, 1e-12) ** self.a]


# =============================================================================
# Reference closures
# =============================================================================
def _as_batch(u, P):
    u = np.asarray(u, float)
    return u.reshape(-1, P), u.shape


def _unbatch(out, shape, m, P):
    if m == 1:
        return out.reshape(shape)
    return out.reshape(shape[:-1] + (m * P,))


class ExactGalerkin:
    """f(M_u) e_0 by symmetric eigendecomposition (the classical intrusive closure)."""
    label = "Galerkin (eig)"

    def __init__(self, nl):
        self.nl, self.basis = nl, nl.basis

    def __call__(self, u):
        P = self.basis.P
        ub, sh = _as_batch(u, P)
        lam, V = np.linalg.eigh(self.basis.mult_matrix(ub))
        v0 = V[:, 0, :]                                   # (B, P): V^T e_0
        outs = [np.einsum('bkq,bq->bk', V, fl * v0) for fl in self.nl.f(lam)]
        return _unbatch(np.concatenate(outs, 1), sh, self.nl.m, P)

    def normalised(self, x):
        """Same, on normalised inputs (used to validate the PINN)."""
        lam, V = np.linalg.eigh(self.basis.mult_matrix(x))
        v0 = V[:, 0, :]
        return np.concatenate([np.einsum('bkq,bq->bk', V, fl * v0)
                               for fl in self.nl.f_normalised(lam)], 1)


class PSP:
    """Pseudo-spectral projection with q Gauss-Hermite points per dimension."""

    def __init__(self, nl, q):
        self.nl, self.basis = nl, nl.basis
        self.Psi, self.Wpsi = self.basis.psp_operator(q)
        self.label = f"PSP (q={q})"

    def __call__(self, u):
        P = self.basis.P
        ub, sh = _as_batch(u, P)
        z = ub @ self.Psi.T
        outs = [fz @ self.Wpsi for fz in self.nl.f(z)]
        return _unbatch(np.concatenate(outs, 1), sh, self.nl.m, P)


class Taylor:
    def __init__(self, nl, K):
        self.nl, self.K = nl, K
        self.label = f"Taylor (K={K})"

    def __call__(self, u):
        P = self.nl.P
        ub, sh = _as_batch(u, P)
        u0, x = self.nl.split(ub)
        G = sum(self.nl.taylor_terms(x, self.K - 1)[:self.K + 1])
        return _unbatch(self.nl.combine(u0, G), sh, self.nl.m, P)


# =============================================================================
# Physics-informed neural closure
# =============================================================================
def sample_directions(basis, n, rng, rho_lo=0.05, rho_hi=0.6):
    """Random unit directions with anisotropic, degree-decaying spectra (n_0 = 0)."""
    A = basis.alphas[1:]
    rho = rng.uniform(rho_lo, rho_hi, size=(n, basis.d))
    # randomly switch off some stochastic dimensions (anisotropic inputs)
    off = rng.random((n, basis.d)) < 0.25
    off[np.all(off, 1)] = False
    rho[off] = 0.0
    w = np.prod(rho[:, None, :] ** A[None, :, :], axis=2)
    z = rng.normal(size=w.shape) * w
    nrm = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.maximum(nrm, 1e-300)
    return np.concatenate([np.zeros((n, 1)), z], 1)


class PINNClosure:
    """
    G(r, n) = sum_{m<=K} r^m g_m(n) + r^{K+1} N_theta(r, n)

    Physics residual (ray ODE divided by r^K, see module docstring):
      rho = B(r)[(K+1) N + r dN/dr] - r A_n N - (K+1) g_{K+1}(n)
    which is linear in (N, dN/dr); no labels, no quadrature and no evaluation of f.
    """

    def __init__(self, nl, K=2, width=64, depth=3, r_max=1.0, R_eff=3.0, seed=0, n_scale=None, fourier=None):
        self.nl, self.basis, self.K = nl, nl.basis, K
        P = self.basis.P
        self.r_max, self.R_eff = r_max, R_eff
        self.n_scale = np.sqrt(P - 1) if n_scale is None else n_scale
        self.net = make_net([P] + [width] * depth + [nl.m * P], seed=seed, fourier=fourier)
        self.y_scale = 1.0
        self.label = "PINN"
        self.history = []

    # ---------------------------------------------------------------------
    def _inputs(self, r, n):
        z = np.concatenate([(r / self.r_max)[:, None], self.n_scale * n[:, 1:]], 1)
        zd = np.zeros_like(z); zd[:, 0] = 1.0 / self.r_max
        return z, zd

    def remainder(self, r, n, tangent=False):
        z, zd = self._inputs(r, n)
        y, yd, cache = self.net.forward(z, zd if tangent else None)
        if tangent:
            return self.y_scale * y, self.y_scale * yd, cache
        return self.y_scale * y

    def G(self, r, n):
        x = r[:, None] * n
        g = self.nl.taylor_terms(x, self.K - 1)[:self.K + 1]
        return sum(g) + (r ** (self.K + 1))[:, None] * self.remainder(r, n)

    def __call__(self, u):
        P = self.basis.P
        ub, sh = _as_batch(u, P)
        u0, x = self.nl.split(ub)
        r = np.linalg.norm(x, axis=1)
        n = x / np.maximum(r, 1e-300)[:, None]
        self.last_r_max = r.max() if r.size else 0.0
        return _unbatch(self.nl.combine(u0, self.G(r, n)), sh, self.nl.m, P)

    # ---------------------------------------------------------------------
    def residual(self, r, n, gK1):
        nl, K = self.nl, self.K
        N, Nd, _ = self.remainder(r, n, tangent=True)
        rn = r[:, None]
        return nl.B_apply(n, r, (K + 1) * N + rn * Nd) - rn * nl.A_apply(n, N) - (K + 1) * gK1

    def loss_and_grad(self, r, n, gK1, w=None):
        nl, K = self.nl, self.K
        N, Nd, cache = self.remainder(r, n, tangent=True)
        rn = r[:, None]
        rho = nl.B_apply(n, r, (K + 1) * N + rn * Nd) - rn * nl.A_apply(n, N) - (K + 1) * gK1
        Bsz = r.shape[0]
        w = np.ones(Bsz) if w is None else w
        loss = np.sum(w[:, None] * rho ** 2) / Bsz
        grho = 2.0 * w[:, None] * rho / Bsz
        BTg = nl.BT_apply(n, r, grho)
        gN = (K + 1) * BTg - rn * nl.AT_apply(n, grho)
        gNd = rn * BTg
        gW, gb = self.net.backward(cache, self.y_scale * gN, self.y_scale * gNd)
        return loss, MLP.flat_grads(gW, gb)

    def anchored_directions(self, anchors, n_dirs, rng, generic_frac=0.2, dir_kw=None, jitter=0.15):
        """Directions/radii in a neighbourhood of the inputs visited by a pilot run."""
        _, xa = self.nl.split(np.asarray(anchors, float))
        ra = np.linalg.norm(xa, axis=1)
        keep = ra > 1e-12
        xa, ra = xa[keep], ra[keep]
        n_gen = int(generic_frac * n_dirs)
        n_anc = n_dirs - n_gen
        idx = rng.integers(0, len(xa), n_anc)
        x = xa[idx]
        # multiplicative jitter on every coefficient + small isotropic noise
        x = x * (1 + jitter * rng.normal(size=x.shape)) \
            + 0.25 * jitter * ra[idx, None] * rng.normal(size=x.shape) / np.sqrt(self.basis.P)
        x[:, 0] = 0.0
        r = np.linalg.norm(x, axis=1)
        Da = x / r[:, None]
        Ra = 1.25 * r
        if self.nl.c != 0:
            Ra = np.minimum(Ra, self.r_limit(Da, cap=np.inf))
        if not getattr(self, "_warm", False):
            self.r_max = max(self.r_max, float(Ra.max()))
        Dg = sample_directions(self.basis, n_gen, rng, **(dir_kw or {}))
        Rg = self.r_limit(Dg)
        return np.concatenate([Da, Dg]), np.concatenate([Ra, Rg])

    def r_limit(self, n, cap=None):
        """Training radius along n: spectral radius of r M_n bounded by R_eff
        (and I + r M_n > 0 for powers), capped by r_max."""
        lam = np.concatenate([np.linalg.eigvalsh(self.basis.mult_matrix(n[i:i + 256]))[:, [0, -1]]
                              for i in range(0, n.shape[0], 256)])
        lim = self.R_eff / np.max(np.abs(lam), axis=1) if cap is None else np.full(len(n), np.inf)
        if self.nl.c != 0:
            lmin = lam[:, 0]
            lim = np.minimum(lim, np.where(lmin < 0, 0.9 / np.maximum(-lmin, 1e-12), np.inf))
        return np.minimum(lim, self.r_max if cap is None else cap)

    def train(self, n_dirs=20000, batch=256, iters=6000, lr=2e-3, lbfgs_iters=1500,
              lbfgs_batch=4096, seed=0, verbose=True, n_val=1000, dir_kw=None, anchors=None,
              generic_frac=0.2, val_on_anchors=True, warm_start=False, adaptive=None):
        """anchors: raw closure inputs u (N_a, P) visited by a pilot run; when given, training
        directions are drawn in a neighbourhood of them (labels are never used).
        adaptive: None (uniform radii, the setting of the paper's baseline closures) or a dict for residual-based
        adaptive sampling (RAD): every `every` Adam iterations the training set (n_dirs points) is redrawn from
        `n_cand` candidates (direction, uniform radius) with probability ~ eps^k/mean(eps^k) + c, eps being the
        weighted residual norm; the L-BFGS batch is drawn from the last training set."""
        rng = np.random.default_rng(seed)
        dir_kw = dir_kw or {}
        self._warm = warm_start
        if anchors is None:
            D = sample_directions(self.basis, n_dirs, rng, **dir_kw)
            Rlim = self.r_limit(D)
        else:
            D, Rlim = self.anchored_directions(anchors, n_dirs, rng, generic_frac, dir_kw)
        gK1_all = self.nl.taylor_terms(D, self.K)[self.K + 1]
        if not warm_start:
            self.y_scale = float(np.sqrt(np.mean(gK1_all ** 2)) + 1e-12)
        # validation set (labels from the exact Galerkin closure; never used for training)
        rngv = np.random.default_rng(seed + 991)
        if anchors is None or not val_on_anchors:
            Dv = sample_directions(self.basis, n_val, rngv, **dir_kw)
            rv = self.r_limit(Dv) * rngv.random(n_val)
        else:
            _, xa = self.nl.split(np.asarray(anchors)[rngv.integers(0, len(anchors), n_val)])
            rv = np.linalg.norm(xa, axis=1)
            Dv = xa / np.maximum(rv, 1e-300)[:, None]
        Gv = ExactGalerkin(self.nl).normalised(rv[:, None] * Dv)

        def val_err():
            Gp = self.G(rv, Dv)
            return np.median(np.linalg.norm(Gp - Gv, axis=1) / np.linalg.norm(Gv, axis=1))

        wts = 1.0 / (np.sum(gK1_all ** 2, axis=1) / self.y_scale ** 2 + 1e-2)
        wts /= wts.mean()

        ad = None if adaptive is None else dict(dict(every=max(1, iters // 8), k=1.0, c=1.0, n_cand=2 * n_dirs),
                                                **adaptive)
        pool = {}

        def resample():
            ic = rng.integers(0, n_dirs, ad["n_cand"])
            rc = Rlim[ic] * rng.random(ad["n_cand"])
            eps = np.concatenate([np.sqrt(wts[ic[j:j + 4096]]) * np.linalg.norm(
                self.residual(rc[j:j + 4096], D[ic[j:j + 4096]], gK1_all[ic[j:j + 4096]]), axis=1)
                for j in range(0, len(ic), 4096)])
            sel = rad_indices(eps, n_dirs, rng, ad["k"], ad["c"])
            pool["i"], pool["r"] = ic[sel], rc[sel]

        def draw(bs):
            if ad is not None and pool:
                j = rng.integers(0, n_dirs, bs)
                idx = pool["i"][j]
                return pool["r"][j], D[idx], gK1_all[idx], wts[idx]
            idx = rng.integers(0, n_dirs, bs)
            return Rlim[idx] * rng.random(bs), D[idx], gK1_all[idx], wts[idx]

        theta = self.net.get_flat()
        opt = Adam(theta.size, lr=lr)
        for it in range(iters):
            opt.lr = lr * 0.5 * (1 + np.cos(np.pi * it / iters)) + 1e-5
            if ad is not None and it > 0 and it % ad["every"] == 0:
                self.net.set_flat(theta)
                resample()
            r, n, g, w = draw(batch)
            self.net.set_flat(theta)
            loss, grad = self.loss_and_grad(r, n, g, w)
            theta = opt.step(theta, grad)
            if it % 500 == 0 or it == iters - 1:
                self.net.set_flat(theta)
                ve = val_err()
                self.history.append(("adam", it, loss, ve))
                if verbose:
                    print(f"  adam {it:6d}  loss {loss:.3e}  val rel.err {ve:.3e}", flush=True)
        self.net.set_flat(theta)
        if lbfgs_iters:
            if ad is not None:
                resample()
            r, n, g, w = draw(lbfgs_batch)

            def fg(th):
                self.net.set_flat(th)
                return self.loss_and_grad(r, n, g, w)
            cnt = [0]

            def cb(th):
                cnt[0] += 1
                if cnt[0] % 250 == 0:
                    self.net.set_flat(th)
                    ve = val_err()
                    self.history.append(("lbfgs", cnt[0], fg(th)[0], ve))
                    if verbose:
                        print(f"  lbfgs {cnt[0]:5d}  loss {self.history[-1][2]:.3e}  val rel.err {ve:.3e}",
                              flush=True)
            res = minimize(fg, theta, jac=True, method="L-BFGS-B", callback=cb,
                           options=dict(maxiter=lbfgs_iters, maxcor=50, gtol=1e-12, ftol=1e-16))
            self.net.set_flat(res.x)
        self.final_val_err = val_err()
        if verbose:
            print(f"  final val rel.err {self.final_val_err:.3e}")
        return self

    # ---------------------------------------------------------------------
    def save(self, path):
        np.savez(path, theta=self.net.get_flat(), sizes=np.array(self.net.sizes), K=self.K,
                 r_max=self.r_max, R_eff=self.R_eff, n_scale=self.n_scale, y_scale=self.y_scale,
                 history=np.array([h[1:] for h in self.history], float), fourier=net_spec(self.net))

    @classmethod
    def load(cls, nl, path):
        z = np.load(path)
        sizes = z["sizes"]
        obj = cls(nl, K=int(z["K"]), width=int(sizes[1]), depth=len(sizes) - 2,
                  r_max=float(z["r_max"]), R_eff=float(z["R_eff"]), n_scale=float(z["n_scale"]),
                  fourier=spec_from_file(z))
        obj.net.set_flat(z["theta"])
        obj.y_scale = float(z["y_scale"])
        return obj


class Lanczos:
    """Krylov (Lanczos) evaluation of the Galerkin closure f(M_u) e_0 with m matrix-free steps:
    f(M_u) e_0 ~ Q_m f(T_m) e_1, using only Galerkin products u * q (full reorthogonalisation)."""

    def __init__(self, nl, m):
        self.nl, self.basis, self.m = nl, nl.basis, m
        self.label = f"Lanczos (m={m})"

    def __call__(self, u):
        b, P, m = self.basis, self.basis.P, self.m
        ub, sh = _as_batch(u, P)
        Bsz = ub.shape[0]
        Q = np.zeros((m, Bsz, P))
        alpha = np.zeros((Bsz, m))
        beta = np.zeros((Bsz, m))
        Q[0, :, 0] = 1.0
        alive = np.ones(Bsz, bool)
        for j in range(m):
            w = b.sprod(ub, Q[j])
            # after a (lucky) breakdown the Krylov space is invariant: pad T with decoupled entries
            # equal to the mean, whose eigenvectors have no component on e_1
            alpha[:, j] = np.where(alive, np.sum(w * Q[j], axis=1), ub[:, 0])
            w -= alpha[:, j, None] * Q[j]
            if j > 0:
                w -= beta[:, j - 1, None] * Q[j - 1]
            for i in range(j + 1):                       # full reorthogonalisation
                w -= np.sum(w * Q[i], axis=1)[:, None] * Q[i]
            if j < m - 1:
                nb = np.linalg.norm(w, axis=1)
                ok = nb > 1e-12 * (1 + np.abs(alpha[:, j]))
                alive &= ok
                beta[:, j] = np.where(alive, nb, 0.0)
                Q[j + 1] = np.where(alive[:, None], w / np.maximum(nb, 1e-300)[:, None], 0.0)
        T = np.zeros((Bsz, m, m))
        idx = np.arange(m)
        T[:, idx, idx] = alpha
        T[:, idx[:-1], idx[1:]] = beta[:, :-1]
        T[:, idx[1:], idx[:-1]] = beta[:, :-1]
        th, V = np.linalg.eigh(T)
        v0 = V[:, 0, :]
        outs = []
        for fl in self.nl.f(th):
            fv = np.where(np.abs(v0) > 1e-14, fl * v0, 0.0)
            y = np.einsum('bkq,bq->bk', V, fv)             # f(T) e_1 in the Krylov basis
            outs.append(np.einsum('kbp,bk->bp', Q, y))
        return _unbatch(np.concatenate(outs, 1), sh, self.nl.m, P)
