"""
Solution-level physics-informed network for the intrusive PC system (approach B).

On each time window [t_w, t_w + Dt] the PC coefficients are represented by an
extreme-learning-machine PINN with a hard-constrained initial condition (X-TFC):

    c(tau) = c(t_w) + sum_j [phi_j(tau) - phi_j(0)] beta_j ,   phi_j(tau) = tanh(a_j z + b_j),
    z = 2 tau - 1,  tau = (t - t_w)/Dt,                          beta_j in R^{n x P}.

The random state is y(t, xi) = c(t) psi(xi).  The non-polynomial right-hand side is never
projected: the residual   r(t_i, xi_q) = d_t y - f(y, xi)   is enforced in the weighted
least-squares sense at collocation times t_i and stochastic nodes xi_q (Gauss-Hermite
tensor nodes or random samples).  The output weights beta are found by Gauss-Newton /
Levenberg-Marquardt, as in X-TFC / PIELM.
"""
import time

import numpy as np


class SpectralXTFC:
    def __init__(self, problem, xi_nodes, xi_weights, L=16, m=20, dt=1.0, seed=0, a_max=2.0,
                 b_max=1.0, gn_tol=1e-11, gn_max=20):
        self.prob, self.basis = problem, problem.basis
        self.P, self.n = problem.basis.P, problem.n_state
        self.xi, self.w = xi_nodes, xi_weights / xi_weights.sum()
        self.sw = np.sqrt(self.w)
        self.Psi = self.basis.eval_basis(xi_nodes)                # (Q, P)
        _, self.par = problem.sample_inputs(xi_nodes)
        rng = np.random.default_rng(seed)
        self.a = rng.uniform(-a_max, a_max, L)
        self.b = rng.uniform(-b_max, b_max, L)
        self.L, self.m, self.dt = L, m, dt
        # Chebyshev-Gauss-Lobatto collocation points in (0, 1]
        k = np.arange(m + 1)
        self.tau = (0.5 * (1 - np.cos(np.pi * k / m)))[1:]
        self.gn_tol, self.gn_max = gn_tol, gn_max
        self.state_w = np.ones(self.n)        # per-state residual weights (set per window)
        self.stats = dict(gn_iters=[], windows=0, wall=0.0)

    # ---------------------------------------------------------------------
    def features(self, tau):
        z = 2 * np.asarray(tau) - 1
        h = np.tanh(np.outer(z, self.a) + self.b)
        h0 = np.tanh(-self.a + self.b)
        dh = (1 - h ** 2) * self.a * 2.0                         # d/dtau
        return h - h0, dh

    def _coeffs(self, c0, beta, H):
        # c(tau_i) = c0 + sum_j H_ij beta_j   -> (m, n, P)
        return c0[None] + np.einsum('ij,jnp->inp', H, beta)

    def residual(self, c0, beta, H, dH):
        C = self._coeffs(c0, beta, H)                             # (m, n, P)
        dC = np.einsum('ij,jnp->inp', dH, beta) / self.dt
        Y = C @ self.Psi.T                                        # (m, n, Q)
        dY = dC @ self.Psi.T
        F = np.stack([self.prob.mc_rhs(None, Y[i], **self.par) for i in range(len(Y))])
        R = (dY - F) * self.sw * self.state_w[None, :, None]      # (m, n, Q)
        return R, Y

    def jacobian(self, Y, H, dH):
        m, n, Q = Y.shape
        Jf = np.stack([self.prob.mc_jac(Y[i], **self.par) for i in range(m)])   # (m, n, n, Q)
        SP = self.sw[:, None] * self.Psi                          # (Q, P)
        eye = np.eye(n)
        # J[i,a,q, j,b,k] = SP[q,k] (dH[i,j]/dt d_ab - H[i,j] Jf[i,a,b,q])
        T = (dH / self.dt)[:, None, None, :, None, None] * eye[None, :, None, None, :, None] \
            - H[:, None, None, :, None, None] * np.transpose(Jf, (0, 1, 3, 2))[:, :, :, None, :, None]
        J = T * SP[None, None, :, None, None, :] * self.state_w[None, :, None, None, None, None]
        return J.reshape(m * n * Q, self.L * n * self.P)

    def solve_window(self, c0, beta0=None):
        # balance the residual of state variables of very different magnitude: weights from the
        # size of the state and of its change over one window, at the window start (no reference data)
        Y0 = c0 @ self.Psi.T
        F0 = self.prob.mc_rhs(None, Y0, **self.par)
        rms_f = np.sqrt(np.sum(self.w * F0 ** 2, axis=1))
        scale = np.maximum(np.linalg.norm(c0, axis=1), self.dt * rms_f)
        self.state_w = 1.0 / np.maximum(scale, 1e-8)
        H, dH = self.features(self.tau)
        beta = np.zeros((self.L, self.n, self.P)) if beta0 is None else beta0.copy()
        lam = 1e-10
        R, Y = self.residual(c0, beta, H, dH)
        nrm = np.linalg.norm(R)
        for it in range(self.gn_max):
            J = self.jacobian(Y, H, dH)
            Jt = J.T
            A = Jt @ J
            g = Jt @ R.ravel()
            while True:
                step = np.linalg.solve(A + lam * (np.trace(A) / A.shape[0]) * np.eye(A.shape[0]), g)
                bn = beta - step.reshape(beta.shape)
                Rn, Yn = self.residual(c0, bn, H, dH)
                nn = np.linalg.norm(Rn)
                if nn < nrm or lam > 1e3:
                    break
                lam *= 10
            beta, R, Y, lam = bn, Rn, Yn, max(lam / 10, 1e-12)
            small = abs(nrm - nn) <= self.gn_tol * max(nrm, 1e-300)
            nrm = nn
            if small or nrm < 1e-13:
                break
        self.stats["gn_iters"].append(it + 1)
        return beta, nrm

    def propagate(self, c_init, t_end, t_eval):
        """Marches windows of length dt; returns coefficients at t_eval (n_t, n, P)."""
        t0 = time.perf_counter()
        c = c_init.reshape(self.n, self.P).copy()
        out = np.zeros((len(t_eval), self.n, self.P))
        tw, beta = 0.0, None
        res_norms = []
        out[np.isclose(t_eval, 0.0)] = c
        while tw < t_end - 1e-12:
            dt = min(self.dt, t_end - tw)
            self_dt, self.dt = self.dt, dt
            beta, rn = self.solve_window(c, beta)
            res_norms.append(rn)
            sel = (t_eval > tw + 1e-12) & (t_eval <= tw + dt + 1e-12)
            if sel.any():
                H, _ = self.features((t_eval[sel] - tw) / dt)
                out[sel] = self._coeffs(c, beta, H)
            H1, _ = self.features(np.array([1.0]))
            c = self._coeffs(c, beta, H1)[0]
            self.dt = self_dt
            tw += dt
            self.stats["windows"] += 1
        self.stats["wall"] = time.perf_counter() - t0
        self.stats["res_norms"] = np.array(res_norms)
        return out
