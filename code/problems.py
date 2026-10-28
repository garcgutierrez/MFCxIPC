"""
Benchmark problems: deterministic (vectorised, for Monte Carlo) and intrusive Galerkin RHS.

Every problem exposes
  sample_inputs(xi)            -> physical inputs for Monte Carlo
  mc_rhs(t, y, params)         -> vectorised deterministic RHS
  pc_initial(basis)            -> initial PC coefficients (flattened)
  pc_rhs(t, c, closures)       -> Galerkin RHS using the supplied closure(s)
  observables                  -> names of state components
"""
import numpy as np

from pce import lognormal_coeffs, linear_coeffs


# =============================================================================
# 1. Damped nonlinear pendulum   th'' + gamma th' + Omega^2 sin(th) = 0
#    xi_1 -> initial angle (Gaussian), xi_2 -> natural frequency (log-normal)
# =============================================================================
class Pendulum:
    name = "pendulum"
    observables = ["theta", "omega"]

    def __init__(self, basis, theta0=2.0, s_theta=0.15, Omega=1.0, s_Omega=0.05, gamma=0.1):
        self.basis = basis
        self.theta0, self.s_theta = theta0, s_theta
        self.Omega, self.s_Omega, self.gamma = Omega, s_Omega, gamma
        self.n_state = 2
        d = basis.d
        # Omega^2 = Omega_bar^2 exp(2 s_Omega xi_2) has an exact Hermite expansion
        if d >= 2:
            self.Om2 = lognormal_coeffs(basis, 2 * np.log(Omega), 2 * s_Omega, 1)
        else:
            self.Om2 = Omega ** 2 * basis.e0

    # ---- Monte Carlo -----------------------------------------------------
    def sample_inputs(self, xi):
        th0 = self.theta0 + self.s_theta * xi[:, 0]
        Om2 = self.Omega ** 2 * (np.exp(2 * self.s_Omega * xi[:, 1]) if xi.shape[1] > 1 else 1.0)
        y0 = np.stack([th0, np.zeros_like(th0)])
        return y0, dict(Om2=Om2 * np.ones_like(th0))

    def mc_rhs(self, t, y, Om2):
        th, om = y
        return np.stack([om, -self.gamma * om - Om2 * np.sin(th)])

    def mc_jac(self, y, Om2):
        th, om = y
        z, o = np.zeros_like(th), np.ones_like(th)
        return np.array([[z, o], [-Om2 * np.cos(th), -self.gamma * o]])

    # ---- intrusive Galerkin ------------------------------------------------
    def pc_initial(self):
        b = self.basis
        th = linear_coeffs(b, self.theta0, self.s_theta, 0)
        return np.concatenate([th, np.zeros(b.P)])

    def pc_rhs(self, t, c, closure):
        P = self.basis.P
        th, om = c[:P], c[P:]
        S = closure(th)[:P]                 # sin block of the sin/cos closure
        return np.concatenate([om, -self.gamma * om - self.basis.sprod(self.Om2, S)])

    def closure_inputs(self, c):
        return [c[:self.basis.P]]


# =============================================================================
# 2. Genetic toggle switch (Gardner et al. 2000)
#    du/dt = a1/(1+v^beta) - u ,  dv/dt = a2/(1+w^gamma) - v ,  w = u/(1+IPTG/K)^eta
#    random: a1, a2, eta, K (log-normal); beta, gamma fixed.
# =============================================================================
class ToggleSwitch:
    name = "toggle"
    observables = ["u", "v"]

    def __init__(self, basis, a1=156.25, a2=15.6, beta=2.5, gamma=1.0, eta=2.0015,
                 K=2.9618e-5, iptg=1e-5, cov=(0.1, 0.1, 0.1, 0.1), y0=(1.0, 1.0), q_param=None):
        self.basis = basis
        self.mean = dict(a1=a1, a2=a2, eta=eta, K=K)
        self.cov = cov
        self.beta, self.gamma, self.iptg = beta, gamma, iptg
        self.y0 = np.array(y0, float)
        self.n_state = 2
        d = basis.d
        names = ["a1", "a2", "eta", "K"]
        self.sig = {nm: (np.sqrt(np.log(1 + cov[i] ** 2)) if i < d else 0.0) for i, nm in enumerate(names)}
        # log-normal parameters with prescribed mean and CoV
        self.coef = {}
        for i, nm in enumerate(names):
            s = self.sig[nm]
            mu_log = np.log(self.mean[nm]) - 0.5 * s ** 2
            self.coef[nm] = lognormal_coeffs(basis, mu_log, s, i) if i < d else self.mean[nm] * basis.e0
        # 1/(1+IPTG/K)^eta depends on parameters only -> projected once, accurately
        q = q_param or (2 * basis.p + 6)

        def inv_c(X):
            p = self._params_from_xi(X)
            return (1.0 + self.iptg / p["K"]) ** (-p["eta"])
        self.inv_c = basis.project(inv_c, q)

    def _params_from_xi(self, xi):
        out = {}
        for i, nm in enumerate(["a1", "a2", "eta", "K"]):
            s = self.sig[nm]
            z = xi[:, i] if i < xi.shape[1] else 0.0
            out[nm] = self.mean[nm] * np.exp(s * z - 0.5 * s ** 2)
        return out

    def sample_inputs(self, xi):
        p = self._params_from_xi(xi)
        n = xi.shape[0]
        y0 = np.tile(self.y0[:, None], (1, n))
        return y0, dict(a1=p["a1"], a2=p["a2"], inv_c=(1.0 + self.iptg / p["K"]) ** (-p["eta"]))

    def mc_rhs(self, t, y, a1, a2, inv_c):
        u, v = y
        w = u * inv_c
        return np.stack([a1 / (1 + np.abs(v) ** self.beta) - u, a2 / (1 + np.abs(w) ** self.gamma) - v])

    def mc_jac(self, y, a1, a2, inv_c):
        u, v = y
        w = u * inv_c
        vb = np.abs(v) ** self.beta
        wg = np.abs(w) ** self.gamma
        dudv = -a1 * self.beta * vb / np.maximum(np.abs(v), 1e-300) * np.sign(v) / (1 + vb) ** 2
        dvdu = -a2 * self.gamma * wg / np.maximum(np.abs(w), 1e-300) * np.sign(w) * inv_c / (1 + wg) ** 2
        m1 = -np.ones_like(u)
        return np.array([[m1, dudv], [dvdu, m1]])

    def pc_initial(self):
        b = self.basis
        return np.concatenate([self.y0[0] * b.e0, self.y0[1] * b.e0])

    def pc_rhs(self, t, c, closures):
        """closures = dict(pow_beta=..., inv=...)."""
        b, P = self.basis, self.basis.P
        u, v = c[:P], c[P:]
        w = b.sprod(u, self.inv_c)
        vb = closures["pow_beta"](v)
        wg = w if self.gamma == 1 else closures["pow_gamma"](w)
        den = np.stack([b.e0 + vb, b.e0 + wg])
        inv = closures["inv"](den)
        du = b.sprod(self.coef["a1"], inv[0]) - u
        dv = b.sprod(self.coef["a2"], inv[1]) - v
        return np.concatenate([du, dv])


# =============================================================================
# 3. Solid-fuel ignition (Frank-Kamenetskii) reaction-diffusion
#    u_t = kappa u_xx + delta(x) exp(u),  x in (0,1),  u(0)=u(1)=0,  u(x,0)=0
#    log delta(x) = log(delta_bar) - var/2 + sum_k s_k phi_k(x) xi_k   (k = 1..d-1)
#    phi_1 = 1 (global level), phi_k = sqrt(2) sin((k-1) pi x)  (k >= 2),  s_k = s_1 / (k-1)
#    kappa log-normal driven by xi_d.  Second-order finite differences (method of lines).
# =============================================================================
class Ignition:
    name = "ignition"

    def __init__(self, basis, nx=31, delta=2.0, s_delta=0.1, kappa=1.0, cov_kappa=0.1):
        self.basis, self.nx = basis, nx
        d = basis.d
        self.x = np.linspace(0, 1, nx + 2)[1:-1]
        self.h = 1.0 / (nx + 1)
        self.n_kl = d - 1 if d >= 2 else 1
        self.kappa_dim = d - 1 if d >= 2 else None
        self.mean_delta, self.mean_kappa = delta, kappa
        # modal amplitudes a_k(x) for every node: (nx, n_kl)
        A = np.zeros((nx, self.n_kl))
        A[:, 0] = s_delta
        for k in range(1, self.n_kl):
            A[:, k] = s_delta / k * np.sqrt(2) * np.sin(k * np.pi * self.x)
        self.A = A
        self.sk = np.sqrt(np.log(1 + cov_kappa ** 2))
        var = np.sum(A ** 2, axis=1)
        self.mu_log = np.log(delta) - 0.5 * var
        # exact Hermite coefficients of exp(mu + sum_k a_k xi_k) at every node
        alph = basis.alphas
        from math import factorial
        coef = np.exp(self.mu_log + 0.5 * var)[:, None] * np.ones((nx, basis.P))
        for k in range(self.n_kl):
            n = alph[:, k]
            coef *= A[:, [k]] ** n[None, :] / np.sqrt([factorial(int(v)) for v in n])[None, :]
        if self.kappa_dim is not None:
            coef[:, alph[:, self.kappa_dim] > 0] = 0.0
        self.delta = coef
        self.kappa = (lognormal_coeffs(basis, np.log(kappa) - 0.5 * self.sk ** 2, self.sk, self.kappa_dim)
                      if self.kappa_dim is not None else kappa * basis.e0)
        self.n_state = nx
        L = (np.diag(-2 * np.ones(nx)) + np.diag(np.ones(nx - 1), 1) + np.diag(np.ones(nx - 1), -1))
        self.L = L / self.h ** 2
        self.mid = nx // 2

    def sample_inputs(self, xi):
        n = xi.shape[0]
        z = xi[:, :self.n_kl]
        de = np.exp(self.mu_log[:, None] + self.A @ z.T)                     # (nx, N)
        ka = (self.mean_kappa * np.exp(self.sk * xi[:, self.kappa_dim] - 0.5 * self.sk ** 2)
              if self.kappa_dim is not None else self.mean_kappa * np.ones(n))
        return np.zeros((self.nx, n)), dict(delta=de, kappa=ka)

    def mc_rhs(self, t, y, delta, kappa):
        return kappa * (self.L @ y) + delta * np.exp(y)

    def mc_jac(self, y, delta, kappa):
        raise NotImplementedError

    def pc_initial(self):
        return np.zeros(self.nx * self.basis.P)

    def pc_rhs(self, t, c, closure):
        b, P = self.basis, self.basis.P
        U = c.reshape(self.nx, P)
        diff = b.sprod(np.broadcast_to(self.kappa, U.shape), self.L @ U)
        E = closure(U)
        return (diff + b.sprod(self.delta, E)).ravel()


# =============================================================================
# 4. Forced oscillator with a massless node and a hardening spring (index-1 DAE)
#    m x'' + c x' + k1 (x - y) = F cos(W t),     k1 (x - y) = k2 y + k3 y^3
#    random: x0, k1, k3, F (d = 4)  [+ c, v0 for d = 6];  the node position y is an
#    implicitly defined, bivariate function of the random fields x and (k1, k3).
# =============================================================================
class SeriesSpringOscillator:
    name = "series_spring"
    observables = ["x", "v"]

    def __init__(self, basis, x0=1.0, s_x0=0.15, k1=1.0, cov_k1=0.1, k2=0.5, k3=2.0, cov_k3=0.2,
                 F=0.3, cov_F=0.1, c=0.05, cov_c=0.2, s_v0=0.1, W=0.8):
        from math import log, sqrt
        self.basis, d = basis, basis.d
        self.W, self.k2 = W, k2
        self.n_state = 2
        ln = lambda m, cov: (log(m) - 0.5 * log(1 + cov ** 2), sqrt(log(1 + cov ** 2)))
        self.spec = dict(k1=(k1, cov_k1, 1), k3=(k3, cov_k3, 2), F=(F, cov_F, 3), c=(c, cov_c, 4))
        self.x0, self.s_x0, self.s_v0 = x0, s_x0, s_v0
        self.coef = {}
        for nm, (m, cov, dim) in self.spec.items():
            mu, s = ln(m, cov)
            self.coef[nm] = lognormal_coeffs(basis, mu, s, dim) if dim < d else m * basis.e0
        from implicit import CubicSystem
        self.sys = CubicSystem(basis, self.coef["k1"], k2 * basis.e0, self.coef["k3"])

    def _param(self, nm, xi):
        m, cov, dim = self.spec[nm]
        if dim >= xi.shape[1]:
            return m * np.ones(xi.shape[0])
        s = np.sqrt(np.log(1 + cov ** 2))
        return m * np.exp(s * xi[:, dim] - 0.5 * s ** 2)

    def sample_inputs(self, xi):
        x0 = self.x0 + self.s_x0 * xi[:, 0]
        v0 = self.s_v0 * xi[:, 5] if xi.shape[1] > 5 else np.zeros(xi.shape[0])
        par = {nm: self._param(nm, xi) for nm in self.spec}
        return np.stack([x0, v0]), par

    def mc_rhs(self, t, y, k1, k3, F, c):
        from implicit import cubic_root
        x, v = y
        yn = cubic_root(k3, k1 + self.k2, k1 * x)
        tt = 0.0 if t is None else t
        return np.stack([v, F * np.cos(self.W * tt) - c * v - k1 * (x - yn)])

    def pc_initial(self):
        b = self.basis
        x = linear_coeffs(b, self.x0, self.s_x0, 0)
        v = linear_coeffs(b, 0.0, self.s_v0, 5) if b.d > 5 else np.zeros(b.P)
        return np.concatenate([x, v])

    def pc_rhs(self, t, c, closure):
        b, P = self.basis, self.basis.P
        x, v = c[:P], c[P:]
        yn = closure(x)
        k = self.coef
        dv = np.cos(self.W * t) * k["F"] - b.sprod(k["c"], v) - b.sprod(k["k1"], x - yn)
        return np.concatenate([v, dv])


def lognormal_field_coeffs(basis, mu_log, A, dims):
    """Exact Hermite coefficients of exp(mu_log(x) + sum_k A[x, k] xi_{dims[k]}) at every node x."""
    from math import factorial
    var = np.sum(A ** 2, axis=1)
    coef = np.exp(mu_log + 0.5 * var)[:, None] * np.ones((len(mu_log), basis.P))
    alph = basis.alphas
    used = np.zeros(basis.d, bool)
    for k, dim in enumerate(dims):
        n = alph[:, dim]
        coef *= A[:, [k]] ** n[None, :] / np.sqrt([factorial(int(v)) for v in n])[None, :]
        used[dim] = True
    coef[:, np.any(alph[:, ~used] > 0, axis=1)] = 0.0
    return coef


# =============================================================================
# 5. Nonlinear heat conduction in enthalpy form with an implicit constitutive law
#    h_t = kappa T_xx + Q sin(pi x),   h = T + k3(x, xi) T^3,   T(0) = T(1) = 0,  h(x, 0) = 0
#    xi_1 -> kappa, xi_2 -> Q, xi_3.. -> KL modes of log k3(x).  Temperature from enthalpy at every
#    node requires the solution of a cubic whose coefficient is a random field.
# =============================================================================
class EnthalpyHeat:
    name = "enthalpy"

    def __init__(self, basis, nx=15, kappa=1.0, cov_kappa=0.1, Q=15.0, cov_Q=0.1, k3=1.0, s_k3=0.2):
        b, d = basis, basis.d
        self.basis, self.nx = b, nx
        self.x = np.linspace(0, 1, nx + 2)[1:-1]
        h = 1.0 / (nx + 1)
        L = (np.diag(-2 * np.ones(nx)) + np.diag(np.ones(nx - 1), 1) + np.diag(np.ones(nx - 1), -1)) / h ** 2
        self.L = L
        self.src = np.sin(np.pi * self.x)
        self.sk = np.sqrt(np.log(1 + cov_kappa ** 2))
        self.sq = np.sqrt(np.log(1 + cov_Q ** 2))
        self.kappa_bar, self.Q_bar = kappa, Q
        self.kappa = lognormal_coeffs(b, np.log(kappa) - 0.5 * self.sk ** 2, self.sk, 0)
        self.Q = lognormal_coeffs(b, np.log(Q) - 0.5 * self.sq ** 2, self.sq, 1)
        self.n_kl = d - 2
        A = np.zeros((nx, self.n_kl))
        A[:, 0] = s_k3
        for k in range(1, self.n_kl):
            A[:, k] = s_k3 / k * np.sqrt(2) * np.sin(k * np.pi * self.x)
        self.A = A
        self.mu_log = np.log(k3) - 0.5 * np.sum(A ** 2, axis=1)
        self.K3 = lognormal_field_coeffs(b, self.mu_log, A, list(range(2, d)))   # (nx, P)
        self.n_state = nx
        self.mid = nx // 2

    def sample_inputs(self, xi):
        n = xi.shape[0]
        kap = self.kappa_bar * np.exp(self.sk * xi[:, 0] - 0.5 * self.sk ** 2)
        Q = self.Q_bar * np.exp(self.sq * xi[:, 1] - 0.5 * self.sq ** 2)
        k3 = np.exp(self.mu_log[:, None] + self.A @ xi[:, 2:].T)               # (nx, N)
        return np.zeros((self.nx, n)), dict(kappa=kap, Q=Q, k3=k3)

    def mc_rhs(self, t, y, kappa, Q, k3):
        from implicit import cubic_root
        T = cubic_root(k3, 1.0, y)
        return kappa * (self.L @ T) + Q * self.src[:, None]

    def pc_initial(self):
        return np.zeros(self.nx * self.basis.P)

    def pc_rhs(self, t, c, closure):
        b, P = self.basis, self.basis.P
        H = c.reshape(self.nx, P)
        T = closure(H, self.K3)
        dH = b.sprod(np.broadcast_to(self.kappa, T.shape), self.L @ T) + self.src[:, None] * self.Q[None, :]
        return dH.ravel()
