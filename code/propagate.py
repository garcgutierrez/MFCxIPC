"""Monte Carlo reference, intrusive propagation and comparison metrics."""
import time

import numpy as np
from scipy.integrate import solve_ivp


def monte_carlo(problem, xi, t_eval, rtol=1e-10, atol=1e-12, method="DOP853", chunk=20000):
    """Integrates all samples simultaneously (vectorised); returns (n_t, n_state, N)."""
    out = []
    for i in range(0, xi.shape[0], chunk):
        y0, par = problem.sample_inputs(xi[i:i + chunk])
        shp = y0.shape

        def f(t, y):
            return problem.mc_rhs(t, y.reshape(shp), **par).ravel()
        sol = solve_ivp(f, (t_eval[0], t_eval[-1]), y0.ravel(), t_eval=t_eval, method=method,
                        rtol=rtol, atol=atol)
        assert sol.success, sol.message
        out.append(sol.y.T.reshape(len(t_eval), *shp))
    return np.concatenate(out, axis=2)


def intrusive(problem, closure, t_eval, rtol=1e-8, atol=1e-10, method="DOP853", c0=None):
    """Integrates the Galerkin system; returns coefficients (n_t, n_state, P), wall time, nfev."""
    c0 = problem.pc_initial() if c0 is None else c0
    t0 = time.perf_counter()
    sol = solve_ivp(lambda t, c: problem.pc_rhs(t, c, closure), (t_eval[0], t_eval[-1]), c0,
                    t_eval=t_eval, method=method, rtol=rtol, atol=atol)
    wall = time.perf_counter() - t0
    if not sol.success:
        print(f"  [intrusive] integration failed at t = {sol.t[-1]:.3f}:", sol.message)
    nt = sol.y.shape[1]
    C = sol.y.T.reshape(nt, problem.n_state, problem.basis.P)
    return C, wall, sol.nfev, sol.success


# -----------------------------------------------------------------------------
def bhattacharyya(a, b, bins=60):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    ha, e = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    return float(np.sum(np.sqrt(ha * hb)) * (e[1] - e[0]))


def compare(basis, C, Y_mc, Psi):
    """
    C: (n_t, n_state, P) PC coefficients; Y_mc: (n_t, n_state, N) MC states at the same xi;
    Psi: (N, P) basis evaluated at the MC samples.
    Returns dict of arrays (n_t, n_state).
    """
    Y_pc = C @ Psi.T
    sd_mc = Y_mc.std(axis=2)
    den = np.maximum(sd_mc, 1e-300)
    N = Y_mc.shape[2]
    Z = (Y_mc - Y_mc.mean(axis=2, keepdims=True)) / np.maximum(sd_mc, 1e-300)[..., None]
    kurt = np.mean(Z ** 4, axis=2)
    res = dict(
        # PC moments (from the coefficients) against MC sample moments: contain the MC sampling error,
        # whose standard deviation is se_mean = 1/sqrt(N) and se_std ~ sqrt((kurt - 1)/(4N)) in units of sd
        mean_err=np.abs(C[..., 0] - Y_mc.mean(axis=2)) / den,
        std_err=np.abs(np.sqrt(np.sum(C[..., 1:] ** 2, -1)) - sd_mc) / den,
        # paired (common-sample) moment errors: the surrogate is evaluated at the MC samples, so the
        # sampling error largely cancels in the difference
        mean_err_p=np.abs(Y_pc.mean(axis=2) - Y_mc.mean(axis=2)) / den,
        std_err_p=np.abs(Y_pc.std(axis=2) - sd_mc) / den,
        se_mean=np.full(sd_mc.shape, 1.0 / np.sqrt(N)),
        se_std=np.sqrt(np.maximum(kurt - 1.0, 0.0) / (4.0 * N)),
        l2_err=np.sqrt(np.mean((Y_pc - Y_mc) ** 2, axis=2)) / den,
    )
    bc = np.zeros(C.shape[:2])
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            bc[i, j] = bhattacharyya(Y_pc[i, j], Y_mc[i, j])
    res["bc"] = bc
    return res
