"""Focused tests of the central identities and of every hand-written derivative.

Run:  python tests/test_core.py          (no test framework needed; pytest also collects the test_* functions)
"""
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pce
from pce import PCBasis, sparse_quadrature
from closures import Exp, SinCos, Power, ExactGalerkin, PSP, Taylor, Lanczos, PINNClosure, sample_directions
from mlp import MLP
from implicit import CubicSystem, NewtonClosure, cubic_root, gmres_batched, _jv, CubicSolver
from implicit_field import CubicField, FieldNewton, FieldSolver, FieldPINN

RNG = np.random.default_rng(12345)


def rand_input(B, s=0.3, mean=0.0, decay=0.6):
    u = RNG.normal(size=B.P) * s * decay ** B.degree
    u[0] = mean
    return u


def mf(B, u, f):
    lam, V = np.linalg.eigh(B.mult_matrix(u))
    return V @ (f(lam) * V[0])


# --------------------------------------------------------------------------- Galerkin algebra
def test_triple_products_against_quadrature():
    B = PCBasis(2, 4)
    X, W = B.quadrature(12)                      # exact for degree <= 23 >= 3p
    Psi = B.eval_basis(X)
    Cq = np.einsum('q,qi,qj,qk->ijk', W, Psi, Psi, Psi)
    assert np.abs(Cq - B.C).max() < 1e-12


def test_multiplication_matrix():
    B = PCBasis(3, 3)
    u, v = rand_input(B, mean=0.7), rand_input(B, mean=-0.2)
    M = B.mult_matrix(u)
    assert np.allclose(M, M.T) and np.allclose(M @ B.e0, u)
    assert np.allclose(M @ v, B.sprod(u, v)) and np.allclose(B.mult_matrix_dense(u), M)
    pce.DENSE_C_MAX, old = 5, pce.DENSE_C_MAX
    try:
        B2 = PCBasis(3, 3)                        # sparse construction path
    finally:
        pce.DENSE_C_MAX = old
    assert np.allclose(B2.sprod(u, v), B.sprod(u, v)) and np.allclose(B2.mult_matrix(u), M)


def test_horner_characterisation():
    """G_q(u) = q(M_u) e0 equals Horner's scheme evaluated in Galerkin arithmetic."""
    B = PCBasis(2, 4)
    u = rand_input(B, mean=0.3)
    c = RNG.normal(size=6)
    y = c[-1] * B.e0
    for ck in c[-2::-1]:
        y = ck * B.e0 + B.sprod(u, y)
    M = B.mult_matrix(u)
    qM = sum(ck * np.linalg.matrix_power(M, k) for k, ck in enumerate(c))
    assert np.allclose(y, qM[:, 0], atol=1e-12)
    # right nesting is what M_u^m e0 reproduces; other nestings differ for m >= 4 (non-associativity)
    uu = B.sprod(u, u)
    assert np.allclose(B.sprod(u, B.sprod(u, uu)), np.linalg.matrix_power(M, 4)[:, 0])


def test_exact_projection_condition():
    """(deg q - 1) deg(u) <= p  =>  q(M_u) e0 = Pi[q(u_hat)]; fails otherwise."""
    B = PCBasis(2, 3)
    X, W = B.quadrature(10)
    Psi = B.eval_basis(X)
    u = np.zeros(B.P); u[0], u[1], u[2] = 0.2, 0.5, -0.3                  # linear input
    M = B.mult_matrix(u)
    uh = Psi @ u
    for deg, exact in [(4, True), (5, False)]:                            # p + 1 = 4
        G = np.linalg.matrix_power(M, deg)[:, 0]
        H = (uh ** deg * W) @ Psi
        assert (np.abs(G - H).max() < 1e-12) == exact


def test_gauss_equivalence_1d():
    """Proposition: d = 1, linear input: f(M_u) e0 = PSP with the (p+1)-point Gauss rule."""
    B = PCBasis(1, 6)
    u = np.zeros(B.P); u[0], u[1] = 0.4, 0.7
    for f in (np.exp, np.sin):
        Q = PSP(Exp(B), B.p + 1)
        assert np.allclose(mf(B, u, f), f(u @ Q.Psi.T) @ Q.Wpsi, atol=1e-13)


def test_mean_is_positive_quadrature_and_jensen():
    B = PCBasis(2, 4)
    u = rand_input(B, mean=0.1, s=0.5)
    lam, V = np.linalg.eigh(B.mult_matrix(u))
    w = V[0] ** 2
    assert abs(w.sum() - 1) < 1e-12 and np.all(w >= 0) and abs(w @ lam - u[0]) < 1e-12
    assert mf(B, u, np.exp)[0] >= np.exp(u[0])                             # Jensen, convex f


def test_positive_input_gives_spd_matrix():
    B = PCBasis(1, 8)
    u = np.zeros(B.P); u[0], u[2] = 1.0 + 0.3, 0.3 * np.sqrt(2)          # 1 + 0.3 xi^2 >= 1
    assert np.linalg.eigvalsh(B.mult_matrix(u)).min() >= 1.0 - 1e-12


def test_sqrt_matrix_function_is_not_galerkin_root():
    B = PCBasis(2, 4)
    u = np.zeros(B.P); u[0], u[1], u[2], u[3], u[4] = 2.0, 0.4, 0.3, 0.15, 0.1
    G = mf(B, u, np.sqrt)
    y = G.copy()
    for _ in range(50):
        y = y - np.linalg.solve(2 * B.mult_matrix(y), B.sprod(y, y) - u)
    assert np.linalg.norm(B.sprod(y, y) - u) < 1e-13
    assert np.linalg.norm(G - y) / np.linalg.norm(y) > 1e-6                # different closures


# --------------------------------------------------------------------------- ray ODE and Taylor terms
def _ray_solution(nl, n, r):
    """Integrate B(s) G' = A_n G from 0 to r."""
    def rhs(s, g):
        dg = nl.A_apply(n[None], g[None])[0]
        if nl.c != 0:                                                        # (I + s M_n) G' = a M_n G
            Mn = nl.basis.mult_matrix(n)
            dg = np.linalg.solve(np.eye(nl.P) + s * Mn, dg)
        return dg
    sol = solve_ivp(rhs, (0, r), nl.g0(), rtol=1e-12, atol=1e-14, method="DOP853")
    return sol.y[:, -1]


def test_ray_ode_and_taylor_recursion():
    B = PCBasis(2, 4)
    x = rand_input(B, s=0.4, mean=0.0)
    x *= 0.6 / np.abs(np.linalg.eigvalsh(B.mult_matrix(x))).max()     # I + M_x must stay SPD for powers
    r = np.linalg.norm(x)
    n = x / r
    for nl, exact in [(Exp(B), lambda: mf(B, x, np.exp)),
                      (SinCos(B), lambda: np.concatenate([mf(B, x, np.sin), mf(B, x, np.cos)])),
                      (Power(B, 2.5), lambda: mf(B, B.e0 + x, lambda z: z ** 2.5)),
                      (Power(B, -1.0), lambda: mf(B, B.e0 + x, lambda z: 1 / z))]:
        assert np.allclose(_ray_solution(nl, n, r), exact(), atol=1e-10), nl
        g = nl.taylor_terms(x[None], 40)
        assert np.allclose(sum(g)[0], exact(), atol=1e-8), nl      # converges: entire f, or rho(M_x) < 1


def test_scaled_residual_vanishes_for_exact_remainder():
    """rho of eq. (residual) is zero when N is the exact remainder (G - Taylor_K)/r^(K+1)."""
    B = PCBasis(2, 3)
    K = 3
    for nl, f in [(Exp(B), np.exp), (Power(B, 2.5), lambda z: z ** 2.5)]:
        x = rand_input(B, s=0.3)
        n = x / np.linalg.norm(x)
        n_rho = np.abs(np.linalg.eigvalsh(B.mult_matrix(n))).max()
        shift = 0.0 if nl.c == 0 else 1.0

        def N(r):
            G = mf(B, shift * B.e0 + r * n, f)
            g = nl.taylor_terms((r * n)[None], K - 1)[:K + 1]
            return (G - sum(g)[0]) / r ** (K + 1)
        r, h = 0.6 / n_rho, 1e-4                                         # spectral radius of r M_n = 0.6
        Nd = (N(r + h) - N(r - h)) / (2 * h)
        gK1 = nl.taylor_terms(n[None], K)[K + 1][0]
        rho = (nl.B_apply(n[None], np.array([r]), ((K + 1) * N(r) + r * Nd)[None])[0]
               - r * nl.A_apply(n[None], N(r)[None])[0] - (K + 1) * gK1)
        assert np.linalg.norm(rho) < 1e-6 * max(1, np.linalg.norm(gK1)), (nl, np.linalg.norm(rho))


def test_lanczos_exactness():
    B = PCBasis(3, 3)
    u = rand_input(B, mean=0.2, s=0.5)
    nl = Exp(B)
    M = B.mult_matrix(u)
    for m in (3, 6):
        L = Lanczos(nl, m)
        # polynomial of degree m-1: exact vector; the Gauss rule e1' f(T) e1 is exact to degree 2m-1
        nl_f, nl.f = nl.f, (lambda z, m=m: [z ** (m - 1)])
        try:
            assert np.allclose(L(u), np.linalg.matrix_power(M, m - 1)[:, 0], atol=1e-10)
            nl.f = lambda z, m=m: [z ** (2 * m - 1)]
            assert abs(L(u)[0] - np.linalg.matrix_power(M, 2 * m - 1)[0, 0]) < 1e-9
        finally:
            nl.f = nl_f
    assert np.allclose(Lanczos(nl, 14)(u), ExactGalerkin(nl)(u), atol=1e-11)


# --------------------------------------------------------------------------- derivatives
def _fd_check(fun, theta, g, n=6, h=1e-6):
    idx = RNG.choice(theta.size, n, replace=False)
    err = 0.0
    for i in idx:
        e = np.zeros_like(theta); e[i] = h
        fd = (fun(theta + e) - fun(theta - e)) / (2 * h)
        err = max(err, abs(fd - g[i]) / max(1e-8, abs(fd) + abs(g[i])))
    return err


def test_mlp_tangent_and_gradient():
    net = MLP([4, 7, 7, 3], seed=1)
    x, xd = RNG.normal(size=(5, 4)), RNG.normal(size=(5, 4))
    y, yd, cache = net.forward(x, xd)
    h = 1e-6
    assert np.allclose(yd, (net.forward(x + h * xd)[0] - net.forward(x - h * xd)[0]) / (2 * h), atol=1e-7)
    gy, gyd = RNG.normal(size=y.shape), RNG.normal(size=y.shape)
    gW, gb = net.backward(cache, gy, gyd)
    g = MLP.flat_grads(gW, gb)
    th0 = net.get_flat()

    def L(th):
        net.set_flat(th)
        y_, yd_, _ = net.forward(x, xd)
        return np.sum(gy * y_) + np.sum(gyd * yd_)
    assert _fd_check(L, th0, g) < 1e-5
    net.set_flat(th0)


def test_pinn_closure_loss_gradient():
    B = PCBasis(2, 3)
    for nl in (Exp(B), SinCos(B), Power(B, 2.5)):
        pc = PINNClosure(nl, K=3, width=16, depth=2, r_max=1.0, R_eff=3.0, seed=2)
        D = sample_directions(B, 32, RNG)
        r = 0.5 * pc.r_limit(D) * RNG.random(32)
        gK1 = nl.taylor_terms(D, pc.K)[pc.K + 1]
        th0 = pc.net.get_flat()
        _, g = pc.loss_and_grad(r, D, gK1)

        def L(th):
            pc.net.set_flat(th)
            return pc.loss_and_grad(r, D, gK1)[0]
        assert _fd_check(L, th0, g) < 1e-4, nl


def test_implicit_jacobians_and_losses():
    B = PCBasis(3, 3)
    k1, k3 = rand_input(B, mean=1.0, s=0.1), rand_input(B, mean=2.0, s=0.2)
    s = CubicSystem(B, k1, 0.5 * B.e0, k3)
    x, y, v = rand_input(B, mean=0.5), rand_input(B, mean=0.3), rand_input(B)
    h = 1e-6
    Jv_fd = (s.residual(x[None], (y + h * v)[None]) - s.residual(x[None], (y - h * v)[None]))[0] / (2 * h)
    assert np.allclose(_jv(s, y[None], v[None])[0], Jv_fd, atol=1e-7)
    assert np.allclose(s.jacobian(y[None])[0] @ v, Jv_fd, atol=1e-7)
    w = rand_input(B)
    assert np.allclose(s.jT_apply(y[None], w[None])[0], s.jacobian(y[None])[0].T @ w, atol=1e-10)
    fs = CubicField(B)
    hh = rand_input(B, mean=1.0, s=0.3)
    Jf = fs.jacobian(k3[None], y[None])[0]
    assert np.allclose(fs.jv(k3[None], y[None], v[None])[0], Jf @ v, atol=1e-10)
    assert np.allclose(fs.jtv(k3[None], y[None], w[None])[0], Jf.T @ w, atol=1e-10)
    fd = (fs.residual(hh[None], k3[None], (y + h * v)[None]) - fs.residual(hh[None], k3[None], (y - h * v)[None]))[0]
    assert np.allclose(Jf @ v, fd / (2 * h), atol=1e-7)
    # residual-trained network: loss gradient
    pn = FieldPINN(fs, width=12, depth=2, seed=3)
    H = np.stack([rand_input(B, mean=1.0, s=0.3) for _ in range(8)])
    K = np.tile(k3, (8, 1))
    wts = np.ones(8)
    th0 = pn.net.get_flat()
    _, g = pn.loss_and_grad(H, K, wts)

    def L(th):
        pn.net.set_flat(th)
        return pn.loss_and_grad(H, K, wts)[0]
    assert _fd_check(L, th0, g) < 1e-4


# --------------------------------------------------------------------------- quadrature, solvers
def test_sparse_quadrature_exactness():
    for d, k in [(3, 2), (5, 3)]:
        X, W = sparse_quadrature(d, k)
        B = PCBasis(d, k)
        Psi = B.eval_basis(X)
        assert np.allclose((Psi * W[:, None]).T @ Psi, np.eye(B.P), atol=1e-11)   # degree 2k exact
        assert abs(W @ X[:, 0] ** (2 * k + 1)) < 1e-10 and abs(W @ X[:, 0] ** (2 * k + 2) - np.prod(
            np.arange(2 * k + 1, 0, -2))) > 1e-6                                      # degree 2k+2 not exact


def test_cardano_and_solvers():
    a3, a1, b = 2.0, 1.5, RNG.normal(size=50) * 4
    y = cubic_root(a3, a1, b)
    assert np.allclose(a3 * y ** 3 + a1 * y, b, atol=1e-10)
    A = RNG.normal(size=(6, 6)) + 6 * np.eye(6)
    rhs = RNG.normal(size=(1, 6))
    assert np.allclose(gmres_batched(lambda v: v @ A.T, rhs, 6)[0], np.linalg.solve(A, rhs[0]), atol=1e-10)
    B = PCBasis(3, 3)
    fs = CubicField(B)
    k3 = np.tile(rand_input(B, mean=1.0, s=0.2), (4, 1))
    H = np.stack([rand_input(B, mean=1.5, s=0.3) for _ in range(4)])
    ref = FieldNewton(fs)(H, k3)
    for meth in ("newton", "nk", "chord"):
        y = FieldSolver(fs, method=meth, init="linear", tol=1e-11)(H, k3)
        assert np.abs(y - ref).max() < 1e-9, meth
    s = CubicSystem(B, rand_input(B, mean=1.0, s=0.1), 0.5 * B.e0, rand_input(B, mean=2.0, s=0.2))
    x = rand_input(B, mean=0.8, s=0.2)
    ref = NewtonClosure(s)(x)
    for meth in ("newton", "nk", "chord"):
        assert np.abs(CubicSolver(s, method=meth, tol=1e-11)(x) - ref).max() < 1e-9, meth


def test_fourier_features_and_adaptive_sampling():
    """Fourier-feature networks: loss gradients (explicit ray-ODE loss with input tangents, implicit residual
    losses) against finite differences, save/load round trips; adaptive-sampling training runs."""
    import tempfile
    from implicit import PINNImplicit
    from mlp import rad_indices
    B = PCBasis(2, 3)
    nl = SinCos(B)
    pc = PINNClosure(nl, K=3, width=16, depth=2, r_max=1.0, R_eff=3.0, seed=2, fourier=(8, 1.0))
    D = sample_directions(B, 32, RNG)
    r = 0.5 * pc.r_limit(D) * RNG.random(32)
    gK1 = nl.taylor_terms(D, pc.K)[pc.K + 1]
    th0 = pc.net.get_flat()
    _, g = pc.loss_and_grad(r, D, gK1)

    def L(th):
        pc.net.set_flat(th)
        return pc.loss_and_grad(r, D, gK1)[0]
    assert _fd_check(L, th0, g) < 1e-4
    pc.net.set_flat(th0)
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "m.npz")
        pc.save(f)
        assert np.allclose(PINNClosure.load(nl, f).G(r, D), pc.G(r, D), atol=1e-14)
    B3 = PCBasis(3, 2)
    fs = CubicField(B3)
    k3 = rand_input(B3, mean=2.0, s=0.2)
    pn = FieldPINN(fs, width=12, depth=2, seed=3, fourier=(6, 0.5))
    H = np.stack([rand_input(B3, mean=1.0, s=0.3) for _ in range(8)])
    K = np.tile(k3, (8, 1))
    th0 = pn.net.get_flat()
    _, g = pn.loss_and_grad(H, K, np.ones(8))

    def Lf(th):
        pn.net.set_flat(th)
        return pn.loss_and_grad(H, K, np.ones(8))[0]
    assert _fd_check(Lf, th0, g) < 1e-4
    pn.net.set_flat(th0)
    pn.train(H, K, n_pool=64, batch=16, iters=16, lbfgs_iters=0, verbose=False, adaptive=dict(n_cand=128, every=4))
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "f.npz")
        pn.save(f)
        assert np.allclose(FieldPINN.from_file(fs, f)(H, K), pn(H, K), atol=1e-14)
    s = CubicSystem(B3, rand_input(B3, mean=1.0, s=0.1), 0.5 * B3.e0, k3)
    pi = PINNImplicit(s, width=12, depth=2, seed=1, fourier=(6, 0.5))
    pi.train(H, n_pool=64, batch=16, iters=16, lbfgs_iters=0, verbose=False, adaptive=dict(n_cand=128, every=4))
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "i.npz")
        pi.save(f)
        assert np.allclose(PINNImplicit.from_file(s, f)(H), pi(H), atol=1e-14)
    pc.train(n_dirs=64, batch=16, iters=16, lbfgs_iters=0, verbose=False, n_val=8, adaptive=dict(n_cand=128, every=4))
    # RAD with k = c = 1: probabilities 1 : 3 for residuals 0 and 10 (mean 5), i.e. 3/4 of the draws
    idx = rad_indices(np.r_[np.zeros(50), 10 * np.ones(50)], 20000, np.random.default_rng(0))
    assert abs(np.mean(idx >= 50) - 0.75) < 0.02


def test_node_parallel_matches_serial():
    """NodeParallel (persistent worker processes owning node blocks) returns the serial result to the solver tolerance
    (iterative solvers stop per block instead of per batch), keeping per-node state (warm starts, frozen Jacobians) and
    slicing a static second argument per block."""
    from implicit_field import FieldSolver
    from parallel import NodeParallel
    B = PCBasis(3, 3)
    fs = CubicField(B)
    k3 = np.stack([rand_input(B, mean=1.0, s=0.2) for _ in range(5)])  # differs per node: tests the block slicing
    H = np.stack([rand_input(B, mean=1.5, s=0.3) for _ in range(5)])
    ser = FieldSolver(fs, method="chord", init="warm", tol=1e-11)
    par = NodeParallel(FieldSolver(fs, method="chord", init="warm", tol=1e-11), len(H), 2, static=k3)
    try:
        for step in range(3):                                    # repeated calls exercise the per-node warm starts
            Hs = H * (1 + 0.01 * step)
            assert np.abs(par(Hs) - ser(Hs, k3)).max() < 1e-10
    finally:
        par.close()


if __name__ == "__main__":
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS {name} ({time.perf_counter() - t0:.1f}s)", flush=True)
        except Exception as e:                                   # report and continue
            fails += 1
            print(f"FAIL {name}: {e!r}", flush=True)
    print(f"{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
