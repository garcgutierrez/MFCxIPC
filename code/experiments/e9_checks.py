"""E9: targeted verification checks requested in the review (accuracy only; wall times are not used).

  python e9_checks.py ghconv              closure definition: f(M_u)e0 vs exact projection Pi[f(u)] as p grows;
                                          matrix-function square root vs Galerkin root (y*y = u)
  python e9_checks.py pconv_ign           ignition, d = 2: error vs PC order p = 2..6 (exact and PSP closures)
  python e9_checks.py tol                 time-integration tolerance and MC-tolerance checks
  python e9_checks.py pconv_enth <p>      heat conduction, d = 8: Galerkin (chord, 1e-10) and sparse PSP at order p
  python e9_checks.py ood_enth <Qbar>     heat conduction with a shifted source mean: network outside its training set
  python e9_checks.py spectra             spectral domain of the inverse/power closures along the toggle trajectory
  python e9_checks.py ood_explicit        generic explicit closures beyond their training radius
Results: results/e9_<check>[_arg].pkl
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from closures import Exp, SinCos, Power, ExactGalerkin, PSP, Lanczos, PINNClosure, sample_directions
from propagate import intrusive, compare, monte_carlo


def save(name, obj):
    with open(os.path.join(RES, f"e9_{name}.pkl"), "wb") as fh:
        pickle.dump(obj, fh)


# -----------------------------------------------------------------------------
def ghconv():
    """Fixed input of degree 2 in d = 2; G^(p) = f(M_u^(p)) e0 restricted to the degree <= 2 modes, against
    the exact projection of f(u_hat) onto the same modes, for growing basis order p."""
    out = {}
    B2 = PCBasis(2, 2)
    u2 = np.zeros(B2.P)
    u2[0], u2[1], u2[2] = 0.0, 0.5, 0.3
    u2[3], u2[4] = 0.2, 0.1                                    # He2(xi1)/sqrt2, xi1 xi2
    X, W = PCBasis(2, 2).quadrature(80)
    Psi2 = B2.eval_basis(X)
    uh = Psi2 @ u2
    for fname, f in [("exp", np.exp), ("sin", np.sin)]:
        H = (f(uh) * W) @ Psi2
        rows = []
        for p in range(2, 15):
            B = PCBasis(2, p)
            u = np.zeros(B.P)
            u[:B2.P] = u2                                      # graded ordering: degree <= 2 modes first
            lam, V = np.linalg.eigh(B.mult_matrix(u))
            G = V @ (f(lam) * V[0])
            Q = PSP(Exp(B), p + 1)                             # tensor Gauss rule with p+1 points
            Gp = (f(u @ Q.Psi.T) @ Q.Wpsi)
            rows.append((p, np.linalg.norm(G[:B2.P] - H) / np.linalg.norm(H),
                         np.linalg.norm(Gp[:B2.P] - H) / np.linalg.norm(H),
                         np.linalg.norm(G[:B2.P] - Gp[:B2.P]) / np.linalg.norm(H)))
            print(f"{fname} p={p:2d}  |G-H|/|H| {rows[-1][1]:.2e}  |PSP-H|/|H| {rows[-1][2]:.2e}  "
                  f"|G-PSP| {rows[-1][3]:.2e}", flush=True)
        out[fname] = np.array(rows)
    # matrix-function square root vs Galerkin root (Newton on y*y = u), positive input
    rows = []
    for p in range(2, 9):
        B = PCBasis(2, p)
        u = np.zeros(B.P)
        u[0], u[1], u[2], u[3], u[4] = 2.0, 0.4, 0.3, 0.15, 0.1
        lam, V = np.linalg.eigh(B.mult_matrix(u))
        G = V @ (np.sqrt(lam) * V[0])
        y = G.copy()
        for _ in range(50):
            y = y - np.linalg.solve(2 * B.mult_matrix(y), B.sprod(y, y) - u)
        Xq, Wq = B.quadrature(60)
        Ps = B.eval_basis(Xq)
        H = (np.sqrt(np.maximum(Ps @ u, 0)) * Wq) @ Ps
        rows.append((p, lam.min(), np.linalg.norm(G - y) / np.linalg.norm(y),
                     np.linalg.norm(B.sprod(G, G) - u) / np.linalg.norm(u),
                     np.linalg.norm(G - H) / np.linalg.norm(H), np.linalg.norm(y - H) / np.linalg.norm(H)))
        print(f"sqrt p={p} lam_min {lam.min():.3f} |G_sqrt - Newton root| {rows[-1][2]:.2e}  residual of G in y*y=u "
              f"{rows[-1][3]:.2e}  |G-H| {rows[-1][4]:.2e} |root-H| {rows[-1][5]:.2e}", flush=True)
    out["sqrt"] = np.array(rows)
    # non-associativity: 4th power
    B = PCBasis(2, 4)
    u = np.random.default_rng(3).normal(size=B.P) * 0.3 * 0.6 ** B.degree
    u[0] = 1.0
    uu = B.sprod(u, u)
    a, bq = B.sprod(uu, uu), B.sprod(u, B.sprod(u, uu))
    M4 = np.linalg.matrix_power(B.mult_matrix(u), 4)[:, 0]
    out["assoc4"] = (np.linalg.norm(a - bq) / np.linalg.norm(bq), np.linalg.norm(bq - M4) / np.linalg.norm(M4))
    print("(u*u)*(u*u) vs u*(u*(u*u)): %.2e; right-nested vs M^4 e0: %.2e" % out["assoc4"])
    save("ghconv", out)


# -----------------------------------------------------------------------------
def pconv_ign():
    from problems import Ignition
    d = 2
    z = np.load(os.path.join(RES, "e5_mc_d2p4.npz"))
    Ymc = z["Y"].astype(float)
    xi = np.random.default_rng(2028 + d).normal(size=(Ymc.shape[2], d))
    t_eval = np.linspace(0, 1.0, 21)
    rows = []
    for p in range(2, 7):
        B = PCBasis(d, p)
        prob = Ignition(B, delta=1.5)
        Psi = B.eval_basis(xi)
        nl = Exp(B)
        for name, cl in [("eig", ExactGalerkin(nl)), ("psp", PSP(nl, p + 1))]:
            C, wall, nfev, ok = intrusive(prob, cl, t_eval, rtol=1e-7, atol=1e-9)
            m = compare(B, C, Ymc, Psi)
            rows.append((p, name, float(m["l2_err"][-1].max()), float(m["mean_err_p"][-1].max()),
                         float(m["std_err_p"][-1].max())))
            print(f"p={p} {name}: max_x L2(T) {rows[-1][2]:.3e}  mean {rows[-1][3]:.2e}  sd {rows[-1][4]:.2e}",
                  flush=True)
    save("pconv_ign", rows)


def tol():
    """(a) IPC time-integration tolerance: ignition d=2 p=4 and heat conduction with sparse PSP;
    (b) MC integration tolerance on a subsample."""
    from problems import Ignition, EnthalpyHeat
    from implicit_field import CubicField, FieldPSP
    out = {}
    B = PCBasis(2, 4)
    prob = Ignition(B, delta=1.5)
    t_eval = np.linspace(0, 1.0, 21)
    C1 = intrusive(prob, ExactGalerkin(Exp(B)), t_eval, rtol=1e-7, atol=1e-9)[0]
    C2 = intrusive(prob, ExactGalerkin(Exp(B)), t_eval, rtol=1e-10, atol=1e-12)[0]
    out["ign_ipc"] = float(np.abs(C1 - C2).max() / np.abs(C2).max())
    xi = np.random.default_rng(2028 + 2).normal(size=(2000, 2))
    Y1 = monte_carlo(prob, xi, t_eval, rtol=1e-8, atol=1e-10)
    Y2 = monte_carlo(prob, xi, t_eval, rtol=1e-11, atol=1e-13)
    out["ign_mc"] = float(np.abs(Y1 - Y2).max() / np.abs(Y2).max())
    B8 = PCBasis(8, 3)
    pe = EnthalpyHeat(B8, nx=15)
    fs = CubicField(B8)
    D1 = intrusive(pe, FieldPSP(fs, "s4", pe.K3), t_eval, rtol=1e-7, atol=1e-9)[0]
    D2 = intrusive(pe, FieldPSP(fs, "s4", pe.K3), t_eval, rtol=1e-10, atol=1e-12)[0]
    out["enth_ipc"] = float(np.abs(D1 - D2).max() / np.abs(D2).max())
    xi8 = np.random.default_rng(5000 + 8).normal(size=(1000, 8))
    E1 = monte_carlo(pe, xi8, t_eval, rtol=1e-8, atol=1e-10)
    E2 = monte_carlo(pe, xi8, t_eval, rtol=1e-11, atol=1e-13)
    out["enth_mc"] = float(np.abs(E1 - E2).max() / np.abs(E2).max())
    print(out, flush=True)
    save("tol", out)


# -----------------------------------------------------------------------------
def pconv_enth(p):
    from problems import EnthalpyHeat
    from implicit_field import CubicField, FieldPSP, FieldSolver
    from implicit import cubic_root
    p = int(p)
    B = PCBasis(8, p)
    prob, fs = EnthalpyHeat(B, nx=15), CubicField(B)
    z = np.load(os.path.join(RES, "e8_mc_d8p3.npz"))
    Ymc = z["Y"].astype(float)
    xi = np.random.default_rng(5000 + 8).normal(size=(Ymc.shape[2], 8))
    Psi = B.eval_basis(xi)
    t_eval = np.linspace(0, 1.0, 21)
    out = dict(p=p, P=B.P, nnz=B.nnz, methods={})
    for name, cl in [("galerkin", FieldSolver(fs, method="chord", init="warm", tol=1e-10)),
                     (f"psp_s{p + 1}", FieldPSP(fs, f"s{p + 1}", prob.K3))]:
        C, wall, nfev, ok = intrusive(prob, cl, t_eval, rtol=1e-7, atol=1e-9)
        m = compare(B, C, Ymc, Psi)
        out["methods"][name] = dict(C=C[-1], wall=wall, nfev=nfev, ok=ok, l2=float(m["l2_err"][-1].max()),
                                    bc=float(m["bc"][-1, prob.mid]), mean_p=float(m["mean_err_p"][-1].max()),
                                    std_p=float(m["std_err_p"][-1].max()))
        print(f"p={p} P={B.P} {name}: max_x L2(h,1) {out['methods'][name]['l2']:.3e} BC {out['methods'][name]['bc']:.4f}"
              f"  (wall {wall:.0f}s, nfev {nfev})", flush=True)
    save(f"pconv_enth_p{p}", out)


def ood_enth(Qbar):
    """Network trained for Q = 15 used for a different source amplitude; reference: converged Galerkin."""
    from problems import EnthalpyHeat
    from implicit_field import CubicField, FieldSolver, FieldPINN
    Qbar = float(Qbar)
    B = PCBasis(8, 3)
    prob, fs = EnthalpyHeat(B, nx=15, Q=Qbar), CubicField(B)
    pn = FieldPINN(fs, width=128, depth=3).load_weights(os.path.join(MODELS, "enthalpy_d8p3.npz"))
    t_eval = np.linspace(0, 1.0, 21)
    ref = FieldSolver(fs, method="chord", init="warm", tol=1e-10)
    Cref = intrusive(prob, ref, t_eval, rtol=1e-7, atol=1e-9)[0]
    out = dict(Qbar=Qbar, methods={}, Tmid_ref=None)
    nrm = np.linalg.norm(Cref, axis=2) + 1e-300
    for name, cl in [("pinn", pn),
                     ("pinn_nk2", FieldSolver(fs, method="nk", init=pn, fixed_steps=2, m=5, record=True)),
                     ("pinn_nkc8", FieldSolver(fs, method="nk", init=pn, tol=1e-8, m=10, record=True)),
                     ("chord_w8", FieldSolver(fs, method="chord", init="warm", tol=1e-8, record=True))]:
        C, wall, nfev, ok = intrusive(prob, cl, t_eval, rtol=1e-7, atol=1e-9)
        d = dict(ok=ok, nfev=nfev)
        if ok:
            e = np.linalg.norm(C - Cref, axis=2) / nrm
            d.update(coef_err=float(np.nanmax(e[1:])), iters=float(np.mean(getattr(cl, "iters", [np.nan]) or [np.nan])))
            if hasattr(cl, "init_res") and cl.init_res:
                d.update(init_res_med=float(np.median(cl.init_res)), final_res_max=float(np.max(cl.final_res)))
        out["methods"][name] = d
        print(f"Q={Qbar} {name}: {d}", flush=True)
    # inputs visited, relative to the training anchors
    H = Cref.reshape(-1, B.P)
    out["h_norm_max"] = float(np.linalg.norm(H, axis=1).max())
    save(f"ood_enth_Q{Qbar:g}", out)


# -----------------------------------------------------------------------------
def spectra():
    """Smallest eigenvalues of the multiplication matrices fed to the power and inverse closures."""
    from problems import ToggleSwitch
    o = pickle.load(open(os.path.join(RES, "e4_toggle.pkl"), "rb"))
    B = PCBasis(4, 3)
    prob = ToggleSwitch(B, y0=(10.0, 0.5))
    C = o["methods"]["eig"]["C"]
    V = C[:, 1, :]
    lam_v = np.linalg.eigvalsh(B.mult_matrix(V))
    den = B.e0 + ExactGalerkin(Power(B, prob.beta))(V)
    lam_d = np.linalg.eigvalsh(B.mult_matrix(den))
    out = dict(v_min=float(lam_v[:, 0].min()), v_min_rel=float((lam_v[:, 0] / V[:, 0]).min()),
               den_min=float(lam_d[:, 0].min()))
    print(out, flush=True)
    save("spectra", out)


def ood_explicit():
    """Generic closures evaluated on directions of the training family but radii beyond r_lim."""
    from e1_closures import CASES
    out = {}
    for name, d, p, mk, R_eff, K in CASES:
        B = PCBasis(d, p)
        nl = mk(B)
        pc = PINNClosure.load(nl, os.path.join(MODELS, f"{name}.npz"))
        rng = np.random.default_rng(99)
        D = sample_directions(B, 2000, rng)
        rl = pc.r_limit(D)
        rows = []
        for fac in [0.5, 1.0, 1.25, 1.5]:
            r = fac * rl
            X = r[:, None] * D
            if nl.c != 0:                                    # powers: stay inside the domain of definition
                lam = np.linalg.eigvalsh(B.mult_matrix(X))[:, 0]
                keep = lam > -0.95
                X, D2, r2 = X[keep], D[keep], r[keep]
            else:
                D2, r2 = D, r
            Ge = ExactGalerkin(nl).normalised(X)
            e = np.linalg.norm(pc.G(r2, D2) - Ge, axis=1) / np.linalg.norm(Ge, axis=1)
            rows.append((fac, float(np.median(e)), float(np.percentile(e, 95)), len(e)))
            print(f"{name} r = {fac} r_lim: median {rows[-1][1]:.2e} p95 {rows[-1][2]:.2e} (n={len(e)})", flush=True)
        out[name] = rows
    save("ood_explicit", out)


def metrics_ode():
    """Recompute the MC references of the pendulum and the toggle switch (same samples) and store the paired
    moment errors (and MC standard errors) of every stored propagation in the result pickles."""
    from problems import Pendulum, ToggleSwitch
    for fname, mk, seed, d, p in [("e3_pendulum.pkl", lambda B: Pendulum(B), 2026, 2, 6),
                                  ("e4_toggle.pkl", lambda B: ToggleSwitch(B, y0=(10.0, 0.5)), 2027, 4, 3)]:
        path = os.path.join(RES, fname)
        out = pickle.load(open(path, "rb"))
        B = PCBasis(d, p)
        prob = mk(B)
        xi = np.random.default_rng(seed).normal(size=(100_000, d))
        Y = monte_carlo(prob, xi, out["t"])
        Psi = B.eval_basis(xi)
        for name, m in out["methods"].items():
            if "C" not in m:
                continue
            C = m["C"]
            new = compare(B, C, Y[:len(C)], Psi)
            for key in ("mean_err_p", "std_err_p", "se_mean", "se_std"):
                m[key] = new[key]
            m["l2_check"] = float(np.nanmax(np.abs(new["l2_err"] - m["l2_err"][:len(C)])))
            print(f"{fname} {name}: L2 reproduced to {m['l2_check']:.1e}; paired mean/sd at end "
                  f"{new['mean_err_p'][-1].max():.1e}/{new['std_err_p'][-1].max():.1e} "
                  f"(old {m['mean_err'][len(C) - 1].max():.1e}/{m['std_err'][len(C) - 1].max():.1e})", flush=True)
        pickle.dump(out, open(path, "wb"))


def own_traj():
    """Closure error of the network-based closures along their OWN trajectories (heat conduction), compared with
    the error along the reference trajectory, and coefficient deviation of every run from the converged solution."""
    from problems import EnthalpyHeat
    from implicit_field import CubicField, FieldNewton, FieldPINN, FieldSolver
    B = PCBasis(8, 3)
    prob, fs = EnthalpyHeat(B, nx=15), CubicField(B)
    pn = FieldPINN(fs, width=128, depth=3).load_weights(os.path.join(MODELS, "enthalpy_d8p3.npz"))
    newton = FieldNewton(fs)
    out = {}
    for name, cl in [("pinn", pn), ("pinn_nk1", FieldSolver(fs, method="nk", init=pn, fixed_steps=1, m=5)),
                     ("pinn_nk2", FieldSolver(fs, method="nk", init=pn, fixed_steps=2, m=5))]:
        r = np.load(os.path.join(RES, f"e8_run_d8p3_{name}.npz"))
        H = r["C"][1:].reshape(-1, B.P)
        K = np.tile(prob.K3, (r["C"].shape[0] - 1, 1))
        ref = newton(H, K)
        e = np.linalg.norm(cl(H, K) - ref, axis=1) / np.linalg.norm(ref, axis=1)
        res = np.linalg.norm(fs.residual(H, K, cl(H, K)), axis=1) / np.linalg.norm(H, axis=1)
        out[name] = dict(err_med=float(np.median(e)), err_max=float(e.max()), res_med=float(np.median(res)),
                         res_max=float(res.max()))
        print(f"{name} along own trajectory: closure err {np.median(e):.2e}/{e.max():.2e}, residual "
              f"{np.median(res):.2e}/{res.max():.2e}", flush=True)
    save("own_traj", out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    t0 = time.perf_counter()
    fn = dict(ghconv=ghconv, pconv_ign=pconv_ign, tol=tol, pconv_enth=pconv_enth, ood_enth=ood_enth,
              spectra=spectra, ood_explicit=ood_explicit, metrics_ode=metrics_ode, own_traj=own_traj)[cmd]
    fn(*sys.argv[2:])
    print(f"done {cmd} {sys.argv[2:]} in {time.perf_counter() - t0:.0f}s")
