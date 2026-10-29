"""Tests of gpu_backend.py against the library (N = 60 nodes, three call sequences per method).

Run:  python tests/test_gpu.py            (needs CuPy and a GPU; the NumPy path of the new code is tested as well)

For every method of the benchmark the outputs of the new code on the GPU and on the CPU (xp = numpy) must match the library's
CPU outputs to a relative 1e-10 (per node, in the 2-norm) on three call sequences, and the iteration counts must agree.
Sequences (indices of the S = 2000 interpolated calls, see experiments/e12_scaling.py): A = the first 12 calls, B = 20 calls
at stride 97 over the whole trajectory (large jumps between calls, so that the solvers iterate), C = the last 12 calls.
Methods without a library counterpart (active-set variants, the vectorised GMRES) are checked against the converged
reference and against the original GMRES.

Two documented exceptions on the GPU, both a consequence of floating-point summation order and not of a different algorithm:
  * solvers that stop on a tolerance (newton_w8, nk_w8, chord_w8, chord_w10): the stopping test and, for the chord solvers, the
    refresh rule (largest contraction ratio over the batch, dominated by nodes whose residual is at the roundoff floor) can
    flip on a call.  The two runs then follow different iteration paths that both stop at ||R|| <= tol ||h||, so the outputs
    agree to the solver tolerance (limit: 50 tol), and the iteration counts may differ by one or two on a few calls of the
    large-jump sequence B (reported).  On the CPU path of the new code (xp = numpy) outputs and iteration counts equal
    the library's;
  * pseudo-spectral projection on the Smolyak rules (negative weights): the absolute error is about 1e-14 (cancellation in
    the quadrature sums), so nodes whose solution is 1e-4 of the largest one show a per-node relative difference up to 1e-9.
    For these two methods the difference is measured relative to the largest node of the call.
"""
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))
import numpy as np

import e12_scaling as e12
import gpu_backend as gb
from pce import PCBasis
from closures import Exp
from implicit import gmres_batched
from implicit_field import CubicField, FieldNewton, FieldSolver

N = 60
SEQS = {"A": list(range(0, 12)), "B": list(range(0, 1940, 97)), "C": list(range(1988, 2000))}
TOL = 1e-10
HAVE_GPU = gb.cp is not None
DEVICES = ["cpu1"] + (["gpu"] if HAVE_GPU else [])
_SEQ = {}


def sequence(problem):
    if problem not in _SEQ:
        _SEQ[problem] = e12.Sequence(problem, N)
    return _SEQ[problem]


def rel(a, b):
    return float(np.max(np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-300)))


def run_closure(call, seq, calls, to_dev, xp):
    outs = []
    for s in calls:
        outs.append(gb.to_host(call(to_dev(seq.h(s)))))
        gb.sync(xp)
    return outs


def library_outputs(problem, method, seq, calls):
    """Library closure on the call sequence: (outputs, per-call iteration counts or None)."""
    B = PCBasis(e12.D, e12.P_ORDER)
    if problem == "heat":
        import e8_enthalpy as e8
        fs = CubicField(B)
        if method in gb.HEAT_FIXED:
            cl = FieldSolver(fs, method="chord", init="warm", fixed_steps=gb.HEAT_FIXED[method], m=10, record=True)
        else:
            cl = e8.make_closure(method, e12.LibK3(seq.K3), fs)
        outs = [cl(seq.h(s), seq.K3) for s in calls]
    else:
        import e6_ignition_highd as e6
        cl = e6.make_closure(method, B)
        outs = [cl(seq.h(s)) for s in calls]
    its = list(getattr(cl, "iters", None) or [])
    return outs, (its if its else None)


def new_outputs(problem, method, seq, calls, device):
    call, cl, _ = e12.build(problem, method, N, device, seq)
    xp = gb.get_xp(device)
    outs = run_closure(call, seq, calls, (lambda h: xp.asarray(h)) if device == "gpu" else (lambda h: h), xp)
    its = list(getattr(cl, "iters", None) or [])
    return outs, (its if its else None), cl


def rel_scaled(a, b):
    """Difference relative to the largest node of the call."""
    return float(np.max(np.linalg.norm(a - b, axis=1)) / max(np.max(np.linalg.norm(b, axis=1)), 1e-300))


SPARSE_PSP = ("psp_s3", "psp_s4")
SOLVER_TOL = {m: v[2] for m, v in gb.HEAT_SOLVERS.items()}


def check_method(problem, method):
    seq = sequence(problem)
    worst, notes = 0.0, []
    for name, calls in SEQS.items():
        lib, lib_its = library_outputs(problem, method, seq, calls)
        for dev in DEVICES:
            new, new_its, _ = new_outputs(problem, method, seq, calls, dev)
            measure = rel_scaled if method in SPARSE_PSP else rel
            err = max(measure(a, b) for a, b in zip(new, lib))
            same = lib_its is None or new_its is None or lib_its == new_its
            if not same:
                notes.append(f"seq {name} {dev}: iterations differ on {sum(a != b for a, b in zip(new_its, lib_its))} of "
                             f"{len(lib_its)} calls (max difference {max(abs(a - b) for a, b in zip(new_its, lib_its))})")
                assert dev != "cpu1", f"{method} {notes[-1]}"
            limit = 50 * SOLVER_TOL[method] if (method in SOLVER_TOL and dev != "cpu1") else TOL
            worst = max(worst, err)
            assert err < limit, f"{problem}/{method} seq {name} {dev}: relative difference {err:.2e} (limit {limit:.0e})"
    return worst, notes


def test_vectorised_gmres():
    """gmres_vec = implicit.gmres_batched (1e-12) on the Jacobian of the cubic law, NumPy and CuPy."""
    B = PCBasis(e12.D, e12.P_ORDER)
    fs = CubicField(B)
    seq = sequence("heat")
    h, k3 = seq.h(1500), seq.K3
    y = h + 0.01 * np.random.default_rng(1).normal(size=h.shape) * np.abs(h).max()
    R = fs.residual(h, k3, y)
    for m in (5, 10):
        ref = gmres_batched(lambda v: fs.jv(k3, y, v), R, m)
        for dev in DEVICES:
            xp = gb.get_xp(dev)
            ops = gb.Ops(B, xp)
            f = gb.CubicFieldX(ops)
            yd, kd, Rd = ops.asarray(y), ops.asarray(k3), ops.asarray(R)
            out = gb.to_host(gb.gmres_vec(xp, lambda v: f.jv(kd, yd, v), Rd, m))
            err = rel(out, ref)
            assert err < 1e-12, f"gmres_vec m={m} {dev}: {err:.2e}"
            print(f"  gmres_vec m={m} {dev}: {err:.1e}")


def test_active_set_variants():
    """Active-set solvers reach the tolerance of their base solver; the mean node iterates no more than the slowest node."""
    B = PCBasis(e12.D, e12.P_ORDER)
    seq = sequence("heat")
    for method, base, tol in (("chord_w8_as", "chord_w8", 1e-8), ("chord_w10_as", "chord_w10", 1e-10),
                              ("nk_w8_as", "nk_w8", 1e-8)):
        for dev in DEVICES:
            xp = gb.get_xp(dev)
            calls = SEQS["B"]
            ref = [FieldNewton(CubicField(B))(seq.h(s), seq.K3) for s in calls]
            out, its, cl = new_outputs("heat", method, seq, calls, dev)
            _, its0, _ = new_outputs("heat", base, seq, calls, dev)
            err = max(rel(a, b) for a, b in zip(out, ref))
            assert err < 20 * tol, f"{method} {dev}: error {err:.2e} against tolerance {tol}"
            assert all(m_ <= m for m_, m in zip(cl.iters_mean, its))
            print(f"  {method} {dev}: error vs Newton {err:.1e}; max iters {its[:6]}.. mean-node {np.round(cl.iters_mean[:6], 2)}"
                  f"; batch solver {its0[:6]}..")


def make_test(problem, method):
    def t():
        worst, notes = check_method(problem, method)
        print(f"  {problem}/{method}: worst relative difference {worst:.1e}" + ("; " + "; ".join(notes) if notes else ""))
    t.__name__ = f"test_{problem}_{method.replace('@', '_')}"
    return t


TESTS = [test_vectorised_gmres, test_active_set_variants]
LIBRARY_HEAT = [m for m in gb.HEAT_METHODS if m not in gb.HEAT_ACTIVE]
TESTS += [make_test("heat", m) for m in LIBRARY_HEAT] + [make_test("ign", m) for m in gb.IGN_METHODS]
for _t in TESTS:                                          # let pytest collect the generated tests
    globals()[_t.__name__] = _t

if __name__ == "__main__":
    fails = 0
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"FAIL {t.__name__}: {e}", flush=True)
    print(f"{len(TESTS) - fails}/{len(TESTS)} passed" + ("" if HAVE_GPU else "  (no CuPy: GPU tests skipped)"))
    sys.exit(1 if fails else 0)
