"""E12: node scaling and GPU benchmark of the closures (GPU_SCALING_TASK.md).

    python e12_scaling.py env                                         hardware / software / measured GEMM rates
    python e12_scaling.py inputs [heat|ign]                           build and check the interpolated call sequences, cache references
    python e12_scaling.py run <problem> <method> <N> <device> [--calls S'] [--window first|late] [--budget s]
    python e12_scaling.py collect                                     results/e12_scaling.{json,md,pdf}, e12_paragraph.tex

problem: heat | ign.   device: cpu1 (new code, one thread) | cpu1orig (library code, one thread, N <= 1000) |
cpuall (parallel.NodeParallel, one process per block of nodes, one BLAS thread each) | gpu (CuPy, inputs already on the GPU) |
gpux (CuPy, inputs copied to the GPU and the result back at every call) | gpu32 (float32 side result).
Every run writes results/e12_<problem>_<method>_<N>_<device>[_late].json and appends to results/logs/e12_*.log.

Call sequence: the stored trajectory (heat: results/e8_enthalpy.pkl chord_w10, 15 nodes; ignition: e6_ignition_highd.pkl eig,
31 nodes) is interpolated in space onto N interior nodes with a cubic spline through the zero Dirichlet values and
linearly in time between the 21 stored instants, S = 2000 closure calls.  window=first runs calls 0..S'-1 (the task rule);
window=late runs the last S' calls after one untimed priming call (same states, harder part of the trajectory).
"""
import os
import sys

# one BLAS thread everywhere (the multi-core CPU device forks one single-threaded process per block of nodes)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import glob
import json
import pickle
import platform
import resource
import subprocess
import time

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, MODELS
from pce import PCBasis
from closures import Exp, ExactGalerkin
from implicit_field import CubicField
import gpu_backend as gb

LOGDIR = os.path.join(RES, "logs")
CACHE = os.environ.get("E12_CACHE", os.path.join(os.path.expanduser("~"), "e12_cache"))
S_TOTAL, CHECK_EVERY, NT = 2000, 50, 21
D, P_ORDER = 8, 3
SRC = dict(heat=dict(pkl="e8_enthalpy.pkl", key="chord_w10", nx=15, model="enthalpy_d8p3.npz"),
           ign=dict(pkl="e6_ignition_highd.pkl", key="eig", nx=31, model="exp_d8p3_ignition.npz"))
SIZES = [None, 60, 250, 1000, 4000, 16000]             # None = the stored mesh (15 for heat, 31 for ignition)
DEVICES = ["cpu1", "cpu1orig", "cpuall", "gpu", "gpux", "gpu32"]
GPU_WORK = 1.0e9                                       # bytes of temporaries per chunk on the GPU (CPU: 256e6)
MIN_CALLS = 2
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)


def log(msg, name=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if name:
        with open(os.path.join(LOGDIR, f"e12_{name}.log"), "a") as fh:
            fh.write(line + "\n")


def nodes(problem, N):
    return SRC[problem]["nx"] if N is None else int(N)


# =============================================================================
# Call sequences
# =============================================================================
def stored_trajectory(problem):
    f = os.path.join(CACHE, f"traj_{problem}.npy")
    if os.path.exists(f):
        return np.load(f)
    with open(os.path.join(RES, SRC[problem]["pkl"]), "rb") as fh:
        C = pickle.load(fh)["methods"][SRC[problem]["key"]]["C"]
    np.save(f, C)
    return C


class Sequence:
    """Interpolated states h_s (heat: enthalpy, ignition: temperature) at N nodes and the constant k3 field (heat)."""

    def __init__(self, problem, N):
        self.problem, self.N = problem, N
        C = stored_trajectory(problem)                                  # (21, nx, P)
        nx = C.shape[1]
        self.P = C.shape[2]
        xs = np.linspace(0, 1, nx + 2)
        Y = np.zeros((nx + 2, NT * self.P))
        Y[1:-1] = C.transpose(1, 0, 2).reshape(nx, -1)
        x = np.linspace(0, 1, N + 2)[1:-1]
        self.Hs = CubicSpline(xs, Y)(x).reshape(N, NT, self.P).transpose(1, 0, 2).copy()   # (21, N, P)
        self.K3 = self._k3() if problem == "heat" else None

    def _k3(self):
        f = os.path.join(CACHE, f"k3_{self.N}.npy")
        if os.path.exists(f):
            return np.load(f)
        from problems import EnthalpyHeat
        K3 = EnthalpyHeat(PCBasis(D, P_ORDER), nx=self.N).K3
        np.save(f, K3)
        return K3

    def h(self, s, S=S_TOTAL):
        tau = s / (S - 1) * (NT - 1)
        i = min(int(tau), NT - 2)
        w = tau - i
        return (1.0 - w) * self.Hs[i] + w * self.Hs[i + 1]


def checked_calls(first, last):
    """Calls (global indices) at which accuracy is checked: every CHECK_EVERY-th call (or, for short sequences,
    about five calls per sequence)."""
    n = last - first
    every = CHECK_EVERY if n >= 4 * CHECK_EVERY else max(1, n // 5)
    return [s for s in range(first, last) if (s + 1) % every == 0]


# =============================================================================
# References: converged solutions at the checked calls
# =============================================================================
def reference(problem, N, seq, s):
    d = os.path.join(CACHE, f"ref_{problem}_{N}")
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, f"s{s}.npy")
    if os.path.exists(f):
        return np.load(f)
    B = PCBasis(D, P_ORDER)
    h = seq.h(s)
    if problem == "heat":
        if N <= 250:
            from implicit_field import FieldNewton
            ref = FieldNewton(CubicField(B))(h, seq.K3)                      # cold start, tol 1e-12 (library)
        else:                                                              # same algorithm on the GPU, in chunks
            ops = gb.Ops(B, gb.cp, work_bytes=GPU_WORK)
            sol = gb.FieldSolverX(ops, method="newton", init="linear", tol=1e-12, maxit=40)
            ref = gb.to_host(sol(ops.asarray(h), ops.asarray(seq.K3)))
    else:
        ref = ExactGalerkin(Exp(B))(h)
    np.save(f, ref)
    return ref


def rel_err(out, ref):
    nr = np.linalg.norm(ref, axis=1)
    ok = nr > 1e-14
    return (np.linalg.norm(out - ref, axis=1)[ok] / nr[ok])


# =============================================================================
# Closure construction on a device
# =============================================================================
class LibK3:                                            # what e8_enthalpy.make_closure needs from the problem
    def __init__(self, K3):
        self.K3 = K3


def build(problem, method, N, device, seq):
    """Returns (call, closure, extras): call(h_host_or_dev) -> output; closure carries counters and iteration lists."""
    B = PCBasis(D, P_ORDER)
    model = os.path.join(MODELS, SRC[problem]["model"])
    notes = {}
    if device == "cpu1orig":
        if problem == "heat":
            import e8_enthalpy as e8
            fs = CubicField(B)
            if method in gb.HEAT_ACTIVE:
                raise ValueError("no library counterpart")
            if method in gb.HEAT_FIXED:
                from implicit_field import FieldSolver
                cl = FieldSolver(fs, method="chord", init="warm", fixed_steps=gb.HEAT_FIXED[method], m=10, record=True)
            else:
                cl = e8.make_closure(method, LibK3(seq.K3), fs)
            return (lambda h: cl(h, seq.K3)), cl, notes
        import e6_ignition_highd as e6
        cl = e6.make_closure(method, B)
        return (lambda u: cl(u)), cl, notes
    xp = gb.get_xp(device)
    dtype = np.float32 if device == "gpu32" else np.float64
    ops = gb.Ops(B, xp, dtype=dtype, work_bytes=GPU_WORK if xp is not np else 256e6)
    if problem == "heat":
        cl = gb.build_heat(method, ops, seq.K3, model, CubicField(B))
        if isinstance(cl, gb.FieldPSPX):
            notes["k3q_cached"] = bool(cl.cached)
        K3d = ops.asarray(seq.K3)
        if isinstance(cl, gb.FieldSolverX):
            notes["mk3_cached"] = bool(cl._mk3(K3d) is not None)
        if device == "cpuall":
            from parallel import NodeParallel
            W = min(os.cpu_count() or 1, N)
            par = NodeParallel(cl, N, W, static=seq.K3)
            notes["workers"] = W
            return (lambda h: par(h)), par, notes
        return (lambda h: cl(h, K3d)), cl, notes
    cl = gb.build_ignition(method, ops, model, Exp(B))
    if device == "cpuall":
        from parallel import NodeParallel
        W = min(os.cpu_count() or 1, N)
        par = NodeParallel(cl, N, W)
        notes["workers"] = W
        return (lambda u: par(u)), par, notes
    return (lambda u: cl(u)), cl, notes


# =============================================================================
# One timed run
# =============================================================================
def run_sequence(problem, method, N, device, seq, first, last, prime, checks):
    """One repetition: fresh closure, optional untimed priming call, timed calls first..last-1.
    Returns dict(times, outputs at checks, counters, iterations, notes, calls_done)."""
    xp = gb.get_xp(device)
    gpu = device.startswith("gpu")
    xfer = device == "gpux"
    dt = np.float32 if device == "gpu32" else np.float64
    call, cl, notes = build(problem, method, N, device, seq)
    if prime is not None:
        h = seq.h(prime)
        call(xp.asarray(h, dtype=dt) if gpu else h)
        gb.sync(xp)
    n0 = len(getattr(cl, "iters", None) or [])
    c0 = dict(getattr(cl, "n", {}))
    c0n = dict(getattr(cl, "n_nodes", {}))
    times, outs = [], {}
    budget = BUDGET[0]
    for s in range(first, last):
        h = seq.h(s)
        if gpu:
            hd = xp.asarray(h, dtype=dt)
            if not xfer:
                gb.sync(xp)
        gb.sync(xp)
        t0 = time.perf_counter()
        if xfer:
            hd = xp.asarray(h, dtype=dt)
        out = call(hd if gpu else h)
        if xfer:
            outh = gb.to_host(out)
        gb.sync(xp)
        times.append(time.perf_counter() - t0)
        if s in checks:
            outs[s] = np.asarray(gb.to_host(out), dtype=np.float64)
        if sum(times) > budget and len(times) >= MIN_CALLS:
            break
    its = list(getattr(cl, "iters", None) or [])[n0:]
    itm = list(getattr(cl, "iters_mean", None) or [])[n0:]
    cnt = {k: v - c0[k] for k, v in getattr(cl, "n", {}).items()} if hasattr(cl, "n") else {}
    cntn = {k: v - c0n[k] for k, v in getattr(cl, "n_nodes", {}).items()} if hasattr(cl, "n_nodes") else {}
    if device == "cpuall":
        cl.close()
    return dict(times=times, outs=outs, iters=its, iters_mean=itm, counters=cnt, node_counters=cntn, notes=notes)


BUDGET = [600.0]


def stage_run(problem, method, N, device, calls=None, window="first", budget=600.0, reps=3):
    BUDGET[0] = budget
    S = S_TOTAL
    Sp = S if calls is None else int(calls)
    first, last = (0, Sp) if window == "first" else (S - Sp, S)
    prime = None if window == "first" else first - 1
    name = f"{problem}_{method}_{N}_{device}" + ("" if window == "first" else "_late")
    log(f"RUN {name}: calls {first}..{last - 1} ({Sp} of {S}), budget {budget:.0f} s", name)
    seq = Sequence(problem, N)
    checks = set(checked_calls(first, last))
    xp = gb.get_xp(device)
    # warm-up: one call of a throw-away instance (library imports, kernels, cuBLAS handles, memory pool)
    wcall, wcl, _ = build(problem, method, N, device, seq)
    h0 = seq.h(first if prime is None else prime)
    dt = np.float32 if device == "gpu32" else np.float64
    wcall(xp.asarray(h0, dtype=dt) if device.startswith("gpu") else h0)
    gb.sync(xp)
    if device == "cpuall":
        wcl.close()
    del wcall, wcl
    results = []
    t_all = time.perf_counter()
    for r in range(reps):
        res = run_sequence(problem, method, N, device, seq, first, last, prime, checks)
        results.append(res)
        tot = sum(res["times"])
        log(f"  rep {r}: {len(res['times'])} calls, {tot:.3f} s, {1e3 * tot / len(res['times']):.3f} ms/call", name)
        if tot > 60 or len(res["times"]) < last - first:
            break
    r0 = results[0]
    done = len(r0["times"])
    walls = [sum(r["times"]) for r in results if len(r["times"]) == done]
    per_call = [w / done for w in walls]
    rec = dict(problem=problem, method=method, N=N, device=device, window=window, S_total=S, S_used=Sp, first_call=first,
               calls_done=done, truncated=bool(done < last - first), budget_s=budget, reps=len(walls),
               wall_s=float(np.median(walls)), wall_all=walls, t_call_s=float(np.median(per_call)),
               t_call_median_s=float(np.median(r0["times"])), t_node_call_us=float(np.median(per_call) / N * 1e6),
               counters=r0["counters"], node_counters=r0["node_counters"], notes=r0["notes"],
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               child_rss_mb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
               gpu_pool_mb=gb.pool_bytes() / 2 ** 20 if device.startswith("gpu") else None)
    if r0["iters"]:
        it = np.array(r0["iters"], float)
        rec["iters_max_mean"], rec["iters_max_max"] = float(it.mean()), float(it.max())
        rec["iters_calls"] = [int(v) for v in it[:200]]
    if r0["iters_mean"]:
        rec["iters_node_mean"] = float(np.mean(r0["iters_mean"]))
    # accuracy against the converged reference
    errs = {}
    for s, out in r0["outs"].items():
        errs[s] = rel_err(out, reference(problem, N, seq, s))
    if errs:
        allv = np.concatenate(list(errs.values()))
        rec["err_median"], rec["err_max"] = float(np.median(allv)), float(allv.max())
        rec["err_checked_calls"] = sorted(errs)
        rec["err_per_call_median"] = [float(np.median(errs[s])) for s in sorted(errs)]
        rec["err_per_call_max"] = [float(np.max(errs[s])) for s in sorted(errs)]
        rec["err_finite"] = bool(np.isfinite(allv).all())
    path = os.path.join(RES, f"e12_{name}.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    log(f"DONE {name}: {1e3 * rec['t_call_s']:.3f} ms/call, {rec['t_node_call_us']:.2f} us/node/call, "
        f"err median {rec.get('err_median', float('nan')):.2e} max {rec.get('err_max', float('nan')):.2e}, "
        f"iters(max) {rec.get('iters_max_mean', float('nan')):.2f}, rss {rec['peak_rss_mb']:.0f} MB, "
        f"pool {rec['gpu_pool_mb']}", name)
    return rec


# =============================================================================
# env / inputs
# =============================================================================
def gemm_rate(xp, dtype, n=2048, reps=5):
    a = xp.asarray(np.random.default_rng(0).normal(size=(n, n)), dtype=dtype)
    b = a.T.copy()
    a @ b
    gb.sync(xp)
    t = time.perf_counter()
    for _ in range(reps):
        a @ b
    gb.sync(xp)
    return 2 * n ** 3 * reps / (time.perf_counter() - t) / 1e9


def stage_env():
    env = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), platform=platform.platform(), python=platform.python_version(),
               numpy=np.__version__)
    import scipy
    env["scipy"] = scipy.__version__
    env["cpu"] = next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    env["cpu_threads"] = os.cpu_count()
    env["ram_gb"] = round(int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))) / 2 ** 20, 1)
    env["blas_threads_used"] = 1
    env["cpu_dgemm_gflops_1thread"] = round(gemm_rate(np, np.float64), 1)
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(os.cpu_count()):
            env["cpu_dgemm_gflops_allthreads"] = round(gemm_rate(np, np.float64, n=4096, reps=3), 1)
    except ImportError:
        env["cpu_dgemm_gflops_allthreads"] = None
    if gb.cp is not None:
        cp = gb.cp
        pr = cp.cuda.runtime.getDeviceProperties(0)
        env.update(cupy=cp.__version__, gpu=pr["name"].decode(), gpu_mem_gb=round(pr["totalGlobalMem"] / 2 ** 30, 2),
                   gpu_cc=f"{pr['major']}.{pr['minor']}", gpu_sms=pr["multiProcessorCount"],
                   cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cuda_driver=cp.cuda.runtime.driverGetVersion())
        try:
            env["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            pass
        env["gpu_dgemm_gflops"] = round(gemm_rate(cp, np.float64, n=4096, reps=5), 1)
        env["gpu_sgemm_gflops"] = round(gemm_rate(cp, np.float32, n=4096, reps=5), 1)
        x = cp.ones(2 ** 26, dtype=np.float64)
        y = x + 1.0
        gb.sync(cp)
        t = time.perf_counter()
        for _ in range(10):
            y = x + 1.0
        gb.sync(cp)
        env["gpu_stream_gbs"] = round(10 * 16 * x.size / (time.perf_counter() - t) / 1e9, 1)
    with open(os.path.join(RES, "e12_env.json"), "w") as fh:
        json.dump(env, fh, indent=1)
    print(json.dumps(env, indent=1))


def stage_inputs(problems):
    """Interpolated sequences: input-norm ranges against the stored trajectories; references for the checked calls."""
    out = {}
    for problem in problems:
        C = stored_trajectory(problem)
        nrm = np.linalg.norm(C, axis=2)
        stored = dict(nx=C.shape[1], norm_min_nonzero_t=float(nrm[1:].min()), norm_max=float(nrm.max()),
                      norm_dyn_max=float(np.linalg.norm(C[:, :, 1:], axis=2).max()))
        out[problem] = dict(stored=stored, N={})
        for N in [nodes(problem, n) for n in SIZES]:
            t0 = time.time()
            seq = Sequence(problem, N)
            ns = np.linalg.norm(seq.Hs, axis=2)
            rec = dict(norm_max=float(ns.max()), norm_min_nonzero_t=float(ns[1:].min()),
                       norm_dyn_max=float(np.linalg.norm(seq.Hs[:, :, 1:], axis=2).max()),
                       time_continuity=float(np.abs(np.diff(seq.Hs, axis=0)).max()))
            if problem == "heat":
                rec["k3_norm_range"] = [float(np.linalg.norm(seq.K3, axis=1).min()), float(np.linalg.norm(seq.K3, axis=1).max())]
            out[problem]["N"][str(N)] = rec
            log(f"inputs {problem} N={N}: norm max {rec['norm_max']:.3f} (stored {stored['norm_max']:.3f}), "
                f"dyn max {rec['norm_dyn_max']:.3f} (stored {stored['norm_dyn_max']:.3f})  [{time.time() - t0:.1f}s]",
                "inputs")
    with open(os.path.join(RES, "e12_inputs.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def stage_refs(problem, N, first, last):
    seq = Sequence(problem, N)
    for s in checked_calls(first, last):
        t0 = time.time()
        reference(problem, N, seq, s)
        log(f"reference {problem} N={N} call {s}  [{time.time() - t0:.1f}s]", "inputs")


# =============================================================================
# collect: results/e12_scaling.{json,md,pdf}
# =============================================================================
ERR_FLOOR = 1e-10      # errors below the tolerance of every classical solver count as exact in the accuracy comparison
NETWORK = {"heat": ("pinn", "pinn_nk1", "pinn_nk2"), "ign": ("pinn",)}
TARGET = {"heat": "pinn_nk2", "ign": "pinn"}
METHOD_ORDER = {"heat": gb.HEAT_METHODS, "ign": gb.IGN_METHODS}
DEV_ORDER = ["cpu1", "cpu1orig", "cpuall", "gpu", "gpux", "gpu32"]
DEV_TEXT = {"cpu1": "CPU, 1 thread", "cpu1orig": "CPU, 1 thread, library code (original GMRES)",
            "cpuall": "CPU, all cores (node-parallel processes)", "gpu": "GPU, device-resident inputs",
            "gpux": "GPU, inputs and result copied at every call", "gpu32": "GPU, float32 (side result)"}
PROBLEM_TEXT = {"heat": "heat conduction (implicit law, d = 8, p = 3, P = 165)",
                "ign": "solid-fuel ignition (exp u, d = 8, p = 3, P = 165)"}
FAMILY = {"chord": "#2a78d6", "nk": "#1baf7a", "psp": "#eb6834", "net": "#e34948", "eig": "#2a78d6", "lanczos": "#1baf7a",
          "taylor": "#6250d6"}


def family(problem, m):
    if m in NETWORK[problem]:
        return "net"
    for k in ("chord", "psp", "lanczos", "taylor", "eig"):
        if m.startswith(k):
            return k
    return "nk"                                           # nk_w8, nk_w8_as, newton_w8


def _load_records():
    recs = []
    for f in sorted(glob.glob(os.path.join(RES, "e12_*.json"))):
        if os.path.basename(f) in ("e12_env.json", "e12_inputs.json", "e12_scaling.json", "e12_endtoend.json"):
            continue
        recs.append(json.load(open(f)))
    return recs


def _cell(r, fmt="{:.3g}"):
    if r is None:
        return "-"
    if "failed" in r:
        return "fail"
    return fmt.format(1e3 * r["t_call_s"]) + ("*" if r["truncated"] else "")


def collect():
    recs = _load_records()
    real = list(recs)
    # window "main": the whole trajectory (N <= 60) or its last S' calls (larger N, where the first calls are near zero)
    for r in list(recs):
        if (r["window"] == "first" and r["N"] <= 60) or (r["window"] == "late" and r["N"] > 60):
            recs.append(dict(r, window="main"))
    ok = [r for r in recs if "failed" not in r]
    idx = {(r["problem"], r["method"], r["N"], r["device"], r["window"]): r for r in recs}
    env = json.load(open(os.path.join(RES, "e12_env.json"))) if os.path.exists(os.path.join(RES, "e12_env.json")) else {}
    inputs = json.load(open(os.path.join(RES, "e12_inputs.json"))) if os.path.exists(os.path.join(RES, "e12_inputs.json")) else {}
    out = dict(env=env, inputs=inputs, records=real)
    with open(os.path.join(RES, "e12_scaling.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    L = []
    w = L.append
    w("# E12: node scaling and GPU benchmark of the closures\n")
    w("Generated by `code/experiments/e12_scaling.py collect` from the `results/e12_*.json` records.  "
      "Hand-written commentary is in `results/e12_notes.md` (appended below).\n")
    w("## 1. Hardware, software and protocol\n")
    for k in ("cpu", "cpu_threads", "ram_gb", "platform", "python", "numpy", "scipy", "cupy", "gpu", "gpu_mem_gb", "gpu_cc",
              "nvidia_smi", "cuda_runtime", "cuda_driver", "cpu_dgemm_gflops_1thread", "cpu_dgemm_gflops_allthreads",
              "gpu_dgemm_gflops", "gpu_sgemm_gflops", "gpu_stream_gbs"):
        if k in env:
            w(f"- {k}: {env[k]}")
    w("\nMeasured GEMM rates (n = 4096 for the GPU and the all-thread CPU): the GPU is a consumer card, so its FP64 rate is "
      "a small fraction of its FP32 rate (ratio above); every number below is float64 unless the device is `gpu32`.\n")
    w("Devices: " + "; ".join(f"`{d}` = {DEV_TEXT[d]}" for d in DEV_ORDER) + ".\n")
    w("Call sequences: S = 2000 closure calls from the stored trajectory, interpolated in space (cubic spline through the zero "
      "Dirichlet values) and linearly in time.  `window=first` is the rule of the task (the first S' calls); `window=late` is "
      "an additional run of the last S' calls after one untimed priming call, because the first calls of the trajectory have "
      "small, almost linear states.  Times are for the closure call only (input generation, and for `gpu` also the upload, "
      "are outside the timed region), medians of 3 repetitions or a single run when one takes longer than 60 s.  An asterisk marks a run "
      "stopped by its time budget before S' calls (time per call = mean over the calls done); `fail` = out of memory or "
      "error.  Accuracy: relative error per node against the converged reference on every 50th call "
      "(about five calls for short sequences).\n")
    # S' per N
    sprime = {}
    for r in ok:
        sprime.setdefault((r["problem"], r["N"], r["window"]), set()).add(r["S_used"])
    w("S' used: " + "; ".join(f"{p} N={n} {win}: {sorted(v)}" for (p, n, win), v in sorted(sprime.items())) + "\n")
    sec = 2
    for problem in ("heat", "ign"):
        w(f"## {sec}. {PROBLEM_TEXT[problem]}\n")
        Ns = sorted({r["N"] for r in ok if r["problem"] == problem})
        for window in ("main", "first"):
            if not any(r["window"] == window and r["problem"] == problem for r in recs):
                continue
            w(f"### {sec}.{1 if window == 'main' else 2} Window `{window}`" +
              (" (headline: whole trajectory for N <= 60, last S' calls above)" if window == "main" else
               " (the rule of the task: the first S' calls; for N >= 250 these start from almost zero states)") + "\n")
            devs = [d for d in DEV_ORDER if any(r["device"] == d and r["window"] == window and r["problem"] == problem for r in recs)]
            w("**Time per call [ms]** (rows: methods; columns: number of nodes N)\n")
            for dev in devs:
                w(f"\n{DEV_TEXT[dev]} (`{dev}`)\n")
                w("| method | " + " | ".join(str(n) for n in Ns) + " |")
                w("|---|" + "---:|" * len(Ns))
                for m in METHOD_ORDER[problem]:
                    row = [_cell(idx.get((problem, m, n, dev, window))) for n in Ns]
                    if any(c != "-" for c in row):
                        w(f"| {m}{' (network)' if m in NETWORK[problem] else ''} | " + " | ".join(row) + " |")
            w("\n**Accuracy** (median / maximum over nodes and checked calls of the relative error; `cpu1` run, `gpu` if absent)\n")
            w("| method | " + " | ".join(str(n) for n in Ns) + " |")
            w("|---|" + "---:|" * len(Ns))
            for m in METHOD_ORDER[problem]:
                cells = []
                for n in Ns:
                    r = next((idx[(problem, m, n, d, window)] for d in ("cpu1", "gpu", "cpuall", "gpux")
                              if (problem, m, n, d, window) in idx and "err_median" in idx[(problem, m, n, d, window)]), None)
                    cells.append("-" if r is None else (f"{r['err_median']:.1e} / {r['err_max']:.1e}" if r.get("err_finite", True)
                                 else "non-finite after call %d" % next((c for c, v in zip(r["err_checked_calls"], r["err_per_call_median"])
                                                                          if not np.isfinite(v)), -1)))
                if any(c != "-" for c in cells):
                    w(f"| {m} | " + " | ".join(cells) + " |")
            # speed relative to the best classical method of equal or better accuracy
            tg = TARGET[problem]
            for slack, text in ((1.0, "equal or better accuracy (median and maximum error not larger than the network's)"),
                                (3.0, "comparable accuracy (errors at most 3 times the network's)")):
                w(f"\n**Network `{tg}` against the best classical method of {text}**: time of the best classical method / time "
                  f"of `{tg}` (> 1: the network is faster); in brackets the best classical method.  `same` = classical method "
                  f"on the same device; `any` = best over all devices.\n")
                w("| device | " + " | ".join(str(n) for n in Ns) + " |")
                w("|---|" + "---:|" * len(Ns))
                for dev in devs:
                    for scope in ("same", "any"):
                        if scope == "any" and dev in ("cpu1", "cpu1orig"):
                            continue
                        cells = []
                        for n in Ns:
                            c = compare(idx, problem, n, dev, window, slack, scope)
                            cells.append("-" if c is None else (f"{c[0]:.2f} ({c[1]})" if c[1] else "no classical qualifies"))
                        w(f"| {dev} {scope} | " + " | ".join(cells) + " |")
        # iteration statistics (heat)
        if problem == "heat":
            w(f"### {sec}.3 Iteration statistics: slowest node against mean node\n")
            w("Iterations per call (mean over calls) of the batched solver (`max`: every node iterates until the slowest "
              "converges) and of its active-set variant (`max` of the active variant = slowest node; `mean` = mean over nodes "
              "and calls).  Jacobian-vector products / Jacobian assemblies per call and node-iterations per call and node "
              "are from the counters.  Device `cpu1`, window `first`, else `gpu`.\n")
            w("| N | solver | max (batched) | max (active) | mean node (active) | mean/max |")
            w("|---:|---|---:|---:|---:|---:|")
            for n in Ns:
                for base in ("chord_w8", "chord_w10", "nk_w8"):
                    for win in ("main", "first"):
                        for dev in ("cpu1", "gpu"):
                            b = idx.get((problem, base, n, dev, win))
                            a = idx.get((problem, base + "_as", n, dev, win))
                            if b and a and "iters_max_mean" in b and "iters_max_mean" in a:
                                w(f"| {n} ({win}) | {base} | {b['iters_max_mean']:.2f} | {a['iters_max_mean']:.2f} | "
                                  f"{a['iters_node_mean']:.2f} | {a['iters_node_mean'] / max(a['iters_max_mean'], 1e-9):.2f} |")
                                break
        sec += 1
    w(f"## {sec}. Verdict on H1 and H2 (computed)\n")
    w("Windows: `main` = the whole trajectory for N <= 60 and the last S' calls above (the first calls of the trajectory are "
      "close to the zero state); `first` = the rule of the task.  The accuracy criterion compares errors against the "
      "matrix-function closure (converged Galerkin solve, eigendecomposition), so **pseudo-spectral projection, which targets the exact "
      "projection and differs from that closure by the PC truncation error (property P4), is not among the methods of "
      "\"equal or better accuracy\"**; it is compared with the networks separately below.  All GPU numbers are for a consumer "
      "card (FP64 at 1/33 of FP32), so a negative H2 result is specific to this class of hardware.\n")
    for line in verdict_lines(idx):
        w(line)
    w("\n### Sparse PSP against the network (reported separately)\n")
    w("Time of sparse PSP / time of the network (> 1: the network is faster), with the error of PSP against the matrix-function "
      "closure in brackets (it contains the PC truncation difference).  Network: `pinn_nk2` (heat), `pinn` (ignition).  "
      "Window `main`.\n")
    for problem in ("heat", "ign"):
        Ns = sorted({r["N"] for r in ok if r["problem"] == problem and r["window"] == "main"})
        w(f"\n{PROBLEM_TEXT[problem].split(' (')[0]}\n")
        w("| PSP | device | " + " | ".join(str(n) for n in Ns) + " |")
        w("|---|---|" + "---:|" * len(Ns))
        for m in ("psp_s3", "psp_s4"):
            for dev in ("cpu1", "cpuall", "gpu"):
                cells = []
                for n in Ns:
                    a, t = idx.get((problem, m, n, dev, "main")), idx.get((problem, TARGET[problem], n, dev, "main"))
                    cells.append("-" if not a or not t or "failed" in a or "failed" in t else
                                 f"{a['t_call_s'] / t['t_call_s']:.2f} ({a['err_median']:.0e})")
                w(f"| {m} | {dev} | " + " | ".join(cells) + " |")
    # failures and notes
    fl = [r for r in recs if "failed" in r]
    if fl:
        w("\n## Failed configurations\n")
        for r in fl:
            w(f"- {r['problem']} {r['method']} N={r['N']} {r['device']} ({r['window']}): {r['failed']}")
    w("\n## Memory\n")
    w("Peak resident memory of the process (CPU) and CuPy memory-pool size (GPU), MB, window `first` else `late`, largest N run:\n")
    w("| problem | method | N | device | peak RSS | GPU pool |")
    w("|---|---|---:|---|---:|---:|")
    for problem in ("heat", "ign"):
        for m in METHOD_ORDER[problem]:
            for dev in ("cpu1", "gpu"):
                rs = [r for r in ok if r["problem"] == problem and r["method"] == m and r["device"] == dev]
                if rs:
                    r = max(rs, key=lambda r: (r["N"], r["window"] == "first"))
                    w(f"| {problem} | {m} | {r['N']} | {dev} | {r['peak_rss_mb']:.0f} | "
                      f"{'-' if r['gpu_pool_mb'] is None else format(r['gpu_pool_mb'], '.0f')} |")
    ee = os.path.join(RES, "e12_endtoend.json")
    if os.path.exists(ee):
        w("\n## End-to-end check (task Section 8)\n")
        w("Heat-conduction propagation of `e8_enthalpy.py` (T = 1, rtol 1e-7, atol 1e-9) with the closure on the GPU as right-hand "
          "side.  Reference: the stored CPU run (N = 15) or the same propagation with the new code on the CPU (N = 31).  "
          "(Wall times are not benchmark timings: the machine was not dedicated.)\n")
        w("| N | method | reference | max relative deviation of the coefficients | mean / std at (t=1, mid), GPU | same, reference | RHS calls |")
        w("|---:|---|---|---:|---|---|---:|")
        for k, r in json.load(open(ee)).items():
            w(f"| {r['N']} | {r['method']} | {r['reference']} | {r['max_rel_dev_coeffs']:.1e} | "
              f"{r['mean_std_gpu'][0]:.10f} / {r['mean_std_gpu'][1]:.10f} | {r['mean_std_ref'][0]:.10f} / "
              f"{r['mean_std_ref'][1]:.10f} | {r['nfev']} |")
    notes = os.path.join(RES, "e12_notes.md")
    if os.path.exists(notes):
        w("\n" + open(notes).read())
    par = os.path.join(RES, "e12_paragraph.tex")
    if os.path.exists(par):
        w("\n## Draft paragraph for the paper (LaTeX)\n\n```latex\n" + open(par).read().strip() + "\n```\n")
    with open(os.path.join(RES, "e12_scaling.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    make_figure(ok)
    print("wrote results/e12_scaling.{json,md,pdf}")


def err_of(idx, problem, m, n, window):
    for d in ("cpu1", "gpu", "cpuall", "gpux", "cpu1orig"):
        r = idx.get((problem, m, n, d, window))
        if r and "err_median" in r:
            return r["err_median"], r["err_max"]
    return None


def compare(idx, problem, N, dev, window, slack, scope):
    """(ratio, best classical method) for the target network on `dev`; ratio = time classical / time network."""
    t = idx.get((problem, TARGET[problem], N, dev, window))
    if t is None or "failed" in t:
        return None
    e = err_of(idx, problem, TARGET[problem], N, window)
    if e is None:
        return None
    best = None
    for m in METHOD_ORDER[problem]:
        if m in NETWORK[problem]:
            continue
        for d in ([dev] if scope == "same" else [x for x in DEV_ORDER if x not in ("cpu1orig", "gpu32")]):
            r = idx.get((problem, m, N, d, window))
            if r is None or "failed" in r or "err_median" not in r or d == "gpu32":
                continue
            if max(r["err_median"], ERR_FLOOR) <= slack * max(e[0], ERR_FLOOR) and \
                    max(r["err_max"], ERR_FLOOR) <= slack * max(e[1], ERR_FLOOR):
                if best is None or r["t_call_s"] < best[0]:
                    best = (r["t_call_s"], f"{m}" + ("" if scope == "same" else f"@{d}"))
    if best is None:
        return (float("nan"), "")
    return (best[0] / t["t_call_s"], best[1])


def verdict_lines(idx):
    lines = []
    for problem in ("heat", "ign"):
        tg = TARGET[problem]
        Ns = sorted({k[2] for k in idx if k[0] == problem})
        for slack, nm in ((1.0, "strict criterion (errors not larger than the network's)"),
                          (3.0, "loose criterion (errors at most 3 times the network's)")):
            for hyp, devs in (("H1 (CPU)", ("cpu1", "cpuall")), ("H2 (GPU)", ("gpu", "gpux"))):
                wins, allr = [], []
                for dev in devs:
                    for window in ("main", "first"):
                        for n in Ns:
                            c = compare(idx, problem, n, dev, window, slack, "same")
                            if c and c[1]:
                                allr.append(c[0])
                                if c[0] > 1.0:
                                    wins.append((dev, window, n, c[0], c[1]))
                head = f"- {problem}, `{tg}`, {hyp}, {nm}: "
                if not wins:
                    lines.append(head + f"**not supported** in the tested range (ratio classical/network from {min(allr):.2f} "
                                 f"to {max(allr):.2f}; the network was never faster)" if allr else head + "no data")
                    continue
                by = {}
                for dev, window, n, r, m in wins:
                    by.setdefault((dev, window), []).append((n, r, m))
                parts = []
                for (dev, window), v in sorted(by.items()):
                    v.sort()
                    meths = sorted({m for _, _, m in v})
                    parts.append(f"{dev}/{window}: N = {v[0][0]}..{v[-1][0]} ({len(v)} sizes), {min(r for _, r, _ in v):.2f}x to "
                                 f"{max(r for _, r, _ in v):.2f}x faster than {', '.join(meths)}")
                lines.append(head + "**supported**: " + "; ".join(parts))
    return lines


FIG_METHODS = {"heat": ("chord_w8", "chord_w10", "nk_w8", "newton_w8", "chordfix2", "psp_s4", "pinn", "pinn_nk1", "pinn_nk2"),
               "ign": tuple(gb.IGN_METHODS)}


def make_figure(ok):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    marks = {}
    ls = {"cpu1": "-", "gpu": "--"}
    with PdfPages(os.path.join(RES, "e12_scaling.pdf")) as pdf:
        for window in ("main", "first"):
            if not any(r["window"] == window for r in ok):
                continue
            fig, axes = plt.subplots(1, 2, figsize=(11, 6.2))
            for ax, problem in zip(axes, ("heat", "ign")):
                shown = [m for m in METHOD_ORDER[problem] if m in FIG_METHODS[problem]]
                for k, m in enumerate(shown):
                    for dev in ("cpu1", "gpu"):
                        pts = sorted((r["N"], r["t_node_call_us"]) for r in ok
                                     if r["problem"] == problem and r["method"] == m and r["device"] == dev
                                     and r["window"] == window)
                        if not pts:
                            continue
                        net = m in NETWORK[problem]
                        col = FAMILY[family(problem, m)]
                        mk = "os^vDPXs*<>"[k % 11]
                        ax.plot(*zip(*pts), ls=ls[dev], color=col, marker=mk, ms=7 if net else 4.5, lw=2.2 if net else 1.0,
                                mfc=col if net else "none", alpha=1.0 if net else 0.85)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("number of nodes N")
                ax.set_ylabel("time per node and call [µs]")
                ax.set_title(PROBLEM_TEXT[problem].split(" (")[0] + f" (window: {window})", fontsize=10)
                ax.grid(alpha=0.25, lw=0.5)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                handles = [Line2D([], [], color="#52514e", ls="-", label="CPU, 1 thread"),
                           Line2D([], [], color="#52514e", ls="--", label="GPU (FP64, device-resident)")]
                for k, m in enumerate(shown):
                    net = m in NETWORK[problem]
                    handles.append(Line2D([], [], color=FAMILY[family(problem, m)], marker="os^vDPXs*<>"[k % 11],
                                          mfc=FAMILY[family(problem, m)] if net else "none", ms=6, lw=1.5 if net else 0.8,
                                          label=m + (" (network)" if net else "")))
                ax.legend(handles=handles, fontsize=7.5, ncol=2, frameon=False, loc="upper center",
                          bbox_to_anchor=(0.5, -0.14))
            fig.suptitle("Time per node and call against mesh size (the other methods and devices are in the tables)",
                         fontsize=9, y=0.995)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)



def stage_endtoend():
    """Optional check (task Section 8): the heat-conduction propagation of e8_enthalpy.py (T = 1, rtol 1e-7, atol 1e-9) with
    the closures of gpu_backend on the GPU as right-hand side, against the stored CPU runs (N = 15) and, for N = 31 (no stored
    run), against the same propagation with the new code on the CPU."""
    from problems import EnthalpyHeat
    from propagate import intrusive
    B = PCBasis(D, P_ORDER)
    stored = pickle.load(open(os.path.join(RES, "e8_enthalpy.pkl"), "rb"))
    T_EVAL = np.linspace(0, 1.0, 21)
    out = {}
    for N in (15, 31):
        prob = EnthalpyHeat(B, nx=N)
        for method in ("chord_w10", "pinn_nk2", "psp_s4"):
            runs = {}
            for dev in ("gpu", "cpu1"):
                xp = gb.get_xp(dev)
                ops = gb.Ops(B, xp, work_bytes=GPU_WORK if xp is not np else 256e6)
                cl = gb.build_heat(method, ops, prob.K3, os.path.join(MODELS, SRC["heat"]["model"]), CubicField(B))
                K3d = ops.asarray(prob.K3)
                f = lambda H, K=None, cl=cl, ops=ops, K3d=K3d: gb.to_host(cl(ops.asarray(H), K3d))
                C, wall, nfev, ok = intrusive(prob, f, T_EVAL, rtol=1e-7, atol=1e-9)
                runs[dev] = dict(C=C, wall=wall, nfev=nfev, ok=ok)
            ref = stored["methods"][method]["C"] if (N == 15 and method in stored["methods"]) else runs["cpu1"]["C"]
            refname = "stored CPU run" if (N == 15 and method in stored["methods"]) else "new code on the CPU"
            mid = N // 2
            stat = lambda C: (float(C[-1, mid, 0]), float(np.sqrt(np.sum(C[-1, mid, 1:] ** 2))))
            rec = dict(N=N, method=method, reference=refname, ok=bool(runs["gpu"]["ok"]), nfev=int(runs["gpu"]["nfev"]),
                       nfev_cpu=int(runs["cpu1"]["nfev"]), wall_gpu=runs["gpu"]["wall"], wall_cpu1=runs["cpu1"]["wall"],
                       max_rel_dev_coeffs=float(np.abs(runs["gpu"]["C"] - ref).max() / np.abs(ref).max()),
                       mean_std_gpu=stat(runs["gpu"]["C"]), mean_std_ref=stat(ref))
            out[f"{N}_{method}"] = rec
            log(f"end-to-end N={N} {method}: max rel deviation of coefficients {rec['max_rel_dev_coeffs']:.2e} vs {refname}; "
                f"mean/std at (t=1, mid) {rec['mean_std_gpu']} vs {rec['mean_std_ref']}; wall gpu {rec['wall_gpu']:.1f} s, "
                f"cpu {rec['wall_cpu1']:.1f} s", "endtoend")
    with open(os.path.join(RES, "e12_endtoend.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def run_config(problem, method, N, device, calls, window, budget, reps):
    name = f"{problem}_{method}_{N}_{device}" + ("" if window == "first" else "_late")
    try:
        stage_run(problem, method, N, device, calls=calls, window=window, budget=budget, reps=reps)
    except Exception as e:                           # out of memory, ...: reported as such (task rule 8)
        log(f"FAILED {name}: {type(e).__name__}: {str(e)[:300]}", name)
        with open(os.path.join(RES, f"e12_{name}.json"), "w") as fh:
            json.dump(dict(problem=problem, method=method, N=N, device=device, window=window,
                           failed=f"{type(e).__name__}: {str(e)[:300]}"), fh, indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0]
    if cmd == "env":
        stage_env()
    elif cmd == "inputs":
        stage_inputs(a[1:] or ["heat", "ign"])
    elif cmd == "refs":
        stage_refs(a[1], int(a[2]), int(a[3]), int(a[4]))
    elif cmd == "run":
        problem, method, N, device = a[1], a[2], int(a[3]), a[4]
        opt = dict(zip(a[5::2], a[6::2]))
        run_config(problem, method, N, device, opt.get("--calls"), opt.get("--window", "first"),
                   float(opt.get("--budget", 600)), int(opt.get("--reps", 3)))
    elif cmd == "batch":                              # configurations from stdin, one forked process each
        for line in sys.stdin:
            f = line.split()
            if not f or f[0].startswith("#"):
                continue
            problem, method, N, device, window, calls, budget = f[:7]
            sys.stdout.flush()
            pid = os.fork()                           # imports are shared; CUDA is initialised in the child only
            if pid == 0:
                try:
                    reps = int(f[7]) if len(f) > 7 else 3          # optional 8th column: repetitions
                    run_config(problem, method, int(N), device, int(calls), window, float(budget), reps)
                finally:
                    sys.stdout.flush()
                    os._exit(0)
            os.waitpid(pid, 0)
    elif cmd == "endtoend":
        stage_endtoend()
    elif cmd == "collect":
        collect()
    else:
        raise SystemExit(__doc__)
