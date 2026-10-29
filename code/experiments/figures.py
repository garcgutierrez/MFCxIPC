"""Figures for the paper (vector PDF).  Usage: python figures.py [e1 e3 e4 e5 cost]"""
import json
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, FIG

# Categorical palette in fixed slot order (validated reference palette); colour follows the method.
INK, INK2 = "#0b0b0b", "#52514e"
STYLE = {
    "pinn_anchored": dict(color="#2a78d6", ls="-", lw=1.8, label="PINN closure (anchored)"),
    "pinn_generic": dict(color="#2a78d6", ls=(0, (1, 1)), lw=1.6, label="PINN closure (generic)"),
    "pinn": dict(color="#2a78d6", ls="-", lw=1.8, label="PINN closure"),
    "xtfc": dict(color="#eb6834", ls="--", lw=1.6, label="spectral X-TFC (GH nodes)"),
    "xtfc_gh": dict(color="#eb6834", ls="--", lw=1.6, label="spectral X-TFC (GH nodes)"),
    "xtfc_qmc2P": dict(color="#eb6834", ls=(0, (1, 1)), lw=1.4, label=r"spectral X-TFC (QMC, $Q=2P$)"),
    "xtfc_qmc4P": dict(color="#eb6834", ls="-.", lw=1.4, label=r"spectral X-TFC (QMC, $Q=4P$)"),
    "eig": dict(color="#1baf7a", ls="-", lw=1.4, label=r"eigendecomposition (exact $\mathcal{G}_f$)"),
    "psp": dict(color="#eda100", ls="-.", lw=1.6, label=r"tensor PSP ($q=p+1$)"),
    "taylor4": dict(color="#e87ba4", ls=(0, (5, 1.5)), lw=1.6, label="Taylor-4"),
    "taylor8": dict(color="#e87ba4", ls=(0, (5, 1.5)), lw=1.6, label="Taylor-8"),
    "lanczos": dict(color="#008300", ls=(0, (3, 1, 1, 1)), lw=1.4, label="Lanczos ($m=10$)"),
    "newton": dict(color="#1baf7a", ls="-", lw=1.4, label="Newton (converged)"),
    "nk": dict(color="#008300", ls=(0, (3, 1, 1, 1)), lw=1.4, label="Newton–Krylov (converged)"),
    "pinn_newton1": dict(color="#4a3aa7", ls="--", lw=1.6, label="PINN + 1 Newton step"),
    "pinn_nk1": dict(color="#4a3aa7", ls=(0, (1, 1)), lw=1.6, label="PINN + 1 Newton–Krylov step"),
    "linear": dict(color="#e34948", ls=":", lw=1.4, label="linearised constraint"),
    "pinn_nk2": dict(color="#4a3aa7", ls="-", lw=1.8, label="PINN + 2 Newton–Krylov steps"),
    "chord_w8": dict(color="#1baf7a", ls=(0, (1, 1)), lw=1.8, label=r"chord, warm, $10^{-8}$"),
    # sparse-grid (Smolyak) PSP: same hue as tensor PSP, different dash and marker
    "psp_sparse": dict(color="#eda100", ls=(0, (1, 1)), lw=1.6, label=r"sparse PSP ($k=p$)"),
    "psp_s3": dict(color="#eda100", ls=(0, (1, 1)), lw=1.6, label=r"sparse PSP ($k=p$)"),
    "psp_s4": dict(color="#eda100", ls="--", lw=1.4, label=r"sparse PSP ($k=p+1$)"),
}
MARK = {"psp": "o", "psp_s0": "s", "psp_s1": "^"}
TAY = {2: dict(color="#e87ba4", ls=":", lw=1.4), 4: dict(color="#e87ba4", ls="-.", lw=1.4),
       8: dict(color="#e87ba4", ls="--", lw=1.4)}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "legend.fontsize": 7.2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.5, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "lines.solid_capstyle": "round", "pdf.fonttype": 42, "mathtext.fontset": "cm",
})
W1, W2 = 3.5, 7.2


def tidy_log(ax):
    from matplotlib.ticker import LogLocator, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_minor_formatter(NullFormatter())


def bottom_legend(fig, axs, ncol=4, y=-0.2):
    seen, H, Lb = set(), [], []
    for ax in np.atleast_1d(axs):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l); H.append(h); Lb.append(l)
    fig.legend(H, Lb, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)
    print("saved", name)


# -----------------------------------------------------------------------------
def fig_e1():
    names = [("exp_d2p4", r"$\exp$, $d=2$, $p=4$"), ("exp_d5p3", r"$\exp$, $d=5$, $p=3$"),
             ("sincos_d2p6", r"$\sin/\cos$, $d=2$, $p=6$"), ("pow2.5_d4p3", r"$(\cdot)^{2.5}$, $d=4$, $p=3$"),
             ("pow-1_d4p3", r"$(\cdot)^{-1}$, $d=4$, $p=3$")]
    data = {n: json.load(open(os.path.join(RES, f"e1_{n}.json"))) for n, _ in names
            if os.path.exists(os.path.join(RES, f"e1_{n}.json"))}
    if not data:
        return
    fig, axs = plt.subplots(1, len(data), figsize=(W2, 2.0), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (n, title) in zip(axs, [x for x in names if x[0] in data]):
        r = data[n]
        rho = np.array(r["rho"])
        bins = np.quantile(rho, np.linspace(0, 1, 13))
        mid = 0.5 * (bins[1:] + bins[:-1])
        which = np.digitize(rho, bins[1:-1])

        def binned(e):
            e = np.array(e)
            return np.array([np.median(e[which == i]) for i in range(len(mid))])
        for K in [2, 4, 8]:
            ax.semilogy(mid, binned(r[f"taylor{K}"]), **TAY[K], label=f"Taylor-{K}")
        for m_, ls in [(6, ":"), (8, "--")]:
            if f"lanczos{m_}" in r:
                ax.semilogy(mid, np.maximum(binned(r[f"lanczos{m_}"]), 1e-16), color="#008300", ls=ls, lw=1.4,
                            label=f"Lanczos-{m_}")
        ax.semilogy(mid, binned(r["pinn"]), color="#2a78d6", lw=1.8, marker="o", ms=3, label="PINN closure")
        ax.set_title(title)
    fig.supxlabel(r"spectral radius of $r\mathbf{M}_n$ (binned)", y=-0.1, fontsize=8.5)
    axs[0].set_ylabel("median relative error")
    axs[0].set_ylim(1e-14, 10)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.27))
    save(fig, "e1_closure_error")

    fig, ax = plt.subplots(figsize=(W1, 2.2))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for (n, title), c in zip([x for x in names if x[0] in data], cols):
        # history rows: [phase, iteration, loss, val_err]; L-BFGS iterations continue the count
        hist = data[n]["history"]
        n_adam = max(row[1] for row in hist if row[0] == "adam") + 1
        x = [row[1] + (n_adam if row[0] == "lbfgs" else 0) for row in hist]
        ax.semilogy(x, [row[3] for row in hist], color=c, lw=1.4, label=title)
        ax.axvline(n_adam, color=INK2, lw=0.5, ls=":")
    ax.set_xlabel("iteration (Adam, then L-BFGS)")
    ax.set_ylabel("validation relative error")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=6.5)
    save(fig, "e1_training")


# -----------------------------------------------------------------------------
def _err_panels(ax, out, key, comp, methods, ylabel, logy=True):
    t = out["t"]
    for m in methods:
        d = out["methods"].get(m)
        if d is None or not (d.get("ok") or d.get("partial")):
            continue
        y = d[key][:, comp]
        (ax.semilogy if logy else ax.plot)(t[:len(y)], y, **STYLE[m])
    ax.set_ylabel(ylabel)


def fig_e3():
    path = os.path.join(RES, "e3_pendulum.pkl")
    if not os.path.exists(path):
        return
    out = pickle.load(open(path, "rb"))
    methods = [m for m in ["eig", "psp", "lanczos", "taylor8", "pinn_generic", "pinn_anchored", "xtfc"]
               if m in out["methods"]]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
    _err_panels(axs[0], out, "l2_err", 0, methods, r"$\varepsilon_{L^2}(\theta)$")
    _err_panels(axs[1], out, "std_err_p" if "std_err_p" in out["methods"]["eig"] else "std_err", 0, methods,
                r"$\varepsilon_{\sigma}(\theta)$ (paired)")
    _err_panels(axs[2], out, "bc", 0, methods, r"$BC(\theta)$", logy=False)
    for ax in axs:
        ax.set_xlabel("$t$")
    for ax in axs[:2]:
        ax.set_ylim(bottom=1e-5)
        tidy_log(ax)
    fig.subplots_adjust(wspace=0.38)
    bottom_legend(fig, axs, ncol=3, y=-0.36)
    save(fig, "e3_pendulum_errors")
    # the same without the X-TFC solver (version of the paper restricted to closures)
    closures = [m for m in methods if not m.startswith("xtfc")]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
    _err_panels(axs[0], out, "l2_err", 0, closures, r"$\varepsilon_{L^2}(\theta)$")
    _err_panels(axs[1], out, "std_err_p" if "std_err_p" in out["methods"]["eig"] else "std_err", 0, closures,
                r"$\varepsilon_{\sigma}(\theta)$ (paired)")
    _err_panels(axs[2], out, "bc", 0, closures, r"$BC(\theta)$", logy=False)
    for ax in axs:
        ax.set_xlabel("$t$")
    for ax in axs[:2]:
        ax.set_ylim(bottom=1e-5)
        tidy_log(ax)
    fig.subplots_adjust(wspace=0.38)
    bottom_legend(fig, axs, ncol=3, y=-0.36)
    save(fig, "e3_pendulum_errors_closures")

    # PDFs of theta at the three snapshot times
    Psi = out["Psi_sub"]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.0))
    for j, (ax, it) in enumerate(zip(axs, out["pdf"]["idx"])):
        mc = out["pdf"]["mc"][j, 0]
        lo, hi = np.percentile(mc, [0.2, 99.8])
        edges = np.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), 70)
        ax.hist(mc, bins=edges, density=True, color="#d9d8d3", label="Monte Carlo")
        for m in methods:
            d = out["methods"][m]
            if not (d.get("ok") or d.get("partial")) or it >= len(d["C"]):
                continue
            s = d["C"][it, 0] @ Psi.T
            h, e = np.histogram(s, bins=edges, density=True)
            ax.plot(0.5 * (e[1:] + e[:-1]), h, **{k: v for k, v in STYLE[m].items() if k != "lw"}, lw=1.1)
        ax.set_title(f"$t={out['t'][it]:.0f}$")
        ax.set_xlabel(r"$\theta$")
        ax.grid(False)
    axs[0].set_ylabel("PDF")
    bottom_legend(fig, axs, ncol=3, y=-0.38)
    save(fig, "e3_pendulum_pdfs")

    # closure error along the trajectory
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for m in ["pinn_generic", "pinn_anchored"]:
        d = out["methods"].get(m)
        if d is not None and "closure_err" in d:
            ax.semilogy(out["t"], d["closure_err"], **STYLE[m])
    ax.set_xlabel("$t$")
    ax.set_ylabel("closure relative error")
    ax.legend(loc="lower right")
    save(fig, "e3_closure_along")


def fig_e4():
    path = os.path.join(RES, "e4_toggle.pkl")
    if not os.path.exists(path):
        return
    out = pickle.load(open(path, "rb"))
    methods = [m for m in ["eig", "psp", "taylor4", "lanczos", "pinn_generic", "xtfc_gh", "xtfc_qmc2P", "xtfc_qmc4P"]
               if m in out["methods"]]
    STYLE["pinn_generic"] = dict(STYLE["pinn"])
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
    _err_panels(axs[0], out, "l2_err", 0, methods, r"$\varepsilon_{L^2}(u)$")
    _err_panels(axs[1], out, "l2_err", 1, methods, r"$\varepsilon_{L^2}(v)$")
    Psi = out["Psi_sub"]
    j = len(out["pdf"]["idx"]) - 1
    it = out["pdf"]["idx"][j]
    mc = out["pdf"]["mc"][j, 1]
    lo, hi = np.percentile(mc, [0.1, 99.9])
    edges = np.linspace(lo, hi, 70)
    axs[2].hist(mc, bins=edges, density=True, color="#d9d8d3", label="Monte Carlo")
    for m in methods:
        d = out["methods"][m]
        if d.get("ok"):
            h, e = np.histogram(d["C"][it, 1] @ Psi.T, bins=edges, density=True)
            axs[2].plot(0.5 * (e[1:] + e[:-1]), h, **{k: v for k, v in STYLE[m].items() if k != "lw"}, lw=1.1)
    axs[2].set_xlabel(f"$v(t={out['t'][it]:.0f})$")
    axs[2].set_ylabel("PDF")
    axs[2].grid(False)
    for ax in axs[:2]:
        ax.set_xlabel("$t$")
        ax.set_ylim(bottom=1e-4)
        tidy_log(ax)
    fig.subplots_adjust(wspace=0.38)
    bottom_legend(fig, axs, ncol=3, y=-0.42)
    save(fig, "e4_toggle")


def fig_e5():
    path = os.path.join(RES, "e5_ignition.pkl")
    if not os.path.exists(path):
        return
    out = pickle.load(open(path, "rb"))
    p6 = os.path.join(RES, "e6_ignition_highd.pkl")
    if os.path.exists(p6):
        r6 = pickle.load(open(p6, "rb"))
        if "pinn" in r6["methods"]:
            r6["methods"]["pinn_generic"] = r6["methods"].pop("pinn")
        if "psp_s3" in r6["methods"]:
            r6["methods"]["psp_sparse"] = r6["methods"]["psp_s3"]
        out["d8p3"] = r6
    STYLE["pinn_generic"] = dict(STYLE["pinn"])
    methods = ["eig", "psp", "psp_sparse", "taylor4", "lanczos", "pinn_generic"]
    keys = list(out.keys())
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
    r = out[keys[-1]]
    x = np.r_[0, r["x"], 1]
    pad = lambda v: np.r_[0, v, 0]
    axs[0].plot(x, pad(r["mc_mean"][-1]), color=INK, lw=2.4, alpha=0.35, label="Monte Carlo")
    axs[1].plot(x, pad(r["mc_std"][-1]), color=INK, lw=2.4, alpha=0.35, label="Monte Carlo")
    for m in methods:
        d = r["methods"].get(m)
        if d and d.get("ok"):
            C = d["C"][-1]
            axs[0].plot(x, pad(C[:, 0]), **STYLE[m])
            axs[1].plot(x, pad(np.sqrt(np.sum(C[:, 1:] ** 2, 1))), **STYLE[m])
    axs[0].set_ylabel(r"$\mathbb{E}[u(x,T)]$")
    axs[1].set_ylabel(r"$\mathrm{sd}[u(x,T)]$")
    for ax in axs[:2]:
        ax.set_xlabel("$x$")
    axs[0].set_title("$d=%s$, $p=%s$" % (keys[-1][1], keys[-1][3]))
    axs[1].set_title("$d=%s$, $p=%s$" % (keys[-1][1], keys[-1][3]))
    for k, ls in zip(keys, ["-", "--", ":"]):
        rr = out[k]
        for m in methods:
            d = rr["methods"].get(m)
            if d and d.get("ok"):
                st = dict(STYLE[m]); st["ls"] = ls; st.pop("label")
                axs[2].semilogy(rr["t"][1:], d["l2_err"][1:].max(1), **st)
    axs[2].set_xlabel("$t$")
    axs[2].set_ylabel(r"$\max_x\varepsilon_{L^2}(u)$")
    nice = lambda k: "$d=%s$, $p=%s$" % (k[1], k[3])
    for k, ls in zip(keys, ["-", "--", ":"]):
        axs[2].plot([], [], color=INK2, ls=ls, label=nice(k))
    tidy_log(axs[2])
    fig.subplots_adjust(wspace=0.38)
    bottom_legend(fig, axs, ncol=3, y=-0.42)
    save(fig, "e5_ignition")


def fig_e7():
    path = os.path.join(RES, "e7_series_spring.pkl")
    if not os.path.exists(path):
        return
    out = pickle.load(open(path, "rb"))
    STYLE["pinn"] = dict(STYLE["pinn"], label="PINN closure")
    STYLE["linear"] = dict(STYLE["linear"], label="linearised constraint")
    methods = [m for m in ["newton", "nk", "psp", "psp_s3", "pinn", "pinn_newton1", "pinn_nk1", "linear"]
               if out["methods"].get(m, {}).get("ok")]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
    _err_panels(axs[0], out, "l2_err", 0, methods, r"$\varepsilon_{L^2}(x)$")
    axs[0].set_xlabel("$t$")
    axs[0].set_ylim(1e-4, 10)
    tidy_log(axs[0])
    for m in ["psp", "psp_s3", "pinn", "pinn_newton1", "pinn_nk1", "linear"]:
        d = out["methods"].get(m, {})
        if "closure_err" in d:
            axs[1].semilogy(out["t"], np.maximum(d["closure_err"], 1e-16), **STYLE[m])
    axs[1].set_xlabel("$t$")
    axs[1].set_ylabel("closure error vs. Newton")
    tidy_log(axs[1])
    Psi = out["Psi_sub"]
    j = len(out["pdf"]["idx"]) - 1
    it = out["pdf"]["idx"][j]
    mc = out["pdf"]["mc"][j, 0]
    lo, hi = np.percentile(mc, [0.2, 99.8])
    edges = np.linspace(lo, hi, 70)
    axs[2].hist(mc, bins=edges, density=True, color="#d9d8d3", label="Monte Carlo")
    for m in methods:
        if m == "linear":
            continue
        h, e = np.histogram(out["methods"][m]["C"][it, 0] @ Psi.T, bins=edges, density=True)
        axs[2].plot(0.5 * (e[1:] + e[:-1]), h, **{k: v for k, v in STYLE[m].items() if k != "lw"}, lw=1.1)
    axs[2].set_xlabel(f"$x(t={out['t'][it]:.0f})$")
    axs[2].set_ylabel("PDF")
    axs[2].grid(False)
    fig.subplots_adjust(wspace=0.38)
    bottom_legend(fig, axs, ncol=3, y=-0.42)
    save(fig, "e7_series_spring")


def fig_e8():
    path = os.path.join(RES, "e8_enthalpy.pkl")
    if not os.path.exists(path):
        return
    out = pickle.load(open(path, "rb"))
    STYLE["pinn"] = dict(STYLE["pinn"], label="PINN closure")
    STYLE["linear"] = dict(STYLE["linear"], label="linearised law")
    methods = [m for m in ["newton", "nk", "chord_w8", "psp", "psp_s3", "pinn", "pinn_nk1", "pinn_nk2", "linear"]
               if out["methods"].get(m, {}).get("ok")]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
    x = np.r_[0, out["x"], 1]
    pad = lambda v: np.r_[0, v, 0]
    axs[0].plot(x, pad(out["mcT_mean"][-1]), color=INK, lw=2.4, alpha=0.35, label="Monte Carlo")
    axs[0].plot(x, pad(out["mcT_std"][-1]) * 5, color=INK, lw=2.4, alpha=0.35, ls="--")
    for m in methods:
        if m == "linear":
            continue
        d = out["methods"][m]
        axs[0].plot(x, pad(d["T_mean"]), **STYLE[m])
        st = dict(STYLE[m]); st.pop("label")
        axs[0].plot(x, pad(d["T_std"]) * 5, **dict(st, lw=1.0))
    axs[0].set_xlabel("$x$")
    axs[0].set_ylabel(r"$\mathbb{E}[T]$, $5\,\mathrm{sd}[T]$ at $t=1$")
    for m in methods:
        d = out["methods"][m]
        axs[1].semilogy(out["t"][1:], d["l2_err"][1:].max(1), **STYLE[m])
    axs[1].set_xlabel("$t$")
    axs[1].set_ylabel(r"$\max_x\varepsilon_{L^2}(h)$")
    axs[1].set_ylim(bottom=1e-4)
    tidy_log(axs[1])
    for m in ["psp", "psp_s3", "pinn", "pinn_nk1", "pinn_nk2", "linear"]:
        d = out["methods"].get(m, {})
        if "closure_err" in d:
            e = np.sort(d["closure_err"])
            axs[2].semilogx(np.maximum(e, 1e-16), np.linspace(0, 1, len(e)), **STYLE[m])
    axs[2].set_xlabel("closure error vs. Newton")
    axs[2].set_ylabel("cumulative fraction of inputs")
    fig.subplots_adjust(wspace=0.4)
    bottom_legend(fig, axs, ncol=3, y=-0.42)
    save(fig, "e8_enthalpy")


def fig_cost():
    path = os.path.join(RES, "bench_cost.json")
    if not os.path.exists(path):
        return
    rows = json.load(open(path))
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.75, 2.3), sharey=True)
    for ax, p in zip(axs, [3, 4]):
        rr = sorted([r for r in rows if r["p"] == p], key=lambda r: r["d"])
        ds = [r["d"] for r in rr]
        for k, st in [("eig", STYLE["eig"]), ("psp", STYLE["psp"]), ("psp_s0", STYLE["psp_s3"]),
                      ("psp_s1", STYLE["psp_s4"]), ("taylor4", STYLE["taylor4"]), ("taylor8", STYLE["taylor8"]),
                      ("lanczos10", STYLE["lanczos"]), ("pinn", STYLE["pinn"])]:
            if k == "taylor8":
                st = dict(st, ls=":")
            st = dict(st, marker=MARK.get(k, "o"))
            xs = [r["d"] for r in rr if k in r]
            ys = [r[k] * 1e3 for r in rr if k in r]
            ax.semilogy(xs, ys, ms=3, **st)
        ax.set_xticks(ds)
        ax.set_xticklabels([f"{r['d']}\n({r['P']})" for r in rr])
        ax.set_xlabel("$d$  ($P$)")
        ax.set_title(f"$p={p}$")
        tidy_log(ax)
    axs[0].set_ylabel("time per closure call [ms]")
    bottom_legend(fig, axs, ncol=3, y=-0.42)
    save(fig, "cost_closure")


def fig_cost_implicit():
    path = os.path.join(RES, "bench_implicit.json")
    if not os.path.exists(path):
        return
    rows = sorted(json.load(open(path)), key=lambda r: r["d"])
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    for k in ["newton", "nk", "psp", "psp_s0", "psp_s1", "pinn", "pinn_newton1", "pinn_nk1", "pinn_nk2"]:
        xs = [r["d"] for r in rows if k in r]
        ys = [r[k] * 1e3 / r["batch"] for r in rows if k in r]
        st = dict(STYLE[{"psp_s0": "psp_s3", "psp_s1": "psp_s4"}.get(k, k)])
        if k == "pinn":
            st["label"] = "PINN closure"
        if k == "psp":
            st["label"] = r"tensor PSP ($q=4$)"
        if k in ("psp_s0", "psp_s1"):
            st["label"] = rf"sparse PSP ($k={3 + int(k[-1])}$)"
        st["marker"] = MARK.get(k, "o")
        ax.semilogy(xs, ys, ms=3, **st)
    ax.set_xticks([r["d"] for r in rows])
    ax.set_xticklabels([f"{r['d']}\n({r['P']})" for r in rows])
    ax.set_xlabel("$d$  ($P$),  $p=3$")
    ax.set_ylabel("time per input [ms]")
    tidy_log(ax)
    bottom_legend(fig, [ax], ncol=2, y=-0.5)
    save(fig, "cost_implicit")


def fig_tradeoff():
    """Dot plot: online wall time of every closure (ignition d=8, heat conduction, oscillator), its error in a column on
    the right; for the networks a line extends to the cost of one propagation including pilot run and training.
    Hollow markers: error more than 25 % above that of the converged Galerkin closure."""
    fam = {"classical": dict(color="#1baf7a", marker="D", label="Galerkin closure (eig., Lanczos, Newton, NK)"),
           "chord": dict(color="#008300", marker="h", label="chord Newton, warm start"),
           "taylor": dict(color="#e87ba4", marker="<", label="Taylor-4"),
           "psp": dict(color="#eda100", marker="o", label="tensor PSP"),
           "sparse": dict(color="#eda100", marker="s", label="sparse PSP"),
           "pinn": dict(color="#2a78d6", marker="*", label="network alone"),
           "hyb": dict(color="#4a3aa7", marker="P", label="network + corrections"),
           "lin": dict(color="#e34948", marker="X", label="linearised law")}
    panels = [
        ("e6_ignition_highd.pkl", "Ignition ($d=8$, $P=165$, 31 nodes)", "pde", "eig",
         [("eig", "classical", "eigendecomposition"), ("lanczos", "classical", "Lanczos ($m=10$)"),
          ("taylor4", "taylor", "Taylor-4"), ("psp", "psp", "tensor PSP"), ("psp_s3", "sparse", "sparse PSP, $k=p$"),
          ("psp_s4", "sparse", "sparse PSP, $k=p+1$"), ("pinn", "pinn", "PINN closure")]),
        ("e8_enthalpy.pkl", "Heat conduction ($d=8$, $P=165$, 15 nodes)", "pde", "chord_w10",
         [("newton", "classical", "Newton, cold, $10^{-12}$"), ("nk", "classical", "NK, cold, $10^{-10}$"),
          ("newton_w8", "classical", "Newton, warm, $10^{-8}$"), ("nk_w8", "classical", "NK, warm, $10^{-8}$"),
          ("chord_w8", "chord", "chord, warm, $10^{-8}$"), ("chord_w10", "chord", "chord, warm, $10^{-10}$"),
          ("psp", "psp", "tensor PSP"), ("psp_s3", "sparse", "sparse PSP, $k=3$"),
          ("psp_s4", "sparse", "sparse PSP, $k=4$"), ("pinn", "pinn", "network alone"),
          ("pinn_nk1", "hyb", "network + 1 NK"), ("pinn_nk2", "hyb", "network + 2 NK"),
          ("linear", "lin", "linearised law")]),
        ("e7_series_spring.pkl", "Oscillator ($d=6$, $P=84$)", "ode", "newton_w8",
         [("newton", "classical", "Newton, cold, $10^{-12}$"), ("newton_w8", "classical", "Newton, warm, $10^{-8}$"),
          ("chord_w8", "chord", "chord, warm, $10^{-8}$"), ("psp", "psp", "tensor PSP"),
          ("psp_s3", "sparse", "sparse PSP, $k=3$"), ("psp_s4", "sparse", "sparse PSP, $k=4$"),
          ("pinn", "pinn", "network alone"), ("pinn_newton1", "hyb", "network + 1 Newton"),
          ("linear", "lin", "linearised law")])]
    data = []
    for fname, title, kind, ref, pts in panels:
        o = load_pkl(fname)
        if o is None:
            continue
        M, tr = o["methods"], o.get("train", {})
        err = lambda d: d["l2_err"][-1].max() if kind == "pde" else d["l2_err"][1:, 0].max()
        rows = [(lab, M[k]["wall"], err(M[k]), f) for k, f, lab in pts
                if k in M and "l2_err" in M[k] and np.isfinite(M[k].get("wall", np.nan))]
        rows.sort(key=lambda r: r[1])
        data.append((title, rows, err(M[ref]), tr.get("t_pilot", 0.0) + tr.get("t_train", 0.0)))
    hts = [len(r) + 1.6 for _, r, _, _ in data]
    fig, axs = plt.subplots(len(data), 1, figsize=(5.0, 0.15 * sum(hts) + 0.9), sharex=True,
                            gridspec_kw=dict(height_ratios=hts))
    for ax, (title, rows, eref, t_extra) in zip(np.atleast_1d(axs), data):
        n = len(rows)
        for i, (lab, w, er, f) in enumerate(rows):
            y = n - 1 - i
            st = fam[f]
            good = er <= 1.25 * eref
            if f in ("pinn", "hyb") and t_extra > 0:
                ax.plot([w, w + t_extra], [y, y], color=st["color"], lw=0.8, alpha=0.6, zorder=2)
                ax.plot([w + t_extra], [y], ls="none", marker="|", ms=7, color=st["color"], mew=1.2, zorder=2)
            ax.plot([w], [y], ls="none", marker=st["marker"], ms=6 if st["marker"] == "*" else 4.5,
                    color=st["color"], mfc=st["color"] if good else "white", mec=st["color"] if not good else "white",
                    mew=0.9 if not good else 0.5, label=st["label"], zorder=3)
            m, ex = f"{er:.1e}".split("e")
            ax.text(1.015, y, f"{m}e{int(ex)}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7.2,
                    color=INK if good else INK2)
        ax.set_yticks(range(n))
        ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.4)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xscale("log")
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.text(1.015, n - 0.3, "error", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=7.2,
                color=INK2)
    np.atleast_1d(axs)[-1].set_xlabel("wall time of one propagation [s]")
    fig.tight_layout(h_pad=0.6)
    from matplotlib.lines import Line2D
    H = [Line2D([], [], ls="none", marker=st["marker"], color=st["color"], mec="white", mew=0.5,
                ms=6 if st["marker"] == "*" else 4.5, label=st["label"]) for st in fam.values()]
    H.append(Line2D([], [], color="#2a78d6", lw=0.8, marker="|", ms=7, mew=1.2, markevery=[1],
                    label="incl. pilot run and training"))
    H.append(Line2D([], [], ls="none", marker="o", mfc="white", mec=INK2, mew=0.9, ms=4.5,
                    label=r"hollow: error $>1.25\times$ that of the converged closure"))
    fig.legend(handles=H, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.115))
    save(fig, "tradeoff")


def fig_coherence():
    """E13: time per closure call against the spacing of successive calls, implicit laws (one core)."""
    role = {"warm": dict(color="#1baf7a", ls="-", marker="o"), "cold": dict(color="#e34948", ls=":", marker="v"),
            "nkw": dict(color="#008300", ls=(0, (3, 1, 1, 1)), marker="d"),
            "hyb": dict(color="#2a78d6", ls="-", marker="*"), "hyb2": dict(color="#4a3aa7", ls="--", marker="P"),
            "fixed": dict(color="#2a78d6", ls=(0, (1, 1)), marker="x"), "psp": dict(color="#eda100", ls="--", marker="s")}
    panels = [("heat", "heat conduction ($d=8$, 15 nodes)",
               [("chord_w8", "warm", "chord, warm start"), ("nk_w8", "nkw", "NK, warm start"),
                ("nk_c8", "cold", "NK, cold start"), ("pinn_nkc8", "hyb", "network guess + NK"),
                ("pinn_chord8", "hyb2", "network guess + chord"), ("pinn_nk2", "fixed", "network + 2 NK (fixed)"),
                ("psp_s4", "psp", "sparse PSP ($k=4$)")]),
              ("osc", "oscillator ($d=6$)",
               [("newton_w8", "warm", "Newton, warm start"), ("nk_w8", "nkw", "NK, warm start"),
                ("newton_c8", "cold", "Newton, cold start"), ("pinn_newtonc8", "hyb", "network guess + Newton"),
                ("pinn_chord8", "hyb2", "network guess + chord"),
                ("pinn_newton1", "fixed", "network + 1 Newton (fixed)"), ("psp", "psp", "tensor PSP ($q=4$)")])]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.4))
    for ax, (prob, title, meths) in zip(axs, panels):
        out = load_pkl(f"e13_{prob}.pkl")
        if out is None:
            continue
        sp = np.array(out["spacings"])
        x = sp / (out["n_t"] - 1)
        R = out["results"]
        for m, k, lab in meths:
            if (m, sp[0]) not in R:
                continue
            y = [np.median(R[(m, s)]["t_call"][~R[(m, s)]["first"]]) * 1e3 for s in sp]
            ax.plot(x, y, lw=1.4, ms=4.5, label=lab, **role[k])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"spacing of successive calls, $\Delta t/T$")
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=2.0, columnspacing=0.8,
                  fontsize=6.6)
    axs[0].set_ylabel("time per call [ms]")
    fig.tight_layout(w_pad=2.5)
    save(fig, "coherence")


def fig_scaling():
    """E12: cost per node and call of the closures alone against the number of nodes, on one CPU thread (solid, filled)
    and on the GPU (dashed, hollow).  Whole trajectory on the stored meshes (N <= 60), last calls above."""
    p = os.path.join(RES, "e12_scaling.json")
    if not os.path.exists(p):
        return
    idx = {(r["problem"], r["method"], r["N"], r["device"], r["window"]): r for r in json.load(open(p))["records"]}
    fam = {"chord_w8": ("#008300", "h", r"chord, warm, $10^{-8}$"), "pinn_nk2": ("#4a3aa7", "P", "PINN + 2 NK"),
           "pinn": ("#2a78d6", "*", "PINN alone / PINN closure"), "lanczos": ("#1baf7a", "D", "Lanczos ($m=10$)"),
           "taylor4": ("#e87ba4", "<", "Taylor-4"), "psp_s3": ("#eda100", "s", "sparse PSP, $k=p$"),
           "psp_s4": ("#eda100", "^", "sparse PSP, $k=p+1$")}
    panels = [("heat", "heat conduction ($d=8$, $P=165$)", ["chord_w8", "pinn_nk2", "pinn", "psp_s4"]),
              ("ign", "ignition ($d=8$, $P=165$)", ["lanczos", "taylor4", "pinn", "psp_s3", "psp_s4"])]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=True)
    for ax, (pr, title, meths) in zip(axs, panels):
        for m in meths:
            col, mk, lab = fam[m]
            for dev, ls, filled in (("cpu1", "-", True), ("gpu", "--", False)):
                pts = sorted((N, r["t_node_call_us"]) for (pr_, m_, N, d_, w), r in idx.items()
                             if pr_ == pr and m_ == m and d_ == dev and w == ("first" if N <= 60 else "late"))
                if not pts:
                    continue
                N, t = zip(*pts)
                ax.plot(N, t, color=col, ls=ls, lw=1.3, marker=mk, ms=6 if mk == "*" else 4,
                        mfc=col if filled else "white", mec=col if not filled else "white", mew=0.9 if not filled else 0.4,
                        label=lab if dev == "cpu1" else None)
        ax.axvline(125, color=INK2, lw=0.6, ls=":")          # whole trajectory to the left, last calls to the right
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of nodes")
        ax.set_title(title)
        tidy_log(ax)
    axs[0].set_ylabel(r"time per node and call [$\mu$s]")
    from matplotlib.lines import Line2D
    H = [Line2D([], [], color=col, ls="-", lw=1.3, marker=mk, ms=6 if mk == "*" else 4, mec="white", mew=0.4, label=lab)
         for col, mk, lab in (fam[m] for m in ("chord_w8", "pinn_nk2", "pinn", "lanczos", "taylor4", "psp_s3", "psp_s4"))]
    D = [Line2D([], [], color=INK2, ls="-", marker="o", ms=4, mec="white", mew=0.4, label="one CPU thread"),
         Line2D([], [], color=INK2, ls="--", marker="o", ms=4, mfc="white", mew=0.9, label="GPU (double precision)")]
    fig.legend(handles=H, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.135))
    fig.legend(handles=D, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(w_pad=1.5)
    save(fig, "scaling")


def load_pkl(name):
    p = os.path.join(RES, name)
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


if __name__ == "__main__":
    which = sys.argv[1:] or ["e1", "e3", "e4", "e5", "e7", "e8", "cost", "cost_implicit", "tradeoff", "coherence",
                             "scaling"]
    for w in which:
        globals()[f"fig_{w}"]()
