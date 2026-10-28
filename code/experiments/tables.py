"""LaTeX table bodies and the numbers quoted in the text, generated from the result pickles.

Writes paper/tables/numbers.tex (macros \\Tab<Name> with table bodies and one macro per quoted number), loaded by
main.tex and supplement.tex.  Usage: python tables.py
"""
import json
import os
import pickle
import re

import numpy as np

from common import RES, ROOT

TAB = os.path.join(ROOT, "paper", "tables")
os.makedirs(TAB, exist_ok=True)
BODIES, NUM = {}, {}


def e(x, digits=1):
    """5.5e-3 style; '---' for missing values."""
    if x is None or not np.isfinite(x):
        return "---"
    m, ex = f"{x:.{digits}e}".split("e")
    return m if int(ex) == 0 else f"{m}e{int(ex)}"


def tx(x, digits=1):
    """Text form: $5.5\\times10^{-3}$ (plain for exponents -1..2)."""
    if x is None or not np.isfinite(x):
        return "---"
    m, ex = f"{x:.{digits}e}".split("e")
    ex = int(ex)
    if -1 <= ex <= 2:
        return f"{x:.{max(0, digits - ex)}f}" if ex < 0 else f"{x:.{max(0, digits - ex)}f}"
    return f"${m}\\times10^{{{ex}}}$"


def txr(a):
    """Range in text form with a common exponent when possible."""
    if not a:
        return "---"
    lo, hi = min(a), max(a)
    elo, ehi = int(np.floor(np.log10(lo))), int(np.floor(np.log10(hi)))
    if elo == ehi and not (-1 <= elo <= 2):
        return f"${lo / 10 ** elo:.1f}$--${hi / 10 ** elo:.1f}\\times10^{{{elo}}}$"
    return f"{tx(lo)}--{tx(hi)}"


def wall(x):
    if x is None or not np.isfinite(x):
        return "---"
    return f"{x:.2f}" if x < 1 else (f"{x:.1f}" if x < 100 else f"{x:.0f}")


def load(name):
    p = os.path.join(RES, name)
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def body(name, lines):
    BODIES[name] = "\n".join(lines)
    print(f"--- {name}\n" + "\n".join(lines))


def ok(d):
    return d is not None and d.get("ok")


# -----------------------------------------------------------------------------
def closures():
    names = [("exp_d2p4", r"$\exp$"), ("exp_d5p3", r"$\exp$"), ("sincos_d2p6", r"$\sin/\cos$"),
             ("pow2.5_d4p3", r"$(\cdot)^{2.5}$"), ("pow-1_d4p3", r"$(\cdot)^{-1}$")]
    lines = []
    for n, lab in names:
        r = json.load(open(os.path.join(RES, f"e1_{n}.json")))
        q = lambda k: f"{e(np.median(r[k]))} / {e(np.percentile(r[k], 95))}"
        lines.append(f"{lab:16s} & {r['d']} & {r['p']} & {r['P']} & {r['n_params']:,}".replace(",", r"\,")
                     + f" & {r['t_train'] / 60:.0f} & {q('pinn')} & {q('taylor4')} & {q('taylor8')} & {q('lanczos8')}"
                     f" & {q('lanczos10')} & {q('psp_vs_galerkin')}" + r"\\")
    body("TabClosures", lines)


def pendulum():
    o = load("e3_pendulum.pkl")
    M = o["methods"]
    idx = [40, 80, 120]
    rows = [("Galerkin-eig", "eig"), (r"PSP, tensor ($q=7$)", "psp"), (r"Lanczos ($m=10$)", "lanczos")]
    lines = []
    for label, k in rows:
        d = M[k]
        lines.append(f"{label:38s} & {wall(d['wall'])} & " + " & ".join(e(d['l2_err'][i, 0]) for i in idx) + r"\\")
    lines.append(r"Taylor-8                               & --- & \multicolumn{3}{c}{diverges at $t\approx28.8$}\\")
    lines.append(r"PINN closure (generic)                 & --- & \multicolumn{3}{c}{diverges at $t\approx15$}\\")
    d = M["pinn_anchored"]
    lines.append(f"PINN closure (anchored, self-consistent) & --- & {e(d['l2_err'][40, 0])} & {e(d['l2_err'][80, 0])} & "
                 f"--- ({e(d['l2_err'][112, 0])} at $t=28$)" + r"\\")
    d = M["xtfc"]
    lines.append(r"\midrule")
    lines.append(f"{'spectral X-TFC (GH nodes)':38s} & {wall(d['wall'])} & " + " & ".join(e(d['l2_err'][i, 0]) for i in idx) + r"\\")
    body("TabPendulum", lines)
    NUM.update(PENeig=wall(M["eig"]["wall"]), PENpsp=wall(M["psp"]["wall"]), PENlanczos=wall(M["lanczos"]["wall"]),
               PENxtfc=wall(M["xtfc"]["wall"]))


def toggle():
    o = load("e4_toggle.pkl")
    M = o["methods"]
    rows = [("Galerkin-eig", "eig"), (r"PSP, tensor ($q=4$)", "psp"), ("Taylor-4", "taylor4"),
            (r"Lanczos ($m=10$)", "lanczos"), ("PINN closure (generic)", "pinn_generic"), None,
            (r"X-TFC, GH ($Q=256$)", "xtfc_gh"), (r"X-TFC, QMC ($Q=70$)", "xtfc_qmc2P"),
            (r"X-TFC, QMC ($Q=140$)", "xtfc_qmc4P")]
    lines = []
    for r in rows:
        if r is None:
            lines.append(r"\midrule")
            continue
        label, k = r
        d = M[k]
        l2 = [d["l2_err"][1:, j].max() for j in (0, 1)]
        mu = [d["mean_err_p"][-1, j] for j in (0, 1)]
        sd = [d["std_err_p"][-1, j] for j in (0, 1)]
        lines.append(f"{label:24s} & {wall(d['wall'])} & " + " & ".join(e(x) for x in l2 + mu + sd) + r"\\")
    body("TabToggle", lines)
    walls = [M[k]["wall"] for k in ("eig", "psp", "taylor4", "lanczos", "pinn_generic")]
    xw = [M[k]["wall"] for k in ("xtfc_gh", "xtfc_qmc2P", "xtfc_qmc4P")]
    NUM.update(TOGmin=wall(min(walls)), TOGmax=wall(max(walls)), TOGxmin=wall(min(xw)), TOGxmax=wall(max(xw)))


def ignition():
    r5, r8 = load("e5_ignition.pkl"), load("e6_ignition_highd.pkl")
    cases = [r5["d2p4"]["methods"], r5["d5p3"]["methods"], r8["methods"]]
    alias = [{"psp_s": "psp_sparse", "pinn": "pinn_generic"}] * 2 + [{"psp_s": "psp_s3", "pinn": "pinn"}]
    rows = [("Galerkin-eig", "eig"), ("PSP, tensor ($q=p+1$)", "psp"), ("PSP, sparse ($k=p$)", "psp_s"),
            ("PSP, sparse ($k=p+1$)", "psp_s4"), ("Taylor-4", "taylor4"), ("Lanczos ($m=10$)", "lanczos"),
            ("PINN closure", "pinn")]
    lines = []
    for label, key in rows:
        ds = [c.get(a.get(key, key)) for c, a in zip(cases, alias)]
        w = [wall(d["wall"]) if ok(d) else "---" for d in ds]
        l2 = [e(d["l2_err"][-1].max(), 2 if i == 2 else 1) if ok(d) else "---" for i, d in enumerate(ds)]
        lines.append(f"{label:28s} & " + " & ".join(w) + " & " + " & ".join(l2) + r"\\")
    body("TabIgnition", lines)
    m8 = r8["methods"]
    for name, k in [("IGNeig", "eig"), ("IGNpsp", "psp"), ("IGNsparse", "psp_s3"), ("IGNsparseb", "psp_s4"),
                    ("IGNtaylor", "taylor4"), ("IGNlanczos", "lanczos"), ("IGNpinn", "pinn")]:
        NUM[name] = wall(m8[k]["wall"]) if k in m8 else "---"
    NUM["IGNratioPSP"] = f"{m8['psp']['wall'] / m8['pinn']['wall']:.0f}"
    tr8 = r8.get("train", {})
    if "t_train" in tr8:
        NUM["IGNtpilotmin"] = f"{tr8.get('t_pilot', np.nan) / 60:.0f}"
        NUM["IGNttrainmin"] = f"{tr8['t_train'] / 60:.0f}"


# -----------------------------------------------------------------------------
CONVERGED_TOL = {"newton": 1e-12, "nk": 1e-10, "pinn_nkconv": 1e-10, "pinn_newtonconv": 1e-12}


def _implicit_row(label, d, mid=None, xdot=False, key=None):
    if not ok(d):
        return None
    its = d.get("iters", d.get("newton_iters"))
    if its is None or its < 0:
        its_s = "---"
    else:
        its_s = f"{its:.0f}" if abs(its - round(its)) < 1e-9 else f"{its:.1f}"
    cnt = d.get("counters")
    if cnt is not None and cnt[0] > 0:
        per = cnt[2] / cnt[0]
        if cnt[1] == 0:
            work = (f"{per:.2f}" if per >= 0.1 else e(per)) + "\\,J"
        else:
            work = f"{cnt[1] / cnt[0]:.0f}\\,Jv"
    else:
        work = "---"
    fr = d.get("final_res")
    if fr is not None and len(fr):
        acc = e(np.max(fr))
    elif key in CONVERGED_TOL:
        acc = "$\\le$" + e(CONVERGED_TOL[key], 0)
    else:
        acc = "---"
    if xdot:
        l2 = f"{e(d['l2_err'][1:, 0].max())} & {e(d['l2_err'][1:, 1].max())}"
    else:
        l2 = f"{e(d['l2_err'][-1].max())} & {d['bc'][-1, mid]:.4f}"
    return f"{label:44s} & {wall(d['wall'])} & {d['nfev']} & {its_s} & {work} & {acc} & {l2}" + r"\\"


def enthalpy():
    o = load("e8_enthalpy.pkl")
    M = o["methods"]
    mid = len(o["x"]) // 2
    groups = [
        [("Newton, cold, $10^{-12}$ (original)", "newton"), ("NK, cold, $10^{-10}$ (original)", "nk"),
         ("Newton, cold, $10^{-8}$", "newton_c8"), ("NK, cold, $10^{-8}$", "nk_c8")],
        [("Newton, warm, $10^{-8}$", "newton_w8"), ("NK, warm, $10^{-8}$", "nk_w8"),
         ("chord, warm, $10^{-6}$", "chord_w6"), ("chord, warm, $10^{-8}$", "chord_w8"),
         ("chord, warm, $10^{-10}$", "chord_w10")],
        [("PINN alone", "pinn"), ("PINN + 1 NK correction", "pinn_nk1"), ("PINN + 2 NK corrections", "pinn_nk2"),
         ("PINN + NK to $10^{-8}$", "pinn_nkc8"), ("PINN + NK to $10^{-10}$", "pinn_nkconv")],
        [("PSP, tensor ($Q=65\\,536$)", "psp"), ("PSP, sparse $k=3$ ($Q=849$)", "psp_s3"),
         ("PSP, sparse $k=4$ ($Q=3905$)", "psp_s4"), ("PSP, sparse $k=5$", "psp_s5"), ("linearised law", "linear")]]
    lines = []
    for gi, g in enumerate(groups):
        if gi:
            lines.append(r"\midrule")
        for label, k in g:
            d = dict(M.get(k, {}))
            if k == "pinn_nk2" and "pinn_nk2rec" in M:       # achieved residual from the (untimed) recording run
                d["final_res"] = M["pinn_nk2rec"].get("final_res")
            row = _implicit_row(label, d, mid=mid, key=k)
            if row:
                lines.append(row)
    body("TabEnthalpy", lines)
    w = lambda k: M[k]["wall"] if ok(M.get(k)) else np.nan
    for name, k in [("ENTnewton", "newton"), ("ENTnk", "nk"), ("ENTnewtoncold", "newton_c8"), ("ENTnkcold", "nk_c8"),
                    ("ENTnewtonwarm", "newton_w8"), ("ENTnkwarm", "nk_w8"), ("ENTchord", "chord_w8"),
                    ("ENTchordten", "chord_w10"), ("ENTpsp", "psp"), ("ENTsparse", "psp_s3"), ("ENTsparseb", "psp_s4"),
                    ("ENTsparsec", "psp_s5"), ("ENTpinn", "pinn"), ("ENTnkone", "pinn_nk1"), ("ENTnktwo", "pinn_nk2"),
                    ("ENTnkconv", "pinn_nkconv"), ("ENTpinnnkc", "pinn_nkc8"), ("ENTlinear", "linear")]:
        NUM[name] = wall(w(k))
    NUM["ENTratioNK"] = f"{w('nk') / w('pinn_nk2'):.1f}"
    NUM["ENTratioNewton"] = f"{w('newton') / w('pinn_nk2'):.0f}"
    NUM["ENTratioChord"] = f"{w('pinn_nk2') / w('chord_w8'):.1f}"
    NUM["ENTratioChordTen"] = f"{w('pinn_nk2') / w('chord_w10'):.1f}"
    NUM["ENTratioJac"] = f"{w('nk_w8') / w('chord_w8'):.1f}"
    NUM["ENTchordsixNfev"] = f"{M['chord_w6']['nfev']}" if ok(M.get("chord_w6")) else "---"
    NUM["ENTratioSparse"] = f"{w('pinn_nk2') / w('psp_s4'):.1f}"
    NUM["ENTratioCold"] = f"{w('nk_c8') / w('pinn_nk2'):.1f}"
    NUM["ENTratioTol"] = f"{w('nk') / w('nk_c8'):.1f}"
    NUM["ENTratioWarm"] = f"{w('nk_c8') / w('nk_w8'):.1f}"
    cw = M.get("chord_w8", {})
    if "counters" in cw:
        c = cw["counters"]
        NUM["ENTchordJac"] = f"{int(c[2])}"
        NUM["ENTchordCalls"] = f"{int(c[0])}"
        NUM["ENTchordIts"] = f"{cw['iters']:.1f}"
        NUM["ENTwarmInit"] = tx(float(np.median(cw["init_res"])))
    tr = o.get("train", {})
    tp, tt = tr.get("t_pilot", np.nan), tr.get("t_train", np.nan)
    NUM["ENTtpilot"], NUM["ENTttrain"] = wall(tp), wall(tt)
    # amortisation T_total(N) = T_pilot + T_train + N T_prop against classical alternatives
    for name, k in [("Nk", "nk"), ("Nkcold", "nk_c8"), ("Newton", "newton")]:
        gain = w(k) - w("pinn_nk2")
        NUM[f"ENTbreak{name}"] = f"{(tp + tt) / gain:.0f}" if gain > 0 else r"$\infty$"
    # coefficient deviation from the converged solution (max over output times and nodes)
    ref = M.get("chord_w10") if ok(M.get("chord_w10")) else M.get("newton")
    if ok(ref):
        Cr = ref["C"]
        for name, k in [("ENTdevNKtwo", "pinn_nk2"), ("ENTdevChord", "chord_w8"), ("ENTdevNKwarm", "nk_w8"),
                        ("ENTdevSparse", "psp_s4"), ("ENTdevNewton", "newton")]:
            if ok(M.get(k)) and M[k]["C"].shape == Cr.shape:
                dev = np.linalg.norm(M[k]["C"][1:] - Cr[1:], axis=2) / np.linalg.norm(Cr[1:], axis=2)
                NUM[name] = tx(float(dev.max()))
    ot = load("e9_own_traj.pkl")
    if ot:
        NUM["ENTownNKone"] = tx(ot["pinn_nk1"]["err_max"])
        NUM["ENTownNKoneMed"] = tx(ot["pinn_nk1"]["err_med"])
        NUM["ENTresVsErr"] = f"{ot['pinn_nk2']['err_med'] / max(ot['pinn_nk2']['res_med'], 1e-300):.0f}"
    rec = M.get("pinn_nk2rec", {})
    if rec.get("final_res") is not None and len(rec["final_res"]):
        NUM["ENTnktwoRes"] = tx(float(np.median(rec["final_res"])))
        NUM["ENTnktwoResMax"] = tx(float(np.max(rec["final_res"])))
        NUM["ENTpinnInit"] = tx(float(np.median(rec["init_res"])))


def implicit():
    o = load("e7_series_spring.pkl")
    M = o["methods"]
    groups = [
        [("Newton, cold, $10^{-12}$ (original)", "newton"), ("NK, cold, $10^{-10}$ (original)", "nk"),
         ("Newton, cold, $10^{-8}$", "newton_c8"), ("NK, cold, $10^{-8}$", "nk_c8")],
        [("Newton, warm, $10^{-8}$", "newton_w8"), ("NK, warm, $10^{-8}$", "nk_w8"),
         ("chord, warm, $10^{-8}$", "chord_w8"), ("chord, warm, $10^{-10}$", "chord_w10")],
        [("PINN alone", "pinn"), ("PINN + 1 Newton correction", "pinn_newton1"),
         ("PINN + 1 NK correction", "pinn_nk1"), ("PINN + NK to $10^{-8}$", "pinn_nkc8"),
         ("PINN + NK to $10^{-10}$", "pinn_nkconv")],
        [("PSP, tensor ($q=4$, $Q=4096$)", "psp"), ("PSP, tensor ($q=5$, $Q=15\\,625$)", "psp5"),
         ("PSP, sparse $k=3$ ($Q=389$)", "psp_s3"), ("PSP, sparse $k=4$ ($Q=1433$)", "psp_s4"),
         ("PSP, sparse $k=5$", "psp_s5"), ("linearised constraint", "linear")]]
    lines = []
    for gi, g in enumerate(groups):
        if gi:
            lines.append(r"\midrule")
        for label, k in g:
            d = dict(M.get(k, {}))
            if k == "pinn_newton1" and "pinn_newton1rec" in M:
                d["final_res"] = M["pinn_newton1rec"].get("final_res")
            row = _implicit_row(label, d, xdot=True, key=k)
            if row:
                lines.append(row)
    body("TabImplicit", lines)
    w = lambda k: M[k]["wall"] if ok(M.get(k)) else np.nan
    for name, k in [("OSCnewton", "newton"), ("OSCnk", "nk"), ("OSCnewtoncold", "newton_c8"),
                    ("OSCnewtonwarm", "newton_w8"), ("OSCchord", "chord_w8"), ("OSCpsp", "psp"),
                    ("OSCpspfive", "psp5"), ("OSCsparse", "psp_s3"), ("OSCsparseb", "psp_s4"), ("OSCpinn", "pinn"),
                    ("OSCpinnN", "pinn_newton1"), ("OSCpinnNK", "pinn_nk1"), ("OSCnkconv", "pinn_nkconv")]:
        NUM[name] = wall(w(k))
    tr = o.get("train", {})
    tp, tt = tr.get("t_pilot", np.nan), tr.get("t_train", np.nan)
    NUM["OSCtpilot"], NUM["OSCttrain"] = wall(tp), wall(tt)
    NUM["OSCttrainmin"] = f"{tt / 60:.0f}"
    best = min(w("newton_w8"), w("chord_w8"), w("nk_w8"))
    NUM["OSCbest"] = wall(best)
    NUM["OSCratioBest"] = f"{best / w('pinn_newton1'):.1f}"
    gain = best - w("pinn_newton1")
    NUM["OSCbreak"] = f"{(tp + tt) / gain:.0f}" if gain > 0 else r"$\\infty$"
    NUM["OSCratioOrig"] = f"{w('newton') / w('pinn_newton1'):.1f}"


def seeds():
    o7, o8 = load("e7_series_spring.pkl"), load("e8_enthalpy.pkl")
    v7, v8, a8, b8, b7 = [], [], [], [], []
    for k in (0, 1, 2):
        sfx = "" if k == 0 else f"_seed{k}"
        t7, t8 = o7.get("train" if k == 0 else f"train_seed{k}"), o8.get("train" if k == 0 else f"train_seed{k}")
        if t7:
            v7.append(t7["val"])
        if t8:
            v8.append(t8["val"])
        if ok(o8["methods"].get("pinn" + sfx)):
            a8.append(o8["methods"]["pinn" + sfx]["l2_err"][-1].max())
        if ok(o8["methods"].get("pinn_nk2" + sfx)):
            b8.append(o8["methods"]["pinn_nk2" + sfx]["l2_err"][-1].max())
        if ok(o7["methods"].get("pinn_newton1" + sfx)):
            b7.append(o7["methods"]["pinn_newton1" + sfx]["l2_err"][1:, 0].max())
    rng = txr
    NUM.update(SEEDvalOsc=rng(v7), SEEDvalEnt=rng(v8), SEEDpinnEnt=rng(a8), SEEDhybEnt=rng(b8), SEEDhybOsc=rng(b7))
    s1 = []
    for k in (1, 2):
        p = os.path.join(RES, f"e1s_exp_d2p4_seed{k}.json")
        if os.path.exists(p):
            s1.append(json.load(open(p)))
    base = json.load(open(os.path.join(RES, "e1_exp_d2p4.json")))
    meds = [float(np.median(base["pinn"]))] + [s["med"] for s in s1]
    NUM["SEEDexplicit"] = rng(meds)


def checks():
    """Error-attribution checks and quoted numbers."""
    lines = []
    pi = load("e9_pconv_ign.pkl")
    if pi:
        for name, lab in [("eig", r"ignition, $d=2$: $\cG_f$ (eig)"), ("psp", r"ignition, $d=2$: tensor PSP")]:
            vals = {r[0]: r[2] for r in pi if r[1] == name}
            lines.append(f"{lab} & " + " & ".join(e(vals.get(p)) for p in range(2, 7)) + r"\\")
    o8 = load("e8_enthalpy.pkl")
    pe = {p: load(f"e9_pconv_enth_p{p}.pkl") for p in (2, 4)}
    gal = {p: (pe[p]["methods"]["galerkin"]["l2"] if pe[p] and "galerkin" in pe[p]["methods"] else None) for p in (2, 4)}
    psp = {p: (pe[p]["methods"][f"psp_s{p + 1}"]["l2"] if pe[p] and f"psp_s{p + 1}" in pe[p]["methods"] else None)
           for p in (2, 4)}
    if o8 and ok(o8["methods"].get("chord_w10")):
        gal[3] = o8["methods"]["chord_w10"]["l2_err"][-1].max()
    if o8 and ok(o8["methods"].get("psp_s4")):
        psp[3] = o8["methods"]["psp_s4"]["l2_err"][-1].max()
    lines.append(r"\midrule")
    lines.append(r"heat conduction, $d=8$: Galerkin & " + " & ".join(e(gal.get(p)) for p in (2, 3, 4)) + r" & & \\")
    lines.append(r"heat conduction, $d=8$: sparse PSP, $k=p+1$ & " + " & ".join(e(psp.get(p)) for p in (2, 3, 4)) + r" & & \\")
    body("TabChecks", lines)
    t = load("e9_tol.pkl")
    if t:
        NUM.update(TOLign=tx(t["ign_ipc"]), TOLenth=tx(t["enth_ipc"]), TOLmcign=tx(t["ign_mc"]), TOLmcenth=tx(t["enth_mc"]))
    g = load("e9_ghconv.pkl")
    if g:
        NUM["GHexpTwo"], NUM["GHexpTwelve"] = tx(g["exp"][0, 1]), tx(g["exp"][10, 1])
        NUM["GHpspTwelve"] = tx(g["exp"][10, 2])
        NUM["GHsinTwelve"] = tx(g["sin"][10, 1])
        NUM["GHassoc"] = f"{100 * g['assoc4'][0]:.0f}"
    for Q, tag in ((12, "A"), (18, "B")):
        r = load(f"e9_ood_enth_Q{Q}.pkl")
        if r:
            m = r["methods"]
            NUM[f"OOD{tag}pinn"] = tx(m["pinn"].get("coef_err", np.nan))
            NUM[f"OOD{tag}hyb"] = tx(m["pinn_nk2"].get("coef_err", np.nan))
            NUM[f"OOD{tag}hybres"] = tx(m["pinn_nk2"].get("final_res_max", np.nan))
            NUM[f"OOD{tag}its"] = f"{m['pinn_nkc8'].get('iters', np.nan):.1f}"
            NUM[f"OOD{tag}chordits"] = f"{m['chord_w8'].get('iters', np.nan):.1f}"
    ox = load("e9_ood_explicit.pkl")
    if ox:
        NUM["OODexpHalf"], NUM["OODexpEdge"] = tx(ox["exp_d2p4"][0][1]), tx(ox["exp_d2p4"][1][1])
        NUM["OODexpOut"] = tx(ox["exp_d2p4"][3][1])
        NUM["OODsinHalf"], NUM["OODsinEdge"] = tx(ox["sincos_d2p6"][0][1]), tx(ox["sincos_d2p6"][1][1])
        NUM["OODsinOut"] = tx(ox["sincos_d2p6"][3][1])


def bench():
    rows = json.load(open(os.path.join(RES, "bench_cost.json")))
    r = [x for x in rows if x["d"] == 8 and x["p"] == 3][0]
    ms = lambda v: f"{1e3 * v:.1f}" if 1e3 * v >= 1 else f"{1e3 * v:.2f}"
    for name, k in [("BENCHeig", "eig"), ("BENCHpsp", "psp"), ("BENCHsparse", "psp_s0"), ("BENCHsparseb", "psp_s1"),
                    ("BENCHtaylor", "taylor4"), ("BENCHtaylorb", "taylor8"), ("BENCHlanczos", "lanczos10"),
                    ("BENCHpinn", "pinn")]:
        NUM[name] = ms(r[k])
    NUM["BENCHratioSparse"] = f"{r['pinn'] / r['psp_s0']:.0f}"
    rows = json.load(open(os.path.join(RES, "bench_implicit.json")))
    r = [x for x in rows if x["d"] == 8][0]
    for name, k in [("BENCHinewton", "newton"), ("BENCHink", "nk"), ("BENCHipsp", "psp"), ("BENCHisparse", "psp_s0"),
                    ("BENCHisparseb", "psp_s1"), ("BENCHipinn", "pinn"), ("BENCHihybone", "pinn_nk1"),
                    ("BENCHihybtwo", "pinn_nk2")]:
        NUM[name] = ms(r[k] / r["batch"])


def supplement_tables():
    g = load("e9_ghconv.pkl")
    if g:
        lines = []
        for i, p in enumerate(g["exp"][:, 0].astype(int)):
            if p % 2:
                continue
            lines.append(f"{p} & {e(g['exp'][i, 1])} & {e(g['exp'][i, 2])} & {e(g['sin'][i, 1])} & {e(g['sin'][i, 2])}" + r"\\")
        body("TabGH", lines)
        lines = []
        for row in g["sqrt"]:
            lines.append(f"{int(row[0])} & {row[1]:.2f} & {e(row[2])} & {e(row[3])} & {e(row[4])} & {e(row[5])}" + r"\\")
        body("TabSqrt", lines)
    ox = load("e9_ood_explicit.pkl")
    if ox:
        lab = {"exp_d2p4": r"$\exp$, $d=2$", "exp_d5p3": r"$\exp$, $d=5$", "sincos_d2p6": r"$\sin/\cos$, $d=2$",
               "pow2.5_d4p3": r"$(\cdot)^{2.5}$, $d=4$", "pow-1_d4p3": r"$(\cdot)^{-1}$, $d=4$"}
        lines = []
        for k, rows in ox.items():
            lines.append(f"{lab[k]} & " + " & ".join(f"{e(r[1])} / {e(r[2])}" for r in rows) + r"\\")
        body("TabOODexp", lines)


def capacity():
    """Network size / training-budget study (Supplementary Table) and quoted numbers.  Training times are not
    tabulated: the trainings of the study ran six at a time on four cores."""
    import glob
    lines = []
    ns = lambda n: f"{int(n):,}".replace(",", "\\,")
    cfg = lambda w, dpt: f"{dpt}$\\times${w}"
    # explicit closures: test error w.r.t. G_f (median / 95th percentile, Table-1 test distribution)
    o3 = load("e3_pendulum.pkl")
    pend = {"base": r"diverges, $t\approx15$"}
    for tag in ("sincos_d2p6_w256d4b4", "sincos_d2p6_w256d4b4_R10"):
        pth = os.path.join(RES, f"e10_pendulum_{tag}.json")
        if os.path.exists(pth):
            r = json.load(open(pth))
            pend[tag] = "reaches $t=30$" if r["ok"] else f"diverges, $t\\approx{r['t_end']:.0f}$"
    lab = {"exp_d2p4": r"$\exp$, $d=2$", "sincos_d2p6": r"$\sin/\cos$, $d=2$"}
    best = {}
    for name in ("exp_d2p4", "sincos_d2p6"):
        base = json.load(open(os.path.join(RES, f"e1_{name}.json")))
        rows = [(64, 3, 1.0, "", base["n_params"], float(np.median(base["pinn"])), float(np.percentile(base["pinn"], 95)),
                 "base")]
        for f in sorted(glob.glob(os.path.join(RES, f"e10_{name}_w*.json")), key=lambda f: json.load(open(f))["n_params"]):
            r = json.load(open(f))
            note = f", $R_{{\\mathrm{{eff}}}}={r['R_eff']:g}$" if "test_own" in r else ""
            rows.append((r["width"], r["depth"], r["budget"], note, r["n_params"], r["test_base"][0], r["test_base"][1],
                         r["tag"]))
        for i, (w, dpt, b, note, npar, med, p95, tag) in enumerate(rows):
            prop = pend.get(tag, "---") if name == "sincos_d2p6" else "---"
            lines.append(f"{lab[name] if i == 0 else ''} & {cfg(w, dpt)}, {b:g}$\\times${note} & {ns(npar)} & "
                         f"{e(med)} / {e(p95)} & {prop}" + r"\\")
        best[name] = min(rows, key=lambda x: x[5])
        base_row = rows[0]
        NUM["CAP" + ("exp" if name == "exp_d2p4" else "sin") + "Base"] = tx(base_row[5])
        lines.append(r"\midrule")
    # implicit networks: validation error (median) and propagation errors
    o8 = load("e8_enthalpy.pkl")
    M8 = o8["methods"] if o8 else {}
    tr = o8.get("train", {}) if o8 else {}
    rows = [(128, 3, 1.0, tr.get("n_params", np.nan), tr.get("val", np.nan), "")]
    for f in sorted(glob.glob(os.path.join(RES, "e10_enthalpy_d8p3_w*.json")), key=lambda f: json.load(open(f))["n_params"]):
        r = json.load(open(f))
        rows.append((r["width"], r["depth"], r["budget"], r["n_params"], r["val"], f"@w{r['width']}d{r['depth']}b{r['budget']:g}"))
    L8 = lambda k: e(M8[k]["l2_err"][-1].max()) if ok(M8.get(k)) else "---"
    for i, (w, dpt, b, npar, v, sfx) in enumerate(rows):
        lines.append(f"{'heat conduction' if i == 0 else ''} & {cfg(w, dpt)}, {b:g}$\\times$ & {ns(npar)} & {e(v)} & "
                     f"{L8('pinn' + sfx)} / {L8('pinn_nk1' + sfx)} / {L8('pinn_nk2' + sfx)}" + r"\\")
    bestE = min(rows, key=lambda x: x[4])
    lines.append(r"\midrule")
    o7 = load("e7_series_spring.pkl")
    M7 = o7["methods"] if o7 else {}
    tr7 = o7.get("train", {}) if o7 else {}
    rows7 = [(64, 3, 1.0, tr7.get("n_params", np.nan), tr7.get("val", np.nan), "")]
    for f in sorted(glob.glob(os.path.join(RES, "e10_implicit_d6p3_w*.json"))):
        r = json.load(open(f))
        rows7.append((r["width"], r["depth"], r["budget"], r["n_params"], r["val"], f"@w{r['width']}d{r['depth']}b{r['budget']:g}"))
    L7 = lambda k: e(M7[k]["l2_err"][-1, 0]) if ok(M7.get(k)) else "---"
    for i, (w, dpt, b, npar, v, sfx) in enumerate(rows7):
        lines.append(f"{'oscillator' if i == 0 else ''} & {cfg(w, dpt)}, {b:g}$\\times$ & {ns(npar)} & {e(v)} & "
                     f"{L7('pinn' + sfx)} / {L7('pinn_newton1' + sfx)}" + r"\\")
    body("TabCapacity", lines)
    # quoted numbers
    for name, key in (("exp_d2p4", "CAPexp"), ("sincos_d2p6", "CAPsin")):
        b = best[name]
        NUM[key] = tx(b[5])
        NUM[key + "Cfg"] = f"{cfg(b[0], b[1])}, {b[2]:g}$\\times$"
        NUM[key + "Par"] = ns(b[4])
    for tag, key in (("exp_d2p4_w256d4b4", "CAPexpBig"), ("exp_d2p4_w64d3b4", "CAPexpBudget"),
                     ("sincos_d2p6_w256d4b4", "CAPsinBig")):
        pth = os.path.join(RES, f"e10_{tag}.json")
        if os.path.exists(pth):
            r = json.load(open(pth))
            NUM[key] = tx(r["test_base"][0])
            NUM[key + "Pn"] = tx(r["test_base"][1])
            NUM[key + "Par"] = ns(r["n_params"])
    NUM["CAPent"] = tx(bestE[4])
    NUM["CAPentBase"] = tx(rows[0][4])
    NUM["CAPentCfg"] = f"{cfg(bestE[0], bestE[1])}, {bestE[2]:g}$\\times$"
    NUM["CAPentMaxPar"] = ns(max(r[3] for r in rows))
    NUM["CAPentValMin"], NUM["CAPentValMax"] = tx(min(r[4] for r in rows[1:])), tx(max(r[4] for r in rows[1:]))
    caps = [r[5] for r in rows[1:]]
    if caps and all(ok(M8.get("pinn" + c)) for c in caps):
        al = [M8["pinn" + c]["l2_err"][-1].max() for c in caps]
        NUM["CAPentPinnMin"], NUM["CAPentPinnMax"] = tx(min(al)), tx(max(al))
        w2 = [M8["pinn_nk2" + c]["wall"] for c in caps if ok(M8.get("pinn_nk2" + c))]
        l2 = [M8["pinn_nk2" + c]["l2_err"][-1].max() for c in caps if ok(M8.get("pinn_nk2" + c))]
        NUM["CAPentNKtwoWmin"], NUM["CAPentNKtwoWmax"] = wall(min(w2)), wall(max(w2))
        NUM["CAPentNKtwoL"] = tx(max(l2))
        one = {c: M8["pinn_nk1" + c] for c in caps if ok(M8.get("pinn_nk1" + c))}
        cb = min(one, key=lambda c: one[c]["l2_err"][-1].max())
        NUM["CAPentNKoneL"] = tx(one[cb]["l2_err"][-1].max())
        NUM["CAPentNKoneW"] = wall(one[cb]["wall"])
        NUM["CAPentNKoneCfg"] = re.sub(r"@w(\d+)d(\d+)b([\d.]+)", lambda m_: f"{m_.group(2)}$\\times${m_.group(1)}, {m_.group(3)}$\\times$", cb)
        oth = [one[c]["l2_err"][-1].max() for c in one if c != cb]
        if oth:
            NUM["CAPentNKoneOthMin"], NUM["CAPentNKoneOthMax"] = tx(min(oth)), tx(max(oth))
    if len(rows7) > 1:
        sfx = rows7[-1][5]
        NUM["CAPosc"] = tx(rows7[-1][4])
        NUM["CAPoscBase"] = tx(rows7[0][4])
        NUM["CAPoscPar"] = ns(rows7[-1][3])
        for k, nm in (("pinn" + sfx, "CAPoscPinn"), ("pinn_newton1" + sfx, "CAPoscNewton")):
            d = M7.get(k)
            if ok(d):
                NUM[nm + "L"] = tx(d["l2_err"][-1, 0])
                NUM[nm + "Lv"] = tx(d["l2_err"][-1, 1])
                NUM[nm + "W"] = wall(d["wall"])
                NUM[nm + "Cl"] = tx(np.max(d["closure_err"])) if "closure_err" in d else r"\textbf{??}"
    for tag, nm in (("sincos_d2p6_w256d4b4", "PENcapA"), ("sincos_d2p6_w256d4b4_R10", "PENcapB")):
        pth = os.path.join(RES, f"e10_pendulum_{tag}.json")
        if os.path.exists(pth):
            r = json.load(open(pth))
            NUM[nm + "End"] = f"{r['t_end']:.0f}"
            l2 = r["l2"]
            NUM[nm + "Ten"] = tx(l2.get("10", np.nan))
            NUM[nm + "Twenty"] = tx(l2.get("20", np.nan))


def parallel():
    """Wall times on 4 and 8 cores next to the single-core ones."""
    WN = {4: "Four", 8: "Eight"}                   # LaTeX macro names cannot contain digits
    DEVPAR = []
    lines = []
    o8, o6, o7 = load("e8_enthalpy.pkl"), load("e6_ignition_highd.pkl"), load("e7_series_spring.pkl")
    sets = [("heat conduction", o8, [("Newton, cold, $10^{-12}$", "newton"), ("NK, cold, $10^{-10}$", "nk"),
                                     ("Newton, warm, $10^{-8}$", "newton_w8"), ("NK, warm, $10^{-8}$", "nk_w8"),
                                     ("chord, warm, $10^{-8}$", "chord_w8"), ("chord, warm, $10^{-10}$", "chord_w10"),
                                     ("PINN + 2 NK", "pinn_nk2"), ("PINN + NK to $10^{-8}$", "pinn_nkc8"),
                                     ("PSP, tensor", "psp"), ("PSP, sparse $k=3$", "psp_s3"), ("PSP, sparse $k=4$", "psp_s4")]),
            ("ignition, $d=8$", o6, [("Galerkin-eig", "eig"), ("PSP, tensor", "psp"), ("PSP, sparse $k=p$", "psp_s3"),
                                     ("PSP, sparse $k=p+1$", "psp_s4"), ("Taylor-4", "taylor4"), ("Lanczos", "lanczos"),
                                     ("PINN closure", "pinn")]),
            ("oscillator", o7, [("Newton, cold, $10^{-12}$", "newton"), ("Newton, warm, $10^{-8}$", "newton_w8"),
                                               ("chord, warm, $10^{-8}$", "chord_w8"), ("PINN + 1 Newton", "pinn_newton1"),
                                               ("PSP, tensor", "psp"), ("PSP, sparse $k=3$", "psp_s3")])]
    for gi, (prob, o, rows) in enumerate(sets):
        if not o:
            continue
        if gi:
            lines.append(r"\midrule")
        first = True
        for lab, k in rows:
            d = o["methods"].get(k, {})
            if not ok(d):
                continue
            w1, w4, w8 = d.get("wall"), d.get("wall_par4", np.nan), d.get("wall_par8", np.nan)
            sp = lambda w: f"{w1 / w:.1f}" if np.isfinite(w) else "---"
            lines.append(f"{prob if first else ''} & {lab} & {wall(w1)} & {wall(w4)} & {sp(w4)} & {wall(w8)} & {sp(w8)}"
                         + r"\\")
            first = False
            for W in (4, 8):
                if f"dev_par{W}" in d:
                    DEVPAR.append(d[f"dev_par{W}"])
    body("TabParallel", lines)
    if DEVPAR:
        NUM["PARdevMax"] = tx(max(DEVPAR))
    if o8:
        M = o8["methods"]
        g = lambda k, W: M.get(k, {}).get(f"wall_par{W}", np.nan)
        for W in (4, 8):
            NUM[f"PARchord{WN[W]}"] = wall(g("chord_w8", W))
            NUM[f"PARchordten{WN[W]}"] = wall(g("chord_w10", W))
            NUM[f"PARhyb{WN[W]}"] = wall(g("pinn_nk2", W))
            NUM[f"PARsparse{WN[W]}"] = wall(g("psp_s4", W))
            NUM[f"PARpsp{WN[W]}"] = wall(g("psp", W))
            NUM[f"PARnewtonw{WN[W]}"] = wall(g("newton_w8", W))
            NUM[f"PARnewton{WN[W]}"] = wall(g("newton", W))
        NUM["PARratioChordHyb"] = f"{g('pinn_nk2', 4) / g('chord_w8', 4):.1f}"
    if o6:
        M = o6["methods"]
        for W in (4, 8):
            NUM[f"PARigneig{WN[W]}"] = wall(M.get("eig", {}).get(f"wall_par{W}", np.nan))
            NUM[f"PARignpinn{WN[W]}"] = wall(M.get("pinn", {}).get(f"wall_par{W}", np.nan))
            NUM[f"PARignsparse{WN[W]}"] = wall(M.get("psp_s3", {}).get(f"wall_par{W}", np.nan))
            NUM[f"PARigntaylor{WN[W]}"] = wall(M.get("taylor4", {}).get(f"wall_par{W}", np.nan))
    o4 = load("e4_toggle.pkl")
    if o4:
        for W in (4, 8):
            NUM[f"PARtogx{WN[W]}"] = wall(o4["methods"]["xtfc_gh"].get(f"wall_par{W}", np.nan))


def enhanced():
    """Fourier features and residual-based adaptive sampling (e11): sigma sweep and full-budget comparison."""
    import glob
    probs = [("exp", "exp_d2p4", r"$\exp$, $d=2$"), ("sin", "sincos_d2p6", r"$\sin/\cos$, $d=2$"),
             ("ent", "enthalpy_d8p3", "heat conduction"), ("osc", "implicit_d6p3", "oscillator")]
    rd = lambda f: json.load(open(f))
    ns = lambda n: f"{int(n):,}".replace(",", "\\,")
    # sigma sweep (short runs, validation error)
    sig = [0.25, 0.5, 1.0, 2.0, 4.0]
    lines, gains = [], []
    for key, base, lab in probs:
        rows = {r["variant"]: r for r in map(rd, glob.glob(os.path.join(RES, f"e11_{base}_*_s0.25.json")))}
        if not rows:
            continue
        vals = [rows.get("P", {}).get("val", np.nan)] + [rows.get(f"F{x:g}", {}).get("val", np.nan) for x in sig]
        best = int(np.nanargmin(vals[1:])) + 1 if np.isfinite(vals[1:]).any() else -1
        cells = [(r"\textbf{" + e(v) + "}") if i == best else e(v) for i, v in enumerate(vals)]
        lines.append(f"{lab} & " + " & ".join(cells) + r"\\")
        if best > 0:
            gains.append(1 - vals[best] / vals[0])
    body("TabSweep", lines)
    if gains:
        NUM["ENHsweepMin"], NUM["ENHsweepMax"] = f"{100 * min(gains):.0f}", f"{100 * max(gains):.0f}"
    # full budget
    o3, o7, o8 = load("e3_pendulum.pkl"), load("e7_series_spring.pkl"), load("e8_enthalpy.pkl")
    M7 = o7["methods"] if o7 else {}
    M8 = o8["methods"] if o8 else {}
    lines, ratios, ends, rg = [], [], [], {}
    add = lambda k, x: rg.setdefault(k, []).append(x) if np.isfinite(x) else None
    for gi, (key, base, lab) in enumerate(probs):
        full = {r["variant"]: r for r in map(rd, glob.glob(os.path.join(RES, f"e11_{base}_*.json"))) if r["frac"] == 1}
        if not full:
            continue
        if gi and lines:
            lines.append(r"\midrule")
        vs = sorted(full, key=lambda v: (v.startswith("F"), v.endswith("A") and v.startswith("F")))   # A, F.., F..A
        if key in ("exp", "sin"):
            b0 = rd(os.path.join(RES, f"e1_{base}.json"))
            rows = [("plain", b0["n_params"], float(np.median(b0["pinn"])), float(np.percentile(b0["pinn"], 95)), "")]
        elif key == "ent":
            tr = o8.get("train", {})
            rows = [("plain", tr.get("n_params", np.nan), tr.get("val", np.nan), None, "")]
        else:
            tr = o7.get("train", {})
            rows = [("plain", tr.get("n_params", np.nan), tr.get("val", np.nan), None, "")]
        for v in vs:
            r = full[v]
            med, p95 = (r["test_base"] if "test_base" in r else (r["val"], None))
            rows.append((v, r["n_params"], med, p95, v))
        NUM[f"ENH{key}Base"] = tx(rows[0][2])
        for v, npar, med, p95, sfx in rows:
            name = {"plain": "plain", "A": "adaptive (RAD)"}.get(v, v)
            if v.startswith("F"):
                sg = re.match(r"F([\d.]+)", v).group(1)
                name = f"Fourier ($\\sigma={sg}$)" + (" + RAD" if v.endswith("A") else "")
                NUM[f"ENH{key}Sigma"] = sg
            code = {"plain": "P", "A": "A"}.get(v, "FA" if v.endswith("A") else "F")
            if code != "P":
                NUM[f"ENH{key}{code}"] = tx(med)
                if p95 is not None:
                    NUM[f"ENH{key}{code}Pn"] = tx(p95)
            if code != "P":
                ratios.append(med / rows[0][2])
                add(key, med)
            err = e(med) + ("" if p95 is None else f" / {e(p95)}")
            sx = "" if v == "plain" else "@" + v
            if key == "sin":
                if v == "plain":
                    prop = r"diverges, $t\approx15$"
                else:
                    pth = os.path.join(RES, f"e10_pendulum_sincos_d2p6_{v}.json")
                    if os.path.exists(pth):
                        pr = rd(pth)
                        prop = "reaches $t=30$" if pr["ok"] else f"diverges, $t\\approx{pr['t_end']:.0f}$"
                        NUM[f"ENHpen{code}End"] = f"{pr['t_end']:.0f}"
                        ends.append(pr["t_end"])
                        add("penTen", pr["l2"].get("10", np.nan))
                        NUM[f"ENHpen{code}Ten"] = tx(pr["l2"].get("10", np.nan))
                    else:
                        prop = "---"
            elif key == "ent":
                L = lambda k: M8[k]["l2_err"][-1].max() if ok(M8.get(k)) else np.nan
                ls = [L(m + sx) for m in ("pinn", "pinn_nk1", "pinn_nk2")]
                prop = " / ".join(e(x) for x in ls)
                if code != "P":
                    for nm, x in zip(("Pinn", "NKone", "NKtwo"), ls):
                        NUM[f"ENHent{code}{nm}L"] = tx(x)
                        add("ent" + nm, x)
                    if ok(M8.get("pinn_nk1" + sx)):
                        NUM[f"ENHent{code}NKoneW"] = wall(M8["pinn_nk1" + sx]["wall"])
            elif key == "osc":
                L = lambda k, c=0: M7[k]["l2_err"][-1, c] if ok(M7.get(k)) else np.nan
                prop = " / ".join(e(L(m + sx)) for m in ("pinn", "pinn_newton1"))
                if code != "P":
                    NUM[f"ENHosc{code}PinnL"] = tx(L("pinn" + sx))
                    NUM[f"ENHosc{code}PinnLv"] = tx(L("pinn" + sx, 1))
                    add("oscPinn", L("pinn" + sx)); add("oscNewton", L("pinn_newton1" + sx))
                    if ok(M7.get("pinn" + sx)):
                        NUM[f"ENHosc{code}PinnW"] = wall(M7["pinn" + sx]["wall"])
            else:
                prop = "---"
            lines.append(f"{lab if v == 'plain' else ''} & {name} & {ns(npar)} & {err} & {prop}" + r"\\")
    body("TabEnhanced", lines)
    for k, v in rg.items():
        NUM[f"ENH{k}R"] = txr(v)
    if ends:
        NUM["ENHpenMin"], NUM["ENHpenMax"] = f"{min(ends):.0f}", f"{max(ends):.0f}"
    if ratios:
        NUM["ENHratioMin"], NUM["ENHratioMax"] = f"{min(ratios):.2f}", f"{max(ratios):.2f}"


def numbers_file():
    # every macro used in the manuscript but not computed (missing result file) becomes a visible placeholder
    import re
    used = set()
    for root in (os.path.join(ROOT, "paper"), os.path.join(ROOT, "paper", "sections"),
                 os.path.join(ROOT, "paper", "sections_pem")):
        for f in os.listdir(root):
            if f.endswith(".tex"):
                used |= set(re.findall(r"\\((?:IGN|ENT|OSC|PEN|TOG|SEED|OOD|GH|TOL|BENCH|PCONV|CAP|PAR|ENH)[A-Za-z]*)",
                                       open(os.path.join(root, f), encoding="utf8").read()))
    for k in sorted(used - set(NUM)):
        NUM[k] = r"\textbf{??}"
        print("MISSING", k)
    for k in ("TabClosures", "TabPendulum", "TabToggle", "TabIgnition", "TabEnthalpy", "TabImplicit", "TabChecks",
              "TabGH", "TabSqrt", "TabOODexp", "TabCapacity", "TabParallel", "TabSweep", "TabEnhanced"):
        BODIES.setdefault(k, r"\multicolumn{2}{l}{\textbf{??}}\\")
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in NUM.items()]
    for k, v in BODIES.items():
        lines.append(f"\\newcommand{{\\{k}}}{{%\n{v}}}")
    with open(os.path.join(TAB, "numbers.tex"), "w") as fh:
        fh.write("% generated by code/experiments/tables.py -- do not edit\n" + "\n".join(lines) + "\n")
    print("--- numbers\n" + "\n".join(f"{k} = {v}" for k, v in NUM.items()))


if __name__ == "__main__":
    for f in (closures, pendulum, toggle, ignition, enthalpy, implicit, seeds, checks, bench, supplement_tables,
              capacity, parallel, enhanced):
        try:
            f()
        except Exception as ex:                       # keep going: tables are independent
            import traceback
            traceback.print_exc()
            print(f"{f.__name__}: {ex!r}")
    numbers_file()
