# Matrix-function closures of non-polynomial terms in intrusive polynomial chaos

Code, stored results and LaTeX sources of the paper

> R. Echeverría, A. Delgado, A. García-Gutiérrez. *Matrix-function closures of non-polynomial terms in intrusive
> polynomial chaos: formulation, Krylov evaluation and an assessment of physics-informed neural closures.*
> Manuscript in preparation (Universidad de León).

Intrusive (stochastic Galerkin) polynomial chaos needs a closure for every non-polynomial term at every right-hand-side
evaluation.  The repository implements and compares, on five uncertainty-quantification benchmarks with up to eight
random dimensions, the matrix-function closure f(M_u)e_0 evaluated by eigendecomposition, Galerkin Taylor series and
Lanczos; physics-informed neural closures trained on a ray ODE; Newton, Newton-Krylov and chord solvers and
residual-trained networks for implicit constitutive laws; pseudo-spectral projection on tensor and Smolyak sparse grids;
and a solution-level spectral X-TFC solver, all against Monte Carlo.

Everything is written in NumPy/SciPy, including the neural networks and their derivatives (no deep-learning framework).

## Contents

```
code/
  pce.py               Hermite PC basis, triple-product tensor, Galerkin products, O(nnz) multiplication matrices,
                       tensor and Smolyak sparse Gauss-Hermite rules (q = "s<k>" gives the sparse rule of level k)
  closures.py          explicit nonlinearities (exp, sin/cos, powers): eigendecomposition, PSP, Taylor, Lanczos,
                       physics-informed neural closure (ray-ODE loss)
  implicit.py          oscillator constraint: Newton, Newton-Krylov, chord (CubicSolver), Cardano PSP, PINNImplicit
  implicit_field.py    heat-conduction constitutive law at every node: FieldNewton, FieldNewtonKrylov, FieldSolver,
                       FieldPSP, FieldPINN
  mlp.py               MLP with forward-mode tangents, hand-written backpropagation, Adam; random Fourier features,
                       residual-based adaptive sampling (RAD)
  xtfc.py              solution-level spectral X-TFC (Levenberg-Marquardt, window marching)
  problems.py          pendulum, genetic toggle switch, solid-fuel ignition, oscillator with an implicit spring force,
                       heat conduction with an implicit constitutive law
  propagate.py         vectorised Monte Carlo, intrusive time integration, error metrics
  parallel.py          NodeParallel: nodes of a PDE distributed over persistent worker processes (Linux, fork)
  bench_cost.py, bench_implicit.py    per-call cost benchmarks
  tests/test_core.py   18 tests: identities of Section 2, every hand-written derivative against finite differences,
                       serial = parallel, Fourier features and adaptive sampling
  tools/               texcount.py (word count), bench_sprod.py (Galerkin-product cost against batch size)
  experiments/         one script per experiment (see the table below); tables.py and figures.py generate every
                       table body, every number quoted in the text and every figure from results/;
                       sessions/ holds the orchestration scripts of the computing sessions actually run
results/               collected results (*.pkl, *.json), trained networks (models/*.npz), logs of the runs (logs/)
paper/                 main.tex, supplement.tex, sections/, tables/numbers.tex (generated), figures/ (generated),
                       main_pem.tex, supplement_pem.tex, sections_pem/, highlights_pem.txt (engineering version
                       of the same study, prepared for Probabilistic Engineering Mechanics),
                       refs.bib, highlights.txt
```

| paper item | script(s) in `code/experiments` |
|---|---|
| Table 1, Fig. 1, Fig. S1 | `e1_closures.py`, `e1b_assess.py`, `e1_seeds.py` |
| Table 2, Fig. 2, Fig. S2 | `e3_pendulum.py`, `e3b_pendulum_selfconsistent.py`, `e3c_partial.py` |
| Table 3, Fig. S3 | `e4_toggle.py` |
| Table 4, Fig. S4 | `e5_ignition.py`, `e6_ignition_highd.py` |
| Table 5, Tables S2-S4 | `e9_checks.py` |
| Table 6, Fig. 3 | `e8_enthalpy.py` |
| Table 7, Fig. S5 | `e7_series_spring.py` |
| Fig. 4, Fig. S6 | `../bench_cost.py`, `../bench_implicit.py` |
| Table S5 (network size and training budget) | `e10_capacity.py`, `sessions/run_v5_capacity.sh` |
| Table S6 (multi-core wall times) | `sessions/run_v6_parallel.sh` |
| Tables S7-S8 (Fourier features, adaptive sampling) | `e11_enhanced.py`, `sessions/run_v7_enhanced.sh` |
| engineering version (`main_pem.tex`): Table 1 (problems), Fig. 4 (cost against accuracy), Table 8 (guidelines) | written by hand from the results above; Fig. 4: `figures.py tradeoff` |
| single-core wall times of Tables 2-4, 6, 7 | `sessions/run_v4.sh`, `t_retime_ode.py` |

## Requirements

Tested with Python 3.15, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11 on Linux (WSL2); `threadpoolctl` is optional (it
pins BLAS to one thread inside worker processes).  Earlier versions have not been tested.  `parallel.py` uses the fork
start method and therefore needs Linux or macOS.  The paper compiles with [tectonic](https://tectonic-typesetting.github.io)
or with a TeX distribution providing `elsarticle`.

```
python -m pip install -r requirements.txt
```

## Quick start (from the stored results, a few minutes)

```
cd code && python tests/test_core.py          # ~2 s; must report 18/18 passed
cd experiments
python tables.py                              # -> paper/tables/numbers.tex (all table bodies and quoted numbers)
python figures.py                             # -> paper/figures/
cd ../../paper
tectonic -X compile --keep-intermediates main.tex && tectonic -X compile supplement.tex     # keeps main.aux, read by the supplement
tectonic -X compile --keep-intermediates main_pem.tex && tectonic -X compile supplement_pem.tex   # engineering version
```

`tables.py` regenerates `paper/tables/numbers.tex` identically from the files in `results/`; a quantity whose result
file is missing appears as a bold `??` in the PDF.

## Full reproduction

Run from `code/experiments`; single-core times on an Intel i7-4770 (4 cores, 3.9 GB RAM).

```
python e1_closures.py [name] ; python e1b_assess.py ; python e1_seeds.py {1,2}     # 10-60 min per closure
python e3_pendulum.py && python e3b_pendulum_selfconsistent.py && python e3c_partial.py
python e4_toggle.py ; python e9_checks.py metrics_ode
python e5_ignition.py
python e6_ignition_highd.py mc 4 ; python e6_ignition_highd.py train
python e7_series_spring.py mc 2 ; python e7_series_spring.py train {0,1,2}
python e8_enthalpy.py mc 4 ; python e8_enthalpy.py pilot ; python e8_enthalpy.py train {0,1,2}
python e9_checks.py {ghconv,pconv_ign,tol,spectra,ood_explicit} ; python e9_checks.py pconv_enth {2,4}
python e9_checks.py ood_enth {12,18}
python ../bench_cost.py ; python ../bench_implicit.py
sessions/run_v4.sh            # every single-core wall time: one sequential session (~1.5 h)
sessions/run_v5_capacity.sh   # network size / training budget: 11 trainings, up to 6 at a time (~10 h)
sessions/run_v6_parallel.sh   # after v5: 4/8-worker and 4/8-thread timings, larger networks (~2.5 h)
sessions/run_v7_enhanced.sh   # Fourier features / adaptive sampling: sigma sweep, 12 trainings, propagations (~5 h)
python e7_series_spring.py run {pinn,pinn_newton1}_seed{1,2} ; python e8_enthalpy.py run {pinn,pinn_nk2}_seed{1,2}
python e6_ignition_highd.py collect ; python e7_series_spring.py collect ; python e8_enthalpy.py collect
python tables.py ; python figures.py
```

The trainings and Monte Carlo runs keep several arrays of the size of the Monte Carlo ensemble in memory; on a
machine with 4 GB of RAM do not run more than one PDE experiment at a time.

**Timing protocol.**  Every wall time in a table comes from one session of runs executed one after another on an
otherwise idle machine (concurrent runs on the machine used were up to 13 times slower).  Single-core runs use one BLAS
thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`); multi-core runs set `PCE_WORKERS=W` (PDE problems, nodes split
over W processes) or `OPENBLAS_NUM_THREADS=W` (ODE problems).  ODE timings are medians of three runs; per-call
benchmarks are medians of 10-20 calls after a warm-up call.  Pilot runs and training are timed separately.



