# Matrix-function and neural closures of non-polynomial terms in intrusive polynomial chaos

Code, results and manuscript of the paper in `paper/main.tex` (Supplementary Material: `paper/supplement.tex`).
An engineering version of the same study, written for *Probabilistic Engineering Mechanics*, is in `paper/main_pem.tex`
(Supplementary Material: `paper/supplement_pem.tex`); both versions share `tables/numbers.tex`, `figures/` and `refs.bib`.
Everything is implemented in NumPy/SciPy, including the neural networks and their derivatives.

## Environment

Python 3.15, NumPy 2.5, SciPy 1.18, Matplotlib 3.11 (`code/requirements.txt`).  Intel i7-4770 (3.4 GHz, 4 cores /
8 hardware threads, 3.9 GB RAM, WSL2).  Single-core runs use one thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`);
the multi-core session is described under *Timing protocol*.
LaTeX: elsarticle class, `elsarticle-harv` style (engineering version: `elsarticle-num-names`), compiled with tectonic.

## Layout

```
code/
  pce.py            Hermite PC basis, triple-product tensor (dense below P = 250, sparse construction above),
                    Galerkin products, O(nnz) assembly of multiplication matrices, tensor and Smolyak sparse
                    Gauss-Hermite rules (pass q = "s<k>" to any PSP closure for the sparse rule of level k)
  closures.py       explicit nonlinearities (exp, sin/cos, powers): ExactGalerkin (f(M_u)e0 by eigh), PSP,
                    Taylor, Lanczos, PINNClosure (ray-ODE loss)
  implicit.py       oscillator constraint: NewtonClosure, NewtonKrylovClosure, CardanoPSP, PINNImplicit,
                    CubicSolver (Newton / NK / chord, cold or warm start, matched tolerances; revision baselines)
  implicit_field.py heat-conduction constitutive law at every node: FieldNewton, FieldNewtonKrylov, FieldPSP,
                    FieldPINN, FieldSolver (revision baselines)
  mlp.py            MLP with forward-mode tangents, hand-written backprop, Adam; FourierMLP (random Fourier
                    features), rad_indices (residual-based adaptive sampling)
  xtfc.py           solution-level spectral X-TFC (Levenberg-Marquardt, window marching)
  problems.py       Pendulum, ToggleSwitch, Ignition, SeriesSpringOscillator, EnthalpyHeat
  propagate.py      vectorised Monte Carlo, intrusive integration, metrics (L2, paired moment errors and MC
                    standard errors, Bhattacharyya)
  parallel.py       NodeParallel: node-batched closures on persistent forked worker processes, each owning a fixed
                    block of nodes (per-node warm starts / frozen Jacobians preserved); results identical to serial
  bench_cost.py, bench_implicit.py   per-call cost benchmarks
  gpu_backend.py    NumPy/CuPy versions of the heat-conduction and ignition closures and batched solvers (vectorised GMRES,
                    active-set chord/NK) for the node-scaling and GPU benchmark (E12); CuPy is needed only for the GPU devices
  tests/test_gpu.py    gpu_backend.py against the library on the CPU and the GPU, N = 60, three call sequences (21 tests)
  tests/test_core.py   18 tests: identities of Section 2, every hand-written derivative, serial = parallel,
                       Fourier features and adaptive sampling
  tools/texcount.py    word count used to compare manuscript versions
  tools/bench_sprod.py cost of a batched Galerkin product against batch size (why node parallelism gains little)
  experiments/
    e1_closures.py, e1b_assess.py, e1_seeds.py   generic neural closures (Table 1, Fig. 1), seed study
    e3_pendulum.py, e3b_..., e3c_...             pendulum (Table 2, Fig. 2)
    e4_toggle.py                                 toggle switch (Table 3)
    e5_ignition.py, e6_ignition_highd.py         ignition d = 2, 5, 8 (Table 4)
    e7_series_spring.py                          oscillator (Table 7)
    e8_enthalpy.py                               heat conduction (Table 6, Fig. 3)
    e9_checks.py                                 error attribution, closure definition, OOD and tolerance checks
                                                 (Table 5, Supplementary Section S5)
    t_retime_ode.py                              sequential re-timing of Tables 2-3
    e12_scaling.py                               node scaling (15-16000 nodes) and GPU benchmark: subcommands env, inputs,
                                                 run, batch, endtoend, collect -> results/e12_scaling.{json,md,pdf},
                                                 e12_notes.md, e12_paragraph.tex (draft, not used), run_e12.sh
    e10_capacity.py                              network size / training budget study, propagations with the
                                                 larger networks (Supplementary Table S5)
    e11_enhanced.py                              Fourier features and residual-based adaptive sampling at the
                                                 baseline sizes/budgets (Supplementary Tables S7-S8)
    run_*.sh                                     orchestration of the sessions actually run (see History)
    tables.py                                    every table body and every number quoted in the paper
                                                 -> paper/tables/numbers.tex (LaTeX macros)
    figures.py                                   all figures -> paper/figures/
results/            logs, pickles, trained models (results/models/*.npz); results/backup_v1: before the
                    first revision
paper/              main.tex, supplement.tex, sections/*.tex, tables/ (generated), figures/, refs.bib,
                    main_pem.tex, supplement_pem.tex, sections_pem/*.tex, highlights_pem.txt (engineering version),
                    highlights.txt, refs_notes.md and refs_krylov_notes.md (how the references were checked)
versions/           frozen copies of earlier manuscript and code states (2026-10-23_pre-audit)
```

## Reproducing

```
cd code && python tests/test_core.py                     # ~2 s; must report 18/18 passed
cd experiments
python e1_closures.py [name] ; python e1b_assess.py ; python e1_seeds.py {1,2}     # 10-60 min per closure
python e3_pendulum.py && python e3b_pendulum_selfconsistent.py && python e3c_partial.py
python e4_toggle.py ; python e9_checks.py metrics_ode    # paired metrics of the ODE problems
python e6_ignition_highd.py mc 4 ; python e6_ignition_highd.py train
python e7_series_spring.py mc 2 ; python e7_series_spring.py train {0,1,2}
python e8_enthalpy.py mc 4 ; python e8_enthalpy.py pilot ; python e8_enthalpy.py train {0,1,2}
python e9_checks.py {ghconv,pconv_ign,tol,spectra,ood_explicit} ; python e9_checks.py pconv_enth {2,4}
python e9_checks.py ood_enth {12,18}
./run_v4.sh                                              # all wall times: one sequential session, idle machine
./run_v5_capacity.sh                                     # capacity study: 11 trainings, up to 6 concurrently
./run_v6_parallel.sh                                     # after v5: 4/8-worker and 4/8-thread timings, then
                                                         #   propagations with the capacity-study networks
./run_v7_enhanced.sh                                     # Fourier features / adaptive sampling: sigma sweep,
                                                         #   12 trainings (3 at a time), propagations
python e7_series_spring.py run {pinn,pinn_newton1}_seed{1,2} ; python e8_enthalpy.py run {pinn,pinn_nk2}_seed{1,2}
python e6_ignition_highd.py collect ; python e7_series_spring.py collect ; python e8_enthalpy.py collect
python figures.py ; python tables.py
cd ../../paper && tectonic -X compile --keep-intermediates main.tex && tectonic -X compile supplement.tex   # keeps main.aux, read by the supplement
tectonic -X compile --keep-intermediates main_pem.tex && tectonic -X compile supplement_pem.tex             # engineering version
```

Timing protocol: every wall time in a table comes from one session of runs executed one after another on an
otherwise idle machine (`run_v4.sh`; concurrent runs on this machine were up to 13x slower).  ODE tables are medians of
three runs; per-call benchmarks are medians over 10-20 calls after a warm-up call.  Pilot runs and training are timed
separately (stored with the trained models).  Multi-core times (`run_v6_parallel.sh`): for the PDE problems,
`PCE_WORKERS=W` distributes the nodes over W persistent worker processes (`parallel.NodeParallel`, single-threaded BLAS
in each worker); for the ODE problems, `OPENBLAS_NUM_THREADS=W`.  Results are stored next to the single-core ones
(`e6_par_*`, `e7_par_*`, `e8_par_*`, `wall_par<W>` keys) and never overwrite them.  Training times of the capacity study
were measured with six trainings sharing the CPU (three for `run_v7_enhanced.sh`) and are only indicative.

## History

- 2026-10-13/15: first version (`run_*.sh` up to `run_v3.sh`); sparse-grid PSP baselines and seed study added on 10-15.
- 2026-10-23 (review audit; pre-audit copy in `versions/2026-10-23_pre-audit`): corrected formulation (exact
  projection vs Galerkin arithmetic vs matrix-function closure; Galerkin roots are not instances), O(nnz) assembly of
  multiplication matrices (the dense contraction penalised eigendecomposition and dense Newton), strong classical
  baselines for implicit closures (warm starts, matched tolerances, chord Newton), paired moment errors (the earlier
  moment errors contained MC sampling noise of the same size), error-attribution checks (PC order, quadrature,
  tolerances), out-of-distribution and seed checks, test suite, generated tables and numbers, Supplementary Material.
- 2026-10-24: multi-core timings (node-parallel worker processes for the PDEs, multithreaded BLAS for the ODEs) and a
  network-size / training-budget study (`run_v5_capacity.sh`, `run_v6_parallel.sh`, `e10_capacity.py`).
- 2026-10-26: random Fourier features and residual-based adaptive sampling (RAD) for the three network types
  (`e11_enhanced.py`, `run_v7_enhanced.sh`).
- 2026-10-28: engineering version for Probabilistic Engineering Mechanics (`paper/main_pem.tex`): same results, engineering
  framing, theory moved to the Supplementary Material, toggle switch moved to the Supplementary Material, practical
  guidelines table, new cost-accuracy figure (`figures.py tradeoff`).
- 2026-10-29: node-scaling and GPU benchmark (`e12_scaling.py`, `run_e12.sh`, task in `matrix-function-closures-ipc/GPU_SCALING_TASK.md`):
  i9-13900H and RTX 4060 Laptop GPU, 15 to 16000 nodes, one thread / 20 processes / GPU; the network hybrid never beats warm chord
  Newton (2.2-7.2x slower), the GPU speeds all closures up alike; reviewed in `GPU_REVIEW_NOTES.md`; paper: Supplementary S12/S13.

## Not yet done (author input needed)

Repository URL/DOI, licence, CRediT statement and confirmation of the author list (see `paper/main.tex`).
