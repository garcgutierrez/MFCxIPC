#!/bin/bash
# Review revision (2026-10-23): one sequential timing session on an otherwise idle machine, after all
# accuracy-only checks have finished.  Re-times every propagation whose code path changed (O(nnz) assembly of
# multiplication matrices) and adds the strong classical baselines (warm starts, matched tolerances, chord Newton).
cd "$(dirname "$0")/.."   # run from code/experiments
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${PY:-python}
R=../../results
P=$R/v4_phases.log
busy() { ps -eo args | grep -v grep | grep -q -E "e9_checks.py|e1_seeds.py"; }
while busy; do sleep 20; done
echo "timing session start $(date)" > $P
for m in chord_w8 chord_w10 chord_w6 nk_w8 nk_w10 newton_w8 newton_c8 nk_c8 newton nk psp psp_s3 psp_s4 psp_s5 \
         pinn pinn_nk1 pinn_nk2 pinn_nkconv pinn_nkc8 linear; do
  $PY -u e8_enthalpy.py run $m > $R/v4_e8_run_$m.log 2>&1
done
echo "e8 done $(date)" >> $P
for m in newton nk psp psp5 psp_s3 psp_s4 psp_s5 pinn pinn_newton1 pinn_nk1 pinn_newtonconv pinn_nkconv linear \
         newton_c8 nk_c8 newton_w8 nk_w8 chord_w8 chord_w10 pinn_nkc8; do
  $PY -u e7_series_spring.py run $m > $R/v4_e7_run_$m.log 2>&1
done
echo "e7 done $(date)" >> $P
$PY -u e6_ignition_highd.py run eig > $R/v4_e6_run_eig.log 2>&1
$PY -u e5_ignition.py > $R/v4_e5.log 2>&1
$PY -u t_retime_ode.py > $R/v4_retime_ode.log 2>&1
$PY -u -W ignore ../bench_cost.py > $R/v4_bench_cost.log 2>&1
$PY -u -W ignore ../bench_implicit.py > $R/v4_bench_implicit.log 2>&1
echo "timing done $(date)" >> $P
# diagnostics (not timed): achieved residuals of the fixed-correction hybrids
$PY -u e8_enthalpy.py run pinn_nk2rec > $R/v4_e8_run_pinn_nk2rec.log 2>&1
$PY -u e7_series_spring.py run pinn_newton1rec > $R/v4_e7_run_pinn_newton1rec.log 2>&1
$PY -u e6_ignition_highd.py collect > $R/v4_e6_collect.log 2>&1
$PY -u e7_series_spring.py collect > $R/v4_e7_collect.log 2>&1
$PY -u e8_enthalpy.py collect > $R/v4_e8_collect.log 2>&1
echo "V4_DONE $(date)" >> $P
