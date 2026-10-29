#!/bin/bash
# (1) Multi-core timing on an otherwise idle machine, after the capacity study (run_v5_capacity.sh) has finished:
#     PDE problems with the nodes distributed over 4 and 8 worker processes (parallel.NodeParallel), ODE problems with
#     4- and 8-thread OpenBLAS.  Single-core results are kept; the parallel runs are stored as *_par_*.
# (2) Propagations with the networks of the capacity study (single core, timed like Tables 6-7).
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/my_env/bin/python}
R=../../results
P=$R/v6_phases.log
until grep -q CAPACITY_DONE $R/v5_phases.log 2>/dev/null; do sleep 30; done
echo "parallel timing start $(date)" > $P
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for W in 4 8; do
  for m in chord_w8 chord_w10 nk_w8 newton_w8 newton nk psp psp_s3 psp_s4 pinn pinn_nk1 pinn_nk2 pinn_nkc8 linear; do
    PCE_WORKERS=$W $PY -u e8_enthalpy.py run $m > $R/v6_e8_par_${m}_$W.log 2>&1
  done
  for m in eig psp psp_s3 psp_s4 taylor4 lanczos pinn; do
    PCE_WORKERS=$W $PY -u e6_ignition_highd.py run $m > $R/v6_e6_par_${m}_$W.log 2>&1
  done
  echo "workers $W done $(date)" >> $P
done
for NT in 4 8; do
  for m in newton nk newton_w8 nk_w8 chord_w8 psp psp_s3 psp_s4 pinn pinn_newton1 pinn_nk1 linear; do
    OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT $PY -u e7_series_spring.py run $m > $R/v6_e7_par_${m}_$NT.log 2>&1
  done
  OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT $PY -u t_retime_ode.py > $R/v6_retime_ode_$NT.log 2>&1
  echo "blas $NT done $(date)" >> $P
done
echo "capacity propagations start $(date)" >> $P
for tag in w512d4b4 w128d3b4 w256d3b1 w512d3b1; do
  for m in pinn pinn_nk1 pinn_nk2; do
    $PY -u e8_enthalpy.py run "$m@$tag" > $R/v6_e8_run_${m}_$tag.log 2>&1
  done
done
for m in pinn pinn_newton1; do $PY -u e7_series_spring.py run "$m@w256d4b4" > $R/v6_e7_run_${m}_cap.log 2>&1; done
for tag in sincos_d2p6_w256d4b4 sincos_d2p6_w256d4b4_R10; do
  $PY -u e10_capacity.py pendulum $tag > $R/v6_pendulum_$tag.log 2>&1
done
$PY -u e6_ignition_highd.py collect > $R/v6_e6_collect.log 2>&1
$PY -u e7_series_spring.py collect > $R/v6_e7_collect.log 2>&1
$PY -u e8_enthalpy.py collect > $R/v6_e8_collect.log 2>&1
echo "V6_DONE $(date)" >> $P
