#!/bin/bash
cd "$(dirname "$0")/.."   # run from code/experiments
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
until [ -f ../../results/e7_mc_d6p3.npz ]; do sleep 5; done
for m in newton psp pinn pinn_newton1 linear; do python3 -u e7_series_spring.py run $m > ../../results/e7_run_$m.log 2>&1 & done
wait
python3 -u e7_series_spring.py collect > ../../results/e7_collect.log 2>&1
