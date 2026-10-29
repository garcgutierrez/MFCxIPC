#!/bin/bash
cd /mnt/d/chaos/pinn_ipc/code/experiments
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=../../results
busy() { ps -eo args | grep -v grep | grep -q "python3 -u e6_ignition_highd.py $1"; }
while busy train; do sleep 10; done
python3 -u e6_ignition_highd.py run pinn > $R/e6_run_pinn.log 2>&1
while busy "run eig" || busy "run psp"; do sleep 10; done
python3 -u e6_ignition_highd.py collect > $R/e6_collect.log 2>&1
python3 -u -W ignore ../bench_cost.py > $R/bench_cost.log 2>&1
python3 -u -W ignore ../bench_implicit.py > $R/bench_implicit.log 2>&1
python3 figures.py > $R/figures.log 2>&1
echo FINAL_DONE >> $R/figures.log
