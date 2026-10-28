#!/bin/bash
# Revision runs: sequential re-timing, sparse-grid (Smolyak) PSP baselines, cost benchmarks, and
# additional training seeds for the residual-trained implicit closures.
cd "$(dirname "$0")/.."   # run from code/experiments
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=../../results
P=$R/v2_phases.log
echo "A: sequential timing start $(date)" > $P
python3 -u t_retime_ode.py > $R/v2_retime_ode.log 2>&1
python3 -u e5_ignition.py > $R/v2_e5.log 2>&1
for m in psp_s3 psp_s4 pinn taylor4; do python3 -u e6_ignition_highd.py run $m > $R/v2_e6_run_$m.log 2>&1; done
for m in psp_s3 psp_s4; do python3 -u e7_series_spring.py run $m > $R/v2_e7_run_$m.log 2>&1; done
for m in psp_s3 psp_s4 pinn_nk2; do python3 -u e8_enthalpy.py run $m > $R/v2_e8_run_$m.log 2>&1; done
python3 -u -W ignore ../bench_cost.py > $R/v2_bench_cost.log 2>&1
python3 -u -W ignore ../bench_implicit.py > $R/v2_bench_implicit.log 2>&1
echo "B: seed training start $(date)" >> $P
python3 -u e8_enthalpy.py train 1 > $R/v2_e8_train_seed1.log 2>&1 &
python3 -u e8_enthalpy.py train 2 > $R/v2_e8_train_seed2.log 2>&1 &
wait
python3 -u e7_series_spring.py train 1 > $R/v2_e7_train_seed1.log 2>&1 &
python3 -u e7_series_spring.py train 2 > $R/v2_e7_train_seed2.log 2>&1 &
wait
echo "C: seed propagation start $(date)" >> $P
for k in 1 2; do
  for m in pinn pinn_nk2; do python3 -u e8_enthalpy.py run ${m}_seed$k > $R/v2_e8_run_${m}_seed$k.log 2>&1; done
  for m in pinn pinn_newton1; do python3 -u e7_series_spring.py run ${m}_seed$k > $R/v2_e7_run_${m}_seed$k.log 2>&1; done
done
echo "D: collect start $(date)" >> $P
python3 -u e6_ignition_highd.py collect > $R/v2_e6_collect.log 2>&1
python3 -u e7_series_spring.py collect > $R/v2_e7_collect.log 2>&1
python3 -u e8_enthalpy.py collect > $R/v2_e8_collect.log 2>&1
echo "V2_DONE $(date)" >> $P
