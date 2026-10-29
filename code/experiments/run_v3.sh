#!/bin/bash
# Revision, phase E: re-time every remaining propagation of the d >= 6 problems in the same sequential
# session as the sparse-grid runs of run_v2.sh (idle machine), so that each table uses one session.
cd /mnt/d/chaos/pinn_ipc/code/experiments
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=../../results
P=$R/v2_phases.log
until grep -q V2_DONE $P 2>/dev/null; do sleep 20; done
echo "E: same-session re-timing start $(date)" >> $P
for m in lanczos psp eig; do python3 -u e6_ignition_highd.py run $m > $R/v3_e6_run_$m.log 2>&1; done
for m in newton nk psp psp_s3 psp_s4 pinn pinn_newton1 pinn_nk1 pinn_newtonconv pinn_nkconv linear; do
  python3 -u e7_series_spring.py run $m > $R/v3_e7_run_$m.log 2>&1
done
for m in pinn linear pinn_nk1 pinn_nkconv psp pinn_newton1 pinn_newtonconv nk newton; do
  python3 -u e8_enthalpy.py run $m > $R/v3_e8_run_$m.log 2>&1
done
echo "F: collect start $(date)" >> $P
python3 -u e6_ignition_highd.py collect > $R/v3_e6_collect.log 2>&1
python3 -u e7_series_spring.py collect > $R/v3_e7_collect.log 2>&1
python3 -u e8_enthalpy.py collect > $R/v3_e8_collect.log 2>&1
python3 figures.py > $R/v3_figures.log 2>&1
echo "V3_DONE $(date)" >> $P
