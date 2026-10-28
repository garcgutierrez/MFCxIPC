#!/bin/bash
cd "$(dirname "$0")/.."   # run from code/experiments
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=../../results
until grep -q TIMING2_DONE $R/figures.log 2>/dev/null; do sleep 15; done
echo "phase A start $(date)" > $R/e8_phases.log
python3 -u e8_enthalpy.py mc 4 > $R/e8_mc.log 2>&1 &
python3 -u e8_enthalpy.py pilot > $R/e8_pilot.log 2>&1
python3 -u e8_enthalpy.py train > $R/e8_train.log 2>&1 &
TRAIN=$!
wait %1 2>/dev/null   # MC
for m in newton nk psp linear; do python3 -u e8_enthalpy.py run $m > $R/e8_run_$m.log 2>&1 & done
wait $TRAIN
echo "phase B start $(date)" >> $R/e8_phases.log
for m in pinn pinn_nk1 pinn_newton1; do python3 -u e8_enthalpy.py run $m > $R/e8_run_$m.log 2>&1 & done
wait
python3 -u e8_enthalpy.py collect > $R/e8_collect.log 2>&1
echo "phase C (sequential timing) start $(date)" >> $R/e8_phases.log
for m in pinn pinn_nk1 linear pinn_newton1 psp nk newton; do
  python3 -u e8_enthalpy.py run $m > $R/e8_seq_$m.log 2>&1
done
python3 -u e8_enthalpy.py collect > $R/e8_collect_seq.log 2>&1
echo "E8_DONE $(date)" >> $R/e8_phases.log
