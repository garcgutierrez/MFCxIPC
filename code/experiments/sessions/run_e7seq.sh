#!/bin/bash
cd "$(dirname "$0")/.."   # run from code/experiments
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=../../results
until grep -q E8_DONE $R/e8_phases.log 2>/dev/null; do sleep 20; done
for m in newton nk psp pinn pinn_newton1 pinn_nk1 linear; do
  python3 -u e7_series_spring.py run $m > $R/e7_seq_$m.log 2>&1
done
python3 -u e7_series_spring.py collect > $R/e7_collect_seq.log 2>&1
python3 figures.py e7 >> $R/figures.log 2>&1
echo E7SEQ_DONE >> $R/e8_phases.log
