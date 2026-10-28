#!/bin/bash
cd "$(dirname "$0")/.."   # run from code/experiments
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=../../results
for m in pinn_newtonconv pinn_nkconv; do python3 -u e7_series_spring.py run $m > $R/e7_seq_$m.log 2>&1; done
python3 -u e7_series_spring.py collect > $R/e7_collect_seq.log 2>&1
for m in pinn_nkconv pinn_newtonconv pinn_nk2; do python3 -u e8_enthalpy.py run $m > $R/e8_seq_$m.log 2>&1; done
python3 -u e8_enthalpy.py collect > $R/e8_collect_seq.log 2>&1
python3 figures.py e7 e8 >> $R/figures.log 2>&1
echo CONV_DONE >> $R/e8_phases.log
