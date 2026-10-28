#!/bin/bash
cd "$(dirname "$0")/.."   # run from code/experiments
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=../../results
until grep -q TIMING_DONE $R/figures.log 2>/dev/null; do sleep 10; done
python3 -u e5_ignition.py > $R/e5_seq.log 2>&1
python3 figures.py e5 >> $R/figures.log 2>&1
echo TIMING2_DONE >> $R/figures.log
