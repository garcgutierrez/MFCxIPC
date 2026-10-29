#!/bin/bash
# sequential re-run of the d=8 propagations on an otherwise idle machine (fair wall-clock times)
cd /mnt/d/chaos/pinn_ipc/code/experiments
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=../../results
until grep -q FINAL_DONE $R/figures.log 2>/dev/null; do sleep 10; done
for m in pinn taylor4 lanczos psp eig; do
  python3 -u e6_ignition_highd.py run $m > $R/e6_seq_$m.log 2>&1
done
python3 -u e6_ignition_highd.py collect > $R/e6_collect_seq.log 2>&1
python3 figures.py e5 >> $R/figures.log 2>&1
echo TIMING_DONE >> $R/figures.log
