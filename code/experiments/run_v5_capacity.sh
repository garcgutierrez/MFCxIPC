#!/bin/bash
# Capacity / training-budget study (accuracy only): 11 trainings, at most 6 at a time on the 4-core / 8-thread CPU.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${PY:-$HOME/miniconda3/envs/my_env/bin/python}
R=../../results
echo "capacity start $(date)" > $R/v5_phases.log
cat > /tmp/v5_jobs.txt <<JOBS
explicit sincos_d2p6 256 4 4
explicit sincos_d2p6 256 4 4 10 4
enthalpy 512 4 4
enthalpy 128 3 4
osc 256 4 4
explicit exp_d2p4 256 4 4
explicit exp_d2p4 256 3 1
explicit exp_d2p4 64 3 4
explicit sincos_d2p6 256 3 1
enthalpy 256 3 1
enthalpy 512 3 1
JOBS
xargs -P 6 -L 1 -I {} bash -c 'a="{}"; n=$(echo $a | tr " " "_"); '"$PY"' -u e10_capacity.py $a > '"$R"'/v5_$n.log 2>&1' < /tmp/v5_jobs.txt
echo "CAPACITY_DONE $(date)" >> $R/v5_phases.log
