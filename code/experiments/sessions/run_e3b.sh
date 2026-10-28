#!/bin/bash
cd "$(dirname "$0")/.."   # run from code/experiments
until ! pgrep -f e3_pendulum.py >/dev/null; do sleep 5; done
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u e3b_pendulum_selfconsistent.py > ../../results/e3b.log 2>&1
echo "E3b done rc=$?" >> ../../results/e3b.log
