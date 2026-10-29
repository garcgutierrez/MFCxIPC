#!/bin/bash
# launch each experiment as soon as the closures it needs exist
R=../../results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
( until [ -f $R/e1_pow2.5_d4p3.json ] && [ -f $R/e1_pow-1_d4p3.json ]; do sleep 10; done
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u e4_toggle.py > $R/e4.log 2>&1; echo "E4 done rc=$?" ) &
( until [ -f $R/e1_exp_d5p3.json ]; do sleep 10; done
  python3 -u e5_ignition.py > $R/e5.log 2>&1; echo "E5 done rc=$?" ) &
( until [ -f $R/e1_sincos_d2p6.json ]; do sleep 10; done
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u e3_pendulum.py > $R/e3.log 2>&1; echo "E3 done rc=$?" ) &
wait
