#!/bin/bash
# E12: node scaling and GPU benchmark (GPU_SCALING_TASK.md).  One configuration at a time, idle machine.
#   ./run_e12.sh            full sweep (several hours);   ./run_e12.sh quick   N = 15/31 and 60 only
# Call counts S' (of S = 2000): 2000 up to N = 60, then 300, 60, 15, 10, so that the CPU runs of the strong methods stay
# near the 10-minute limit of the task; slower methods are stopped by a per-run time budget (CPU 90 s up to N = 60, 45 s
# above; GPU 300 s / 150 s) and reported as truncated.  window=first is the rule of the task (first S' calls), window=late the
# extra check (last S' calls after an untimed priming call).  The configurations are listed in order of priority and run
# by `e12_scaling.py batch` (one forked process per configuration).
cd "$(dirname "$0")"; [ -f e12_scaling.py ] || cd ..          # run from code/experiments (here or in sessions/)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
HEAT="chord_w8 chord_w10 chord_w8_as chord_w10_as chordfix1 chordfix2 psp_s3 psp_s4 pinn nk_w8 nk_w8_as newton_w8 pinn_nk1 pinn_nk2 psp"
IGN="pinn taylor4 psp_s3 psp_s4 lanczos psp eig"
HEATKEY="chord_w8 chord_w10 chord_w10_as chordfix2 nk_w8 nk_w8_as newton_w8 pinn pinn_nk1 pinn_nk2 psp_s3 psp_s4"
IGNKEY="pinn taylor4 psp_s3 psp_s4 lanczos eig"
HEATX="chord_w10 chord_w10_as nk_w8 psp_s4 pinn pinn_nk2"
IGNX="lanczos taylor4 psp_s4 pinn"
if [ "$1" = quick ]; then SH="15 60"; SI="31 60"; else SH="15 60 250 1000 4000 16000"; SI="31 60 250 1000 4000 16000"; fi

sp() { case $1 in 15|31|60) echo 2000;; 250) echo 300;; 1000) echo 60;; 4000) echo 15;; 16000) echo 10;; esac; }
budget() { case $1 in gpu*) [ $2 -le 60 ] && echo 300 || echo 150;; *) [ $2 -le 60 ] && echo 90 || echo 45;; esac; }
cfg() { # device window "heat methods" "ign methods" "min N for this phase"
  local dev=$1 win=$2 LH=$3 LI=$4 nmin=${5:-0}
  for N in $SH; do [ $N -lt $nmin ] && continue; [ $win = late ] && [ $N -le 60 ] && continue
    [ $dev = cpu1orig ] && [ $N -gt 250 ] && continue
    for m in $LH; do echo "heat $m $N $dev $win $(sp $N) $(budget $dev $N)"; done; done
  for N in $SI; do [ $N -lt $nmin ] && continue; [ $win = late ] && [ $N -le 60 ] && continue
    [ $dev = cpu1orig ] && [ $N -gt 250 ] && continue
    for m in $LI; do echo "ign $m $N $dev $win $(sp $N) $(budget $dev $N)"; done; done
}
list() {
  cfg gpu first "$HEAT" "$IGN"; cfg cpu1 first "$HEAT" "$IGN"
  cfg gpu late "$HEAT" "$IGN"; cfg cpu1 late "$HEAT" "$IGN"
  cfg cpuall first "$HEATKEY" "$IGNKEY"; cfg cpuall late "$HEATKEY" "$IGNKEY" 1000
  cfg gpux first "$HEATX" "$IGNX"; cfg gpux late "$HEATX" "$IGNX"
  cfg gpu32 first "pinn psp_s3 psp_s4" "pinn taylor4 psp_s3 psp_s4"
  cfg cpu1orig first "chord_w10 nk_w8 pinn_nk1 pinn_nk2" "lanczos eig"
}
python e12_scaling.py env > /dev/null
python e12_scaling.py inputs
list > ../../results/logs/e12_configs.txt
python e12_scaling.py batch < ../../results/logs/e12_configs.txt
python e12_scaling.py collect
