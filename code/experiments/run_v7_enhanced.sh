#!/bin/bash
# Fourier features and residual-based adaptive sampling (RAD) for the neural closures (e11_enhanced.py), at the
# baseline network sizes and training budgets.
# (1) sigma sweep with short runs (1/4 of the iterations), 3 at a time; (2) full-budget trainings of the variants
# A, F<sigma*>, F<sigma*>A, 3 at a time; (3) propagations with the new networks, sequential on an idle machine.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${PY:-python}
R=../../results
L=$R/v7_phases.log
echo "enhanced start $(date)" > $L
run_jobs() {   # stdin: one job per line ("<subcommand> <args>"), at most 3 at a time
  xargs -P 3 -L 1 -I {} bash -c 'a="{}"; n=$(echo $a | tr " " "_"); '"$PY"' -u e11_enhanced.py $a > '"$R"'/v7_$n.log 2>&1'
}
J=$(mktemp)
for s in P F0.25 F0.5 F1 F2 F4; do echo "explicit sincos_d2p6 $s 0.25"; done > $J
for s in P F0.25 F0.5 F1 F2 F4; do echo "osc $s 0.25"; echo "enthalpy $s 0.25"; echo "explicit exp_d2p4 $s 0.25"; done >> $J
run_jobs < $J
echo "sweep done $(date)" >> $L
SE=$($PY e11_enhanced.py best_sigma exp_d2p4); SS=$($PY e11_enhanced.py best_sigma sincos_d2p6)
SH=$($PY e11_enhanced.py best_sigma enthalpy); SO=$($PY e11_enhanced.py best_sigma osc)
echo "best sigma: exp $SE sincos $SS enthalpy $SH osc $SO" >> $L
cat > $J <<JOBS
explicit sincos_d2p6 A
explicit sincos_d2p6 F${SS}
explicit sincos_d2p6 F${SS}A
osc A
osc F${SO}
osc F${SO}A
enthalpy A
enthalpy F${SH}
enthalpy F${SH}A
explicit exp_d2p4 A
explicit exp_d2p4 F${SE}
explicit exp_d2p4 F${SE}A
JOBS
run_jobs < $J
rm -f $J
echo "full trainings done $(date)" >> $L
for v in A F${SS} F${SS}A; do $PY -u e10_capacity.py pendulum sincos_d2p6_$v > $R/v7_pendulum_$v.log 2>&1; done
for v in A F${SH} F${SH}A; do
  for m in pinn pinn_nk1 pinn_nk2; do $PY -u e8_enthalpy.py run "$m@$v" > $R/v7_e8_run_${m}_$v.log 2>&1; done
done
for v in A F${SO} F${SO}A; do
  for m in pinn pinn_newton1; do $PY -u e7_series_spring.py run "$m@$v" > $R/v7_e7_run_${m}_$v.log 2>&1; done
done
$PY -u e7_series_spring.py collect > $R/v7_e7_collect.log 2>&1
$PY -u e8_enthalpy.py collect > $R/v7_e8_collect.log 2>&1
echo "V7_DONE $(date)" >> $L
