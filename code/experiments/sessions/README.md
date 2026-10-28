# Session scripts

Orchestration of the computing sessions that produced the stored results, kept as a record and listed in the order
they were run.  Each script waits for a marker written by the previous session (`until grep -q ... ; do sleep ...`),
starts the runs and writes its own marker; paths have been made relative (they `cd` to `code/experiments`) but the
scripts are otherwise unchanged.  To reproduce the results, follow the commands in the top-level README instead.

| script | date | content |
|---|---|---|
| `run_after.sh` | 2026-10-13 | first runs: each experiment as soon as the closures it needs exist |
| `run_e3b.sh` | 2026-10-14 | pendulum: anchored, self-consistent neural closure |
| `run_e7.sh`, `run_e7seq.sh` | 2026-10-14 | oscillator with an implicit spring force |
| `run_final.sh`, `run_timing.sh`, `run_timing2.sh` | 2026-10-14 | ignition d = 8; first sequential re-timing |
| `run_e8.sh`, `run_e8b.sh` | 2026-10-14 | heat conduction with an implicit constitutive law |
| `run_conv.sh` | 2026-10-14 | convergence runs |
| `run_v2.sh`, `run_v3.sh` | 2026-10-15 | sparse-grid PSP baselines, cost benchmarks, re-timing |
| `run_v4.sh` | 2026-10-23 | strong classical baselines, checks; **all single-core wall times in the paper** |
| `run_v5_capacity.sh` | 2026-10-24 | network size / training budget study (six trainings at a time; accuracy only) |
| `run_v6_parallel.sh` | 2026-10-24/25 | **multi-core wall times** (4/8 workers or BLAS threads), propagations with the larger networks |
| `run_v7_enhanced.sh` | 2026-10-26 | Fourier features and adaptive sampling: sigma sweep and trainings (three at a time; accuracy only), then sequential propagations |

Every timing session ran its jobs one after another on an otherwise idle machine; concurrent runs on the 4-core machine
used for the paper were up to 13 times slower.
