#!/bin/bash
# BSUB -J MyProgram
# BSUB -o myprogram_%J.out
# BSUB -e myprogram_%J.err
# BSUB -W 00:10
# BSUB -n 16
# BSUB -R "span[hosts=1]"
# BSUB -R "rusage[mem=32768]"
# BSUB -P mon_projet_hpc
# BSUB -q mc_dev_normal 
export RAYON_NUM_THREADS=$LSB_CPU_NTASKS
cargo run --release --example remeshing_res_2d -- --num-splits "10" --metric-type "aniso" --cost-estimator "Nocost" --partitionner "HilbertPartitionner"