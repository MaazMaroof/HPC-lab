#!/bin/bash
#SBATCH --job-name=matrix_openmp
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

set -euo pipefail

module purge
module load compiler/gcc/12.3

gcc -O2 -fopenmp project.c -o project

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Change matrix size if needed; 600 is a reasonable starting benchmark.
./project 600 "$OMP_NUM_THREADS"
