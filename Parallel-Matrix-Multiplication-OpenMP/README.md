# Parallel Matrix Multiplication Using OpenMP

## Overview
This project compares conventional sequential matrix multiplication with an OpenMP-parallel implementation in C. It verifies that both versions produce the same result and reports elapsed time and speedup.

## Objectives
- Implement standard matrix multiplication using C.
- Parallelize independent output-row calculations with OpenMP.
- Compare sequential and parallel execution time.
- Verify correctness by comparing both result matrices.
- Run the program as a SLURM batch job on an HPC cluster.

## Files
- `project.c` — sequential and OpenMP implementations, timing, correctness check, and speedup.
- `project.sh` — SLURM batch script based on the instructor's template.
- `results/performance_results.csv` — template for recording measurements from real runs.

## Requirements
- GCC with OpenMP support.
- `omp.h` and the OpenMP runtime (normally included with an OpenMP-enabled GCC installation).
- For cluster execution: SLURM and the compiler module configured by your HPC administrator.

## Compile locally
```bash
gcc -O2 -fopenmp project.c -o project
```

## Run locally
The program accepts optional matrix size and thread count. Defaults are `400` and `4`.
```bash
./project
./project 600 4
./project 100 2
```

For a small matrix, the program prints the result matrix. For larger matrices, it prints a note instead of dumping the entire result.

## Run on the HPC cluster
1. Copy `project.c` and `project.sh` to the same directory on the cluster.
2. Check that the partition and compiler module in `project.sh` match your cluster's configuration.
3. Submit:
   ```bash
   sbatch project.sh
   ```
4. Check the job queue (if permitted):
   ```bash
   squeue -u "$USER"
   ```
5. Read the output and error logs named `output_JOBID.log` and `error_JOBID.log`.

The script uses the module name from the instructor-provided template (`compiler/gcc/12.3`). If that module is unavailable, ask the cluster administrator or instructor for the correct module name. Do not assume that a module name or partition is valid without checking.

## Algorithm
For each output element `C[i][j]`, calculate:
`C[i][j] = sum(A[i][k] * B[k][j])` for `k = 0 ... N-1`.

The sequential version processes output rows one after another. The OpenMP version uses `#pragma omp parallel for schedule(static)` to distribute independent output rows across threads. Each thread writes to different output elements, so no synchronization is needed for those writes.

## Complexity
For two square `N x N` matrices, the standard algorithm performs `O(N^3)` arithmetic work. The matrices require `O(N^2)` storage each.

## Performance measurements
The program prints sequential time, OpenMP time, and speedup:
`Speedup = sequential time / OpenMP time`.

Performance depends on matrix size, processor, compiler, thread count, and system load. Parallel execution is not guaranteed to be faster for small matrices. Run multiple sizes and thread counts, and record actual measurements. Do not use illustrative numbers as experimental results.

## Results log
Record measurements from your own runs in `results/performance_results.csv`. Include the execution environment (local machine or cluster) and matrix/thread settings. Do not claim cluster measurements unless the program was actually run on the cluster.

## Limitations and future improvements
- This is a straightforward baseline implementation, not a cache-blocked or BLAS-optimized matrix multiplication.
- Benchmarking is sensitive to system load and memory hierarchy.
- Future work could include repeated trials, average timings, cache blocking, and plots of speedup versus thread count.

## Author
- Student: Maaz Maroof
- Project: Parallel Matrix Multiplication Using OpenMP
- Course/department details: Add your official course, department, and university details if required.
