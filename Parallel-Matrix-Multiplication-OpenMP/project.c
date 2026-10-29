/*
 * Project: Parallel Matrix Multiplication Using OpenMP
 * Description:
 *   Compares sequential matrix multiplication with an OpenMP-parallel version.
 *   Verifies correctness and reports execution time and speedup.
 *
 * Build:
 *   gcc -O2 -fopenmp project.c -o project
 *
 * Run:
 *   ./project [matrix_size] [threads]
 * Example:
 *   ./project 600 4
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static double *allocate_matrix(size_t n) {
    if (n == 0 || n > ((size_t)-1) / n ||
        n * n > ((size_t)-1) / sizeof(double)) {
        return NULL;
    }
    return (double *)malloc(n * n * sizeof(double));
}

static void initialize_matrices(double *a, double *b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            /* Deterministic values keep runs reproducible. */
            a[i * n + j] = (double)((i + j) % 10 + 1);
            b[i * n + j] = (double)((i * 2 + j) % 10 + 1);
        }
    }
}

static void multiply_sequential(const double *a, const double *b,
                                double *c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

static void multiply_openmp(const double *a, const double *b,
                            double *c, size_t n, int threads) {
    omp_set_num_threads(threads);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += a[(size_t)i * n + k] * b[k * n + j];
            }
            c[(size_t)i * n + j] = sum;
        }
    }
}

static int compare_matrices(const double *x, const double *y, size_t n) {
    const double tolerance = 1e-9;
    for (size_t i = 0; i < n * n; ++i) {
        double difference = x[i] - y[i];
        if (difference < 0) difference = -difference;
        if (difference > tolerance) return 0;
    }
    return 1;
}

static void print_preview(const char *name, const double *matrix, size_t n) {
    size_t limit = n < 5 ? n : 5;
    printf("%s (top-left %zu x %zu preview):\n", name, limit, limit);
    for (size_t i = 0; i < limit; ++i) {
        for (size_t j = 0; j < limit; ++j) {
            printf("%10.2f ", matrix[i * n + j]);
        }
        putchar('\n');
    }
}

int main(int argc, char **argv) {
    size_t n = 400;
    int threads = 4;

    if (argc > 1) {
        char *end = NULL;
        unsigned long parsed = strtoul(argv[1], &end, 10);
        if (*argv[1] == '\0' || *end != '\0' || parsed == 0 ||
            parsed > 10000) {
            fprintf(stderr, "Invalid matrix size. Use an integer from 1 to 10000.\n");
            return EXIT_FAILURE;
        }
        n = (size_t)parsed;
    }
    if (argc > 2) {
        char *end = NULL;
        long parsed = strtol(argv[2], &end, 10);
        if (*argv[2] == '\0' || *end != '\0' || parsed < 1 || parsed > 1024) {
            fprintf(stderr, "Invalid thread count. Use an integer from 1 to 1024.\n");
            return EXIT_FAILURE;
        }
        threads = (int)parsed;
    }
    if (argc > 3) {
        fprintf(stderr, "Usage: %s [matrix_size] [threads]\n", argv[0]);
        return EXIT_FAILURE;
    }

    double *a = allocate_matrix(n);
    double *b = allocate_matrix(n);
    double *c_seq = allocate_matrix(n);
    double *c_par = allocate_matrix(n);
    if (!a || !b || !c_seq || !c_par) {
        fprintf(stderr, "Could not allocate matrices for size %zu. Try a smaller size.\n", n);
        free(a); free(b); free(c_seq); free(c_par);
        return EXIT_FAILURE;
    }

    initialize_matrices(a, b, n);
    printf("Parallel Matrix Multiplication Using OpenMP\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Requested OpenMP threads: %d\n\n", threads);

    double start = omp_get_wtime();
    multiply_sequential(a, b, c_seq, n);
    double sequential_time = omp_get_wtime() - start;

    start = omp_get_wtime();
    multiply_openmp(a, b, c_par, n, threads);
    double parallel_time = omp_get_wtime() - start;

    int correct = compare_matrices(c_seq, c_par, n);
    double speedup = parallel_time > 0.0 ? sequential_time / parallel_time : 0.0;

    printf("Sequential time : %.6f seconds\n", sequential_time);
    printf("OpenMP time     : %.6f seconds\n", parallel_time);
    printf("Speedup         : %.3fx\n", speedup);
    printf("Correctness     : %s\n", correct ? "PASS (results match)" : "FAIL (results differ)");

    if (n <= 10) {
        putchar('\n');
        print_preview("Result matrix", c_par, n);
    } else {
        printf("Result matrix preview omitted (matrix is larger than 10 x 10).\n");
    }

    free(a); free(b); free(c_seq); free(c_par);
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
