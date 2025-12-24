/**
 * Benchmark framework to compare OpenMP vs sequential performance
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "ndarray.h"

// High-resolution timer
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Benchmark result structure
typedef struct {
    const char *name;
    double time_sec;
    size_t operations;
} BenchmarkResult;

void print_result(BenchmarkResult *result) {
    printf("  %-40s %10.6f sec", result->name, result->time_sec);
    if (result->operations > 0) {
        double ops_per_sec = result->operations / result->time_sec;
        if (ops_per_sec > 1e9) {
            printf("  (%8.2f G ops/sec)", ops_per_sec / 1e9);
        } else if (ops_per_sec > 1e6) {
            printf("  (%8.2f M ops/sec)", ops_per_sec / 1e6);
        } else if (ops_per_sec > 1e3) {
            printf("  (%8.2f K ops/sec)", ops_per_sec / 1e3);
        } else {
            printf("  (%8.2f ops/sec)", ops_per_sec);
        }
    }
    printf("\n");
}

// Benchmark: Array creation and initialization
BenchmarkResult bench_array_creation(size_t rows, size_t cols, int iterations) {
    BenchmarkResult result = {"Array creation (zeros/ones)", 0.0, 0};
    
    double start = get_time();
    for (int i = 0; i < iterations; ++i) {
        NDArray A = ndarray_new_zeros(NDA_DIMS(rows, cols));
        NDArray B = ndarray_new_ones(NDA_DIMS(rows, cols));
        ndarray_free_all(NDA_LIST(A, B));
    }
    result.time_sec = get_time() - start;
    result.operations = (size_t)iterations * rows * cols * 2;
    
    return result;
}

// Benchmark: Element-wise operations
BenchmarkResult bench_elementwise_ops(size_t rows, size_t cols, int iterations) {
    BenchmarkResult result = {"Element-wise add/multiply", 0.0, 0};
    
    NDArray A = ndarray_new_arange(NDA_DIMS(rows, cols), 0.0, (double)(rows * cols), 1.0);
    NDArray B = ndarray_new_ones(NDA_DIMS(rows, cols));
    
    double start = get_time();
    for (int i = 0; i < iterations; ++i) {
        ndarray_add(A, B);
        ndarray_mul(A, B);
    }
    result.time_sec = get_time() - start;
    result.operations = (size_t)iterations * rows * cols * 2;
    
    ndarray_free_all(NDA_LIST(A, B));
    return result;
}

// Benchmark: Matrix multiplication
BenchmarkResult bench_matmul(size_t m, size_t k, size_t n, int iterations) {
    BenchmarkResult result = {"Matrix multiplication (matmul)", 0.0, 0};
    
    NDArray A = ndarray_new_ones(NDA_DIMS(m, k));
    NDArray B = ndarray_new_ones(NDA_DIMS(k, n));
    
    double start = get_time();
    for (int i = 0; i < iterations; ++i) {
        NDArray C = ndarray_new_matmul(A, B);
        ndarray_free(C);
    }
    result.time_sec = get_time() - start;
    result.operations = (size_t)iterations * m * n * k * 2; // mul + add per element
    
    ndarray_free_all(NDA_LIST(A, B));
    return result;
}

// Benchmark: Transpose
BenchmarkResult bench_transpose(size_t rows, size_t cols, int iterations) {
    BenchmarkResult result = {"Transpose 2D", 0.0, 0};
    
    NDArray A = ndarray_new_arange(NDA_DIMS(rows, cols), 0.0, (double)(rows * cols), 1.0);
    
    double start = get_time();
    for (int i = 0; i < iterations; ++i) {
        NDArray B = ndarray_new_transpose(A);
        ndarray_free(B);
    }
    result.time_sec = get_time() - start;
    result.operations = (size_t)iterations * rows * cols;
    
    ndarray_free(A);
    return result;
}

// Benchmark: Aggregation operations
BenchmarkResult bench_aggregation(size_t rows, size_t cols, int iterations) {
    BenchmarkResult result = {"Aggregations (sum/mean/max/min)", 0.0, 0};
    
    NDArray A = ndarray_new_arange(NDA_DIMS(rows, cols), 0.0, (double)(rows * cols), 1.0);
    
    double start = get_time();
    for (int i = 0; i < iterations; ++i) {
        NDArray s = ndarray_new_axis_aggr(A, -1, NDARRAY_AGGR_SUM);
        NDArray m = ndarray_new_axis_aggr(A, -1, NDARRAY_AGGR_MEAN);
        NDArray mx = ndarray_new_axis_aggr(A, -1, NDARRAY_AGGR_MAX);
        NDArray mn = ndarray_new_axis_aggr(A, -1, NDARRAY_AGGR_MIN);
        ndarray_free_all(NDA_LIST(s, m, mx, mn));
    }
    result.time_sec = get_time() - start;
    result.operations = (size_t)iterations * rows * cols * 4;
    
    ndarray_free(A);
    return result;
}

// Benchmark: Array concatenation
BenchmarkResult bench_concat(size_t rows, size_t cols, int iterations) {
    BenchmarkResult result = {"Array concatenation", 0.0, 0};
    
    NDArray A = ndarray_new_ones(NDA_DIMS(rows, cols));
    NDArray B = ndarray_new_ones(NDA_DIMS(rows, cols));
    
    double start = get_time();
    for (int i = 0; i < iterations; ++i) {
        NDArray C = ndarray_new_concat(0, NDA_LIST(A, B));
        ndarray_free(C);
    }
    result.time_sec = get_time() - start;
    result.operations = (size_t)iterations * rows * cols * 2;
    
    ndarray_free_all(NDA_LIST(A, B));
    return result;
}

// Benchmark: Outer product (tensordot)
BenchmarkResult bench_outer_product(size_t size_a, size_t size_b, int iterations) {
    BenchmarkResult result = {"Outer product (tensordot)", 0.0, 0};
    
    NDArray A = ndarray_new_arange(NDA_DIMS(1, size_a), 0.0, (double)size_a, 1.0);
    NDArray B = ndarray_new_arange(NDA_DIMS(1, size_b), 0.0, (double)size_b, 1.0);
    
    double start = get_time();
    for (int i = 0; i < iterations; ++i) {
        NDArray C = ndarray_new_tensordot(A, B, NDA_NOAXES, NDA_NOAXES);
        ndarray_free(C);
    }
    result.time_sec = get_time() - start;
    result.operations = (size_t)iterations * size_a * size_b;
    
    ndarray_free_all(NDA_LIST(A, B));
    return result;
}

// Benchmark: Tensor contraction (dot product)
BenchmarkResult bench_tensor_contraction(size_t size, int iterations) {
    BenchmarkResult result = {"Tensor contraction (dot product)", 0.0, 0};
    
    NDArray A = ndarray_new_ones(NDA_DIMS(1, size));
    NDArray B = ndarray_new_ones(NDA_DIMS(1, size));
    
    double start = get_time();
    for (int i = 0; i < iterations; ++i) {
        NDArray C = ndarray_new_tensordot(A, B, NDA_AXES(0), NDA_AXES(0));
        ndarray_free(C);
    }
    result.time_sec = get_time() - start;
    result.operations = (size_t)iterations * size * 2; // mul + add
    
    ndarray_free_all(NDA_LIST(A, B));
    return result;
}

// Benchmark: Linspace (parallel loop)
BenchmarkResult bench_linspace(size_t size, int iterations) {
    BenchmarkResult result = {"Linspace generation", 0.0, 0};
    
    double start = get_time();
    for (int i = 0; i < iterations; ++i) {
        NDArray A = ndarray_new_linspace(NDA_DIMS(1, size), 0.0, 1000.0, size);
        ndarray_free(A);
    }
    result.time_sec = get_time() - start;
    result.operations = (size_t)iterations * size;
    
    return result;
}

void print_header(const char *title) {
    printf("\n");
    printf("================================================================================\n");
    printf("%s\n", title);
    printf("================================================================================\n");
}

void print_comparison(BenchmarkResult *seq, BenchmarkResult *par) {
    double speedup = seq->time_sec / par->time_sec;
    printf("  Speedup: %.2fx  ", speedup);
    if (speedup > 1.0) {
        printf("(%.1f%% faster with OpenMP)\n", (speedup - 1.0) * 100);
    } else {
        printf("(%.1f%% slower with OpenMP)\n", (1.0 - speedup) * 100);
    }
}

int main(int argc, char *argv[]) {
    printf("\n");
    printf("################################################################################\n");
    printf("#                  NDARRAY BENCHMARK SUITE                                    #\n");
    printf("################################################################################\n");
    
#ifdef USE_OPENMP
    printf("\nCompiled WITH OpenMP support\n");
#else
    printf("\nCompiled WITHOUT OpenMP support (sequential)\n");
#endif
    
    // Small arrays (cache-friendly)
    print_header("Small Arrays (1000 x 1000)");
    {
        BenchmarkResult r1 = bench_array_creation(1000, 1000, 10);
        print_result(&r1);
        
        BenchmarkResult r2 = bench_elementwise_ops(1000, 1000, 100);
        print_result(&r2);
        
        BenchmarkResult r3 = bench_matmul(1000, 100, 100, 10);
        print_result(&r3);
        
        BenchmarkResult r4 = bench_transpose(1000, 1000, 20);
        print_result(&r4);
        
        BenchmarkResult r5 = bench_aggregation(1000, 1000, 100);
        print_result(&r5);
        
        BenchmarkResult r6 = bench_concat(1000, 1000, 50);
        print_result(&r6);
    }
    
    // Medium arrays
    print_header("Medium Arrays (3000 x 3000)");
    {
        BenchmarkResult r1 = bench_array_creation(3000, 3000, 5);
        print_result(&r1);
        
        BenchmarkResult r2 = bench_elementwise_ops(3000, 3000, 20);
        print_result(&r2);
        
        BenchmarkResult r3 = bench_matmul(3000, 300, 300, 5);
        print_result(&r3);
        
        BenchmarkResult r4 = bench_transpose(3000, 3000, 10);
        print_result(&r4);
        
        BenchmarkResult r5 = bench_aggregation(3000, 3000, 20);
        print_result(&r5);
        
        BenchmarkResult r6 = bench_concat(3000, 3000, 10);
        print_result(&r6);
    }
    
    // Large arrays (stress test)
    print_header("Large Arrays (5000 x 5000)");
    {
        BenchmarkResult r1 = bench_array_creation(5000, 5000, 3);
        print_result(&r1);
        
        BenchmarkResult r2 = bench_elementwise_ops(5000, 5000, 10);
        print_result(&r2);
        
        BenchmarkResult r3 = bench_matmul(5000, 500, 500, 2);
        print_result(&r3);
        
        BenchmarkResult r4 = bench_transpose(5000, 5000, 3);
        print_result(&r4);
        
        BenchmarkResult r5 = bench_aggregation(5000, 5000, 10);
        print_result(&r5);
        
        BenchmarkResult r6 = bench_concat(5000, 5000, 5);
        print_result(&r6);
    }
    
    // Special: Tensor operations
    print_header("Tensor Operations");
    {
        BenchmarkResult r1 = bench_linspace(50000, 20);
        print_result(&r1);
        
        BenchmarkResult r2 = bench_outer_product(200, 200, 20);
        print_result(&r2);
        
        BenchmarkResult r3 = bench_tensor_contraction(1000, 50);
        print_result(&r3);
    }
    
    printf("\n");
    printf("================================================================================\n");
    printf("Benchmark Complete\n");
    printf("================================================================================\n");
    printf("\n");
    
    return 0;
}
