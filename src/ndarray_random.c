#ifndef _WIN32
    #ifndef _GNU_SOURCE
        #define _GNU_SOURCE
    #endif
#endif
#define _USE_MATH_DEFINES

#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "ndarray_internal.h"

#if defined(_GNU_SOURCE) && !defined(_WIN32)
struct drand48_data rng_state;
#ifdef _OPENMP
// srand48_r(time(NULL) ^ (omp_get_thread_num() << 16), &rng_state);
#else
// srand48_r(time(NULL), &rng_state);
#endif
#endif

static double next_rand() {
        double u;
#if defined(_GNU_SOURCE) && !defined(_WIN32)
        drand48_r(&rng_state, &u);
#else
        u = ((double)rand() / RAND_MAX);
#endif
        return u;
}

/**
 * Generate a Poisson random number using Knuth's algorithm
 */
static inline double knuth_poisson(double lambda) {
    double L = exp(-lambda);
    int k = 0;
    double p = 1.0;
    do {
        k++;
        p *= next_rand();
    } while (p > L);
    return k - 1;
}

/**
 * Generate a Gaussian random number using Box-Muller transform
 */
static inline double box_muller_gaussian(double mean, double std) {
    double u1 = next_rand();
    double u2 = next_rand();
    if (u1 < 1e-10) u1 = 1e-10;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + std * z;
}

NDArray ndarray_fill_randnorm(const NDArray A, double mean, double stddev) {
        assert(A != NULL && "ndarray cannot be NULL");
#ifdef _OPENMP
        size_t size = ndarray_size(A);
        if (size >= OMP_THRESHOLD) {
                OMP_PRAGMA(omp parallel)
                {
                        OMP_PRAGMA(omp for)
#endif
                        for_range(size_t, i, 0, ndarray_size(A))
                                A->data[i] = box_muller_gaussian(mean, stddev);
#ifdef _OPENMP
                }
        } else {
                for_range(size_t, i, 0, ndarray_size(A))
                        A->data[i] = box_muller_gaussian(mean, stddev);
        } 
#endif
    return (NDArray)A;
}

NDArray ndarray_new_randnorm(const size_t *dims, double mean, double stddev) {
    NDArray t = ndarray_new(dims);
    return ndarray_fill_randnorm(t, mean, stddev);
}

NDArray ndarray_fill_randunif(const NDArray A, double low, double high) {
    assert(A != NULL && "ndarray cannot be NULL");
    size_t size = ndarray_size(A);
    double range = high - low;
#ifdef _OPENMP
    if (size >= OMP_THRESHOLD) {
        OMP_PRAGMA(omp parallel)
        {
            OMP_PRAGMA(omp for)
#endif
            for (size_t i = 0; i < size; ++i) {
                double u = next_rand();
                A->data[i] = low + range * u;
            }
#ifdef _OPENMP
        }
    } else {
            for (size_t i = 0; i < size; ++i) {
                double u = next_rand();
                A->data[i] = low + range * u;
            }
    }
#endif
    return (NDArray)A;
}

NDArray ndarray_new_randunif(const size_t *dims, double low, double high) {
    NDArray t = ndarray_new(dims);
    return ndarray_fill_randunif(t, low, high);
}

NDArray ndarray_fill_randpoisson(const NDArray A, double lambda) {
    assert(A != NULL && "ndarray cannot be NULL");
    size_t size = ndarray_size(A);
#ifdef _OPENMP
    if (size >= OMP_THRESHOLD) {
        OMP_PRAGMA(omp parallel)
        {
            OMP_PRAGMA(omp for)
#endif
            for (size_t i = 0; i < size; ++i) {
                A->data[i] = knuth_poisson(lambda);
            }
#ifdef _OPENMP
        }
    } else {
        for (size_t i = 0; i < size; ++i) {
            A->data[i] = knuth_poisson(lambda);
        }
    }
#endif
    return (NDArray)A;
}

NDArray ndarray_new_randpoisson(const size_t *dims, double lambda) {
    NDArray t = ndarray_new(dims);
    return ndarray_fill_randpoisson(t, lambda);
}
