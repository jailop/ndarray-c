#ifndef _WIN32
    #ifndef _GNU_SOURCE
        #define _GNU_SOURCE
    #endif
#endif
#define _USE_MATH_DEFINES

#include <time.h>
#include <stdlib.h>
#include "ndarray_internal.h"


/**
 * Generate a Gaussian random number using Box-Muller transform
 */
#if defined(_GNU_SOURCE) && !defined(_WIN32)
static inline double box_muller_gaussian(struct drand48_data *rng_state,
        double mean, double std) {
    double u1, u2;
    drand48_r(rng_state, &u1);
    drand48_r(rng_state, &u2);
    if (u1 < 1e-10) u1 = 1e-10;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + std * z;
}
#else
// Portable version using rand() (not thread-safe but works on all platforms)
static inline double box_muller_gaussian_simple(double mean, double std) {
    double u1 = ((double)rand() / RAND_MAX);
    double u2 = ((double)rand() / RAND_MAX);
    if (u1 < 1e-10) u1 = 1e-10;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + std * z;
}
#endif



NDArray ndarray_new_randnorm(const size_t *dims, double mean, double stddev) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    
#if defined(_GNU_SOURCE) && !defined(_WIN32)
    if (size >= OMP_THRESHOLD) {
        OMP_PRAGMA(omp parallel)
        {
#ifdef _OPENMP
            struct drand48_data rng_state;
            srand48_r(time(NULL) ^ (omp_get_thread_num() << 16), &rng_state);
            OMP_PRAGMA(omp for)
            for (size_t i = 0; i < size; ++i) {
                t->data[i] = box_muller_gaussian(&rng_state, mean, stddev);
            }
#else
            struct drand48_data rng_state;
            srand48_r(time(NULL), &rng_state);
            for (size_t i = 0; i < size; ++i) {
                t->data[i] = box_muller_gaussian(&rng_state, mean, stddev);
            }
#endif
        }
    } else {
        struct drand48_data rng_state;
        srand48_r(time(NULL), &rng_state);
        for (size_t i = 0; i < size; ++i) {
            t->data[i] = box_muller_gaussian(&rng_state, mean, stddev);
        }
    }
#else
    // Portable version - OpenMP disabled for random number generation on non-GNU systems
    srand((unsigned int)time(NULL));
    for (size_t i = 0; i < size; ++i) {
        t->data[i] = box_muller_gaussian_simple(mean, stddev);
    }
#endif
    return t;
}

NDArray ndarray_new_randunif(const size_t *dims, double low, double high) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    double range = high - low;
    
#if defined(_GNU_SOURCE) && !defined(_WIN32)
    if (size >= OMP_THRESHOLD) {
        OMP_PRAGMA(omp parallel)
        {
#ifdef _OPENMP
            struct drand48_data rng_state;
            srand48_r(time(NULL) ^ (omp_get_thread_num() << 16), &rng_state);
            OMP_PRAGMA(omp for)
            for (size_t i = 0; i < size; ++i) {
                double u;
                drand48_r(&rng_state, &u);
                t->data[i] = low + range * u;
            }
#else
            struct drand48_data rng_state;
            srand48_r(time(NULL), &rng_state);
            for (size_t i = 0; i < size; ++i) {
                double u;
                drand48_r(&rng_state, &u);
                t->data[i] = low + range * u;
            }
#endif
        }
    } else {
        struct drand48_data rng_state;
        srand48_r(time(NULL), &rng_state);
        for (size_t i = 0; i < size; ++i) {
            double u;
            drand48_r(&rng_state, &u);
            t->data[i] = low + range * u;
        }
    }
#else
    // Portable version - OpenMP disabled for random number generation on non-GNU systems
    srand((unsigned int)time(NULL));
    for (size_t i = 0; i < size; ++i) {
        t->data[i] = low + range * ((double)rand() / RAND_MAX);
    }
#endif
    
    return t;
}
