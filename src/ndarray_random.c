#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define _USE_MATH_DEFINES

#include <time.h>
#include <stdlib.h>
#include "ndarray_internal.h"


/**
 * Generate a Gaussian random number using Box-Muller transform
 */
static inline double box_muller_gaussian(struct drand48_data *rng_state,
        double mean, double std) {
    double u1, u2;
#ifdef _GNU_SOURCE
    drand48_r(rng_state, &u1);
    drand48_r(rng_state, &u2);
#else
    unsigned int *seed = (unsigned int *)rng_state;
    u1 = rand_r(seed) / (double)RAND_MAX;
    u2 = rand_r(seed) / (double)RAND_MAX;
#endif
    if (u1 < 1e-10) u1 = 1e-10;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + std * z;
}



NDArray ndarray_new_randnorm(size_t *dims, double mean, double stddev) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    
    if (size >= OMP_THRESHOLD) {
        OMP_PRAGMA(omp parallel)
        {
#ifdef _OPENMP
#ifdef _GNU_SOURCE
            struct drand48_data rng_state;
            srand48_r(time(NULL) ^ (omp_get_thread_num() << 16), &rng_state);
#else
            unsigned int rng_state = (unsigned int)(time(NULL)
                    ^ (omp_get_thread_num() << 16));
#endif
            OMP_PRAGMA(omp for)
            for (size_t i = 0; i < size; ++i) {
                t->data[i] = box_muller_gaussian(&rng_state, mean, stddev);
            }
#else
#ifdef _GNU_SOURCE
            struct drand48_data rng_state;
            srand48_r(time(NULL), &rng_state);
#else
            unsigned int rng_state = (unsigned int)time(NULL);
#endif
            for (size_t i = 0; i < size; ++i) {
                t->data[i] = box_muller_gaussian(&rng_state, mean, stddev);
            }
#endif
        }
    } else {
#ifdef _GNU_SOURCE
        struct drand48_data rng_state;
        srand48_r(time(NULL), &rng_state);
#else
        unsigned int rng_state = (unsigned int)time(NULL);
#endif
        for (size_t i = 0; i < size; ++i) {
            t->data[i] = box_muller_gaussian(&rng_state, mean, stddev);
        }
    }
    return t;
}

NDArray ndarray_new_randunif(size_t *dims, double low, double high) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    double range = high - low;
    
    if (size >= OMP_THRESHOLD) {
        OMP_PRAGMA(omp parallel)
        {
#ifdef _OPENMP
#ifdef _GNU_SOURCE
            struct drand48_data rng_state;
            srand48_r(time(NULL) ^ (omp_get_thread_num() << 16), &rng_state);
            OMP_PRAGMA(omp for)
            for (size_t i = 0; i < size; ++i) {
                double u;
                drand48_r(&rng_state, &u);
                t->data[i] = low + range * u;
            }
#else
            unsigned int seed = (unsigned int)(time(NULL) ^ (omp_get_thread_num() << 16));
            OMP_PRAGMA(omp for)
            for (size_t i = 0; i < size; ++i) {
                t->data[i] = low + range * (rand_r(&seed) / (double)RAND_MAX);
            }
#endif
#else
#ifdef _GNU_SOURCE
            struct drand48_data rng_state;
            srand48_r(time(NULL), &rng_state);
            for (size_t i = 0; i < size; ++i) {
                double u;
                drand48_r(&rng_state, &u);
                t->data[i] = low + range * u;
            }
#else
            for (size_t i = 0; i < size; ++i) {
                t->data[i] = low + range * ((double)rand() / RAND_MAX);
            }
#endif
#endif
        }
    } else {
#ifdef _GNU_SOURCE
        struct drand48_data rng_state;
        srand48_r(time(NULL), &rng_state);
        for (size_t i = 0; i < size; ++i) {
            double u;
            drand48_r(&rng_state, &u);
            t->data[i] = low + range * u;
        }
#else
        for (size_t i = 0; i < size; ++i) {
            t->data[i] = low + range * ((double)rand() / RAND_MAX);
        }
#endif
    }
    
    return t;
}
