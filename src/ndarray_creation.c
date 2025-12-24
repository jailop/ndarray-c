/**
 * Array creation functions
 */

#include "ndarray_internal.h"

/**
 * Generate a Gaussian random number using Box-Muller transform
 * with given mean and standard deviation
 */
double generate_gaussian(double mean, double std) {
    static int have_spare = 0;
    static double spare;
    if (have_spare) {
        have_spare = 0;
        return mean + std * spare;
    }
    have_spare = 1;
    double u, v, s;
    do {
        u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + std * (u * s);
}

NDArray ndarray_new_zeros(size_t *dims) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    memset(t->data, 0, sizeof(double) * size);
    return t;
}

NDArray ndarray_new_from_data(size_t *dims, double *data) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    memcpy(t->data, data, sizeof(double) * size);
    return t;
}

NDArray ndarray_new_ones(size_t *dims) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        t->data[i] = 1.0;
    }
    return t;
}

NDArray ndarray_new_full(size_t *dims, double value) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        t->data[i] = value;
    }
    return t;
}

NDArray ndarray_new_arange(size_t *dims, double start, double stop,
        double step) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    for (size_t i = 0; i < size; ++i) {
        double val = start + i * step;
        if (val >= stop) break;
        t->data[i] = val;
    }
    return t;
}

NDArray ndarray_new_linspace(size_t *dims, double start, double stop,
        size_t num) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    if (num <= 1) {
        t->data[0] = start;
        return t;
    }
    double step = (stop - start) / (num - 1);
    size_t max_idx = (size < num) ? size : num;
    OMP_PRAGMA(omp parallel for)
    for (size_t i = 0; i < max_idx; ++i) {
        t->data[i] = start + i * step;
    }
    return t;
}

NDArray ndarray_new_randnorm(size_t *dims, double mean, double stddev) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    for (size_t i = 0; i < size; ++i) {
        t->data[i] = generate_gaussian(mean, stddev);
    }
    return t;
}

NDArray ndarray_new_randunif(size_t *dims, double low, double high) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    for (size_t i = 0; i < size; ++i) {
        t->data[i] = low + (high - low) * ((double)rand() / RAND_MAX);
    }
    return t;
}
