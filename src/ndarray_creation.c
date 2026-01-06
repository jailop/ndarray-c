/**
 * Array creation functions
 */

#define _USE_MATH_DEFINES
#include "ndarray_internal.h"
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * Generate a Gaussian random number using Box-Muller transform
 * with given mean and standard deviation (thread-safe version)
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
    
    // Calculate actual number of elements to fill
    size_t num_elements = 0;
    if (step > 0) {
        num_elements = (size_t)ceil((stop - start) / step);
    }
    if (num_elements > size) num_elements = size;
    
    // Parallelize for large arrays
    if (num_elements >= OMP_THRESHOLD) {
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < num_elements; ++i) {
            t->data[i] = start + i * step;
        }
    } else {
        // Sequential for small arrays
        for (size_t i = 0; i < num_elements; ++i) {
            t->data[i] = start + i * step;
        }
    }
    
    // Zero out remaining elements if any
    if (num_elements < size) {
        memset(t->data + num_elements, 0, (size - num_elements) * sizeof(double));
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
    
    // Use parallelization for large arrays
    if (size >= OMP_THRESHOLD) {
        OMP_PRAGMA(omp parallel)
        {
#ifdef _OPENMP
            // Thread-safe random number generation with unique seed per thread
            unsigned int seed = (unsigned int)(time(NULL) ^ (omp_get_thread_num() << 16));
            
            OMP_PRAGMA(omp for)
            for (size_t i = 0; i < size; i += 2) {
                // Box-Muller transform - generates two independent values
                double u1 = rand_r(&seed) / (double)RAND_MAX;
                double u2 = rand_r(&seed) / (double)RAND_MAX;
                
                // Ensure u1 is not zero to avoid log(0)
                if (u1 < 1e-10) u1 = 1e-10;
                
                double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                t->data[i] = mean + stddev * z0;
                
                if (i + 1 < size) {
                    double z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
                    t->data[i + 1] = mean + stddev * z1;
                }
            }
#else
            // Fallback to serial for non-OpenMP
            for (size_t i = 0; i < size; ++i) {
                t->data[i] = generate_gaussian(mean, stddev);
            }
#endif
        }
    } else {
        // Sequential for small arrays to avoid overhead
        for (size_t i = 0; i < size; ++i) {
            t->data[i] = generate_gaussian(mean, stddev);
        }
    }
    
    return t;
}

NDArray ndarray_new_randunif(size_t *dims, double low, double high) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    double range = high - low;
    
    // Use parallelization for large arrays
    if (size >= OMP_THRESHOLD) {
        OMP_PRAGMA(omp parallel)
        {
#ifdef _OPENMP
            // Thread-safe random number generation with unique seed per thread
            unsigned int seed = (unsigned int)(time(NULL) ^ (omp_get_thread_num() << 16));
            
            OMP_PRAGMA(omp for simd)
            for (size_t i = 0; i < size; ++i) {
                t->data[i] = low + range * (rand_r(&seed) / (double)RAND_MAX);
            }
#else
            // Fallback to serial for non-OpenMP
            for (size_t i = 0; i < size; ++i) {
                t->data[i] = low + range * ((double)rand() / RAND_MAX);
            }
#endif
        }
    } else {
        // Sequential for small arrays to avoid overhead
        for (size_t i = 0; i < size; ++i) {
            t->data[i] = low + range * ((double)rand() / RAND_MAX);
        }
    }
    
    return t;
}
