/**
 * Internal header for ndarray implementation
 * Contains shared utilities and macros
 */

#ifndef NDARRAY_INTERNAL_H
#define NDARRAY_INTERNAL_H

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/ioctl.h>
#include <unistd.h>

#ifdef __APPLE__
    #include <cblas.h>
#else
    #include <openblas/cblas.h>
#endif

#ifdef _OPENMP
    #include <omp.h>
    #define OMP_PRAGMA(x) _Pragma(#x)
#else
    #define OMP_PRAGMA(x)
#endif

#include "ndarray.h"

#define BLOCK_SIZE 64

#ifndef OMP_THRESHOLD
    #define OMP_THRESHOLD 1000
#endif

#ifndef TRANSPOSE_BLOCK_SIZE
    #define TRANSPOSE_BLOCK_SIZE 32
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

size_t ndarray_size(NDArray t);
size_t ndarray_offset(NDArray t, size_t *pos);
size_t compute_stride(NDArray A, int axis);
double generate_gaussian(double mean, double std);

#endif /* NDARRAY_INTERNAL_H */
