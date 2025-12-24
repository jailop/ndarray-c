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
#include <openblas/cblas.h>
#include <omp.h>
#include "ndarray.h"

#define BLOCK_SIZE 64
#define OMP_PRAGMA(x) _Pragma(#x)

/* Internal helper functions */
size_t ndarray_size(NDArray t);
size_t ndarray_offset(NDArray t, size_t *pos);
size_t compute_stride(NDArray A, int axis);
double generate_gaussian(double mean, double std);

#endif /* NDARRAY_INTERNAL_H */
