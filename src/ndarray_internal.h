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
#include <complex.h>
#include <stdbool.h>

#ifndef _WIN32
    #include <sys/ioctl.h>
    #include <unistd.h>
#else
    #include <io.h>
    #define isatty _isatty
    #define fileno _fileno
#endif

#if defined(__APPLE__)
    #include <cblas.h>
#elif defined(_WIN32)
    // On Windows with vcpkg, OpenBLAS includes are in openblas/ directory
    // but CMake adds that to include path, so we include directly
    #include <cblas.h>
#else
    #include <openblas/cblas.h>
#endif

#ifdef _OPENMP
    #include <omp.h>
    #ifdef _MSC_VER
        // MSVC OpenMP 2.0 has limitations:
        // - No simd support
        // - Loop variables must be signed int and cannot be declared in for statement
        #define OMP_PRAGMA(x) _Pragma(#x)
        // Strip "simd" from pragmas for MSVC
        #define OMP_PRAGMA_SIMD(x) _Pragma("omp parallel for")
    #else
        #define OMP_PRAGMA(x) _Pragma(#x)
        #define OMP_PRAGMA_SIMD(x) _Pragma(x)
    #endif
#else
    #define OMP_PRAGMA(x)
    #define OMP_PRAGMA_SIMD(x)
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
size_t compute_stride(NDArray A, int axis);
double generate_gaussian(double mean, double std);

// Type system functions
const NDATypeInfo* ndarray_get_type_info(NDAType dtype);
NDAType ndarray_promote_types(NDAType type1, NDAType type2);
int ndarray_is_complex_type(NDAType dtype);
size_t ndarray_element_size(NDAType dtype);

// Type conversion helpers
void ndarray_convert_double_to_complex64(double src, void *dst);
void ndarray_convert_complex64_to_double(const void *src, double *dst);
void ndarray_convert_float32_to_float64(float src, double *dst);
void ndarray_convert_float64_to_float32(double src, float *dst);

// BLAS dispatch helpers
void ndarray_blas_axpy(size_t n, double alpha, const NDArray X, const NDArray Y, NDArray result);
void ndarray_blas_scal(size_t n, double alpha, const NDArray X);

#endif /* NDARRAY_INTERNAL_H */
