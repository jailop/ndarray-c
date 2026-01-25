#include "ndarray_internal.h"
#include <math.h>

NDArray ndarray_mapfnc(const NDArray A, double (*func)(double)) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    assert(func != NULL && "function pointer cannot be NULL");
    size_t size = ndarray_size(A);
    OMP_PRAGMA(omp parallel for)
    for (size_t i = 0; i < size; ++i) {
        A->data[i] = func(A->data[i]);
    }
    return A;
}


NDArray ndarray_mapfnc_par(const NDArray A, double (*func)(double, double), double v) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    assert(func != NULL && "function pointer cannot be NULL");
    size_t size = ndarray_size(A);
    OMP_PRAGMA(omp parallel for)
    for (size_t i = 0; i < size; ++i) {
        A->data[i] = func(A->data[i], v);
    }
    return A;
}

NDArray ndarray_exp(const NDArray A) { return ndarray_mapfnc(A, exp); }
NDArray ndarray_log(const NDArray A) { return ndarray_mapfnc(A, log); }
NDArray ndarray_sqrt(const NDArray A) { return ndarray_mapfnc(A, sqrt); }
NDArray ndarray_sin(const NDArray A) { return ndarray_mapfnc(A, sin); }
NDArray ndarray_cos(const NDArray A) { return ndarray_mapfnc(A, cos); }
NDArray ndarray_tan(const NDArray A) { return ndarray_mapfnc(A, tan); }
NDArray ndarray_sinh(const NDArray A) { return ndarray_mapfnc(A, sinh); }
NDArray ndarray_cosh(const NDArray A) { return ndarray_mapfnc(A, cosh); }
NDArray ndarray_tanh(const NDArray A) { return ndarray_mapfnc(A, tanh); }
NDArray ndarray_asin(const NDArray A) { return ndarray_mapfnc(A, asin); }
NDArray ndarray_acos(const NDArray A) { return ndarray_mapfnc(A, acos); }
NDArray ndarray_atan(const NDArray A) { return ndarray_mapfnc(A, atan); }
NDArray ndarray_pow(const NDArray A, double exponent) {
    return ndarray_mapfnc_par(A, pow, exponent);
}

