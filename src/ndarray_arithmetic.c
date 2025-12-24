/**
 * Arithmetic operations on ndarrays
 */

#include "ndarray_internal.h"

NDArray ndarray_add(NDArray A, NDArray B) {
    assert(A != NULL && B != NULL && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    assert(A->ndim == B->ndim
            && "ndarrays must have same number of dimensions");
    for (size_t i = 0; i < A->ndim; ++i) {
        assert(A->dims[i] == B->dims[i]
                && "ndarrays must have matching dimensions");
    }
    size_t size = ndarray_size(A);
    // Use CBLAS daxpy: y = alpha*x + y (A = 1.0*B + A)
    cblas_daxpy(size, 1.0, B->data, 1, A->data, 1);
    return A;
}

NDArray ndarray_mul(NDArray A, NDArray B) {
    assert(A != NULL && B != NULL
            && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    assert(A->ndim == B->ndim
            && "ndarrays must have same number of dimensions");
    for (size_t i = 0; i < A->ndim; ++i) {
        assert(A->dims[i] == B->dims[i]
                && "ndarrays must have matching dimensions");
    }
    size_t size = ndarray_size(A);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        A->data[i] *= B->data[i];
    }
    return A;
}

NDArray ndarray_add_scalar(NDArray A, double scalar) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    size_t size = ndarray_size(A);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        A->data[i] += scalar;
    }
    return A;
}

NDArray ndarray_mul_scalar(NDArray A, double scalar) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    size_t size = ndarray_size(A);
    // Use CBLAS dscal: x = alpha*x
    cblas_dscal(size, scalar, A->data, 1);
    return A;
}

NDArray ndarray_mapfnc(NDArray A, double (*func)(double)) {
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
