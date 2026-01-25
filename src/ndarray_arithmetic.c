/**
 * Arithmetic operations on ndarrays
 */

#include "ndarray_internal.h"

NDArray ndarray_add(const NDArray A, const NDArray B) {
    assert((A != NULL && B != NULL) && "ndarrays cannot be NULL");
    assert((A->ndim >= 2 && B->ndim >= 2) && "ndarrays must have at least 2 dimensions");
    assert((A->ndim == B->ndim) && "ndarrays must have same number of dimensions");
    
    for (size_t i = 0; i < A->ndim; ++i) {
        assert(A->dims[i] == B->dims[i] && "ndarrays must have matching dimensions");
    }
    
    // Phase 2: Type-aware addition with same-type BLAS dispatch
    // Check if types match for direct operation
    if (A->dtype != B->dtype) {
        assert(0 && "Mixed-type arithmetic not yet implemented");
        return A;
    }
    
    // Same-type operation - dispatch to appropriate BLAS
    size_t array_size = ndarray_size(A);
    
    switch (A->dtype) {
        case NDA_REAL64:
            cblas_daxpy(array_size, 1.0, A->data, 1, B->data, 1);
            break;
        case NDA_REAL32:
            cblas_saxpy(array_size, 1.0, A->data, 1, B->data, 1);
            break;
        case NDA_COMPLEX64:
            {
                double complex alpha = 1.0 + 0.0 * I;
                cblas_zaxpy(array_size, &alpha, A->data, 1, B->data, 1);
            }
            break;
        case NDA_COMPLEX32:
            {
                float complex alpha = 1.0f + 0.0f * I;
                cblas_caxpy(array_size, &alpha, A->data, 1, B->data, 1);
            }
            break;
        default:
            assert(0 && "Unsupported data type");
            return A;
    }
    
    return A;
}

NDArray ndarray_add_scalar(const NDArray A, double scalar) {
    assert((A != NULL) && "ndarray cannot be NULL");
    assert((A->ndim >= 2) && "ndarray must have at least 2 dimensions");
    
    size_t array_size = ndarray_size(A);
    
    switch (A->dtype) {
        case NDA_REAL64:
            cblas_daxpy(array_size, scalar, A->data, 1, A->data, 1);
            break;
        case NDA_REAL32:
            cblas_saxpy(array_size, scalar, A->data, 1, A->data, 1);
            break;
        case NDA_COMPLEX64:
            {
                double complex alpha = scalar + 0.0 * I;
                cblas_zaxpy(array_size, &alpha, A->data, 1, A->data, 1);
            }
            break;
        case NDA_COMPLEX32:
            {
                float complex alpha = (float)scalar + 0.0f * I;
                cblas_caxpy(array_size, &alpha, A->data, 1, A->data, 1);
            }
            break;
        default:
            assert(0 && "Unsupported data type");
            return A;
    }
    
    return A;
}

NDArray ndarray_mul(const NDArray A, const NDArray B) {
    assert((A != NULL && B != NULL) && "ndarrays cannot be NULL");
    assert((A->ndim >= 2 && B->ndim >= 2) && "ndarrays must have at least 2 dimensions");
    assert((A->ndim == B->ndim) && "ndarrays must have same number of dimensions");
    
    for (size_t i = 0; i < A->ndim; ++i) {
        assert(A->dims[i] == B->dims[i] && "ndarrays must have matching dimensions");
    }
    
    // Phase 2: Type-aware multiplication with same-type BLAS dispatch
    // Check if types match for direct operation
    if (A->dtype != B->dtype) {
        assert(0 && "Mixed-type multiplication not yet implemented");
        return A;
    }
    
    // Same-type operation - dispatch to appropriate BLAS
    size_t array_size = ndarray_size(A);
    
    switch (A->dtype) {
        case NDA_REAL64:
            // Element-wise multiplication using loop
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < array_size; ++i) {
                ((double*)A->data)[i] *= ((double*)B->data)[i];
            }
            break;
        case NDA_REAL32:
            // Element-wise multiplication for float32
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < array_size; ++i) {
                ((float*)A->data)[i] *= ((float*)B->data)[i];
            }
            break;
        case NDA_COMPLEX64:
            // Element-wise multiplication for complex64
            OMP_PRAGMA(omp parallel for)
            for (size_t i = 0; i < array_size; ++i) {
                ((double complex*)A->data)[i] *= ((double complex*)B->data)[i];
            }
            break;
        case NDA_COMPLEX32:
            // Element-wise multiplication for complex32
            OMP_PRAGMA(omp parallel for)
            for (size_t i = 0; i < array_size; ++i) {
                ((float complex*)A->data)[i] *= ((float complex*)B->data)[i];
            }
            break;
        default:
            assert(0 && "Unsupported data type");
            return A;
    }
    
    return A;
}

NDArray ndarray_mul_scalar(const NDArray A, double scalar) {
    assert((A != NULL) && "ndarray cannot be NULL");
    assert((A->ndim >= 2) && "ndarray must have at least 2 dimensions");
    
    size_t array_size = ndarray_size(A);
    
    switch (A->dtype) {
        case NDA_REAL64:
            cblas_dscal(array_size, scalar, A->data, 1);
            break;
        case NDA_REAL32:
            cblas_sscal(array_size, scalar, A->data, 1);
            break;
        case NDA_COMPLEX64:
            {
                double complex alpha = scalar + 0.0 * I;
                cblas_zscal(array_size, &alpha, A->data, 1);
            }
            break;
        case NDA_COMPLEX32:
            {
                float complex alpha = (float)scalar + 0.0f * I;
                cblas_cscal(array_size, &alpha, A->data, 1);
            }
            break;
        default:
            assert(0 && "Unsupported data type");
            return A;
    }
    
    return A;
}