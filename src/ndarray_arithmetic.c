/**
 * Arithmetic operations on ndarrays
 */

#include "ndarray_internal.h"

// Forward declarations for BLAS dispatch helpers
void ndarray_blas_axpy(size_t n, double alpha, const NDArray X, const NDArray Y, NDArray result);
void ndarray_blas_scal(size_t n, double alpha, const NDArray X);

NDArray ndarray_add(const NDArray A, const NDArray B) {
    assert(A != NULL && B != NULL && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    assert(A->ndim == B->ndim
            && "ndarrays must have same number of dimensions");
    for (size_t i = 0; i < A->ndim; ++i) {
        assert(A->dims[i] == B->dims[i]
                && "ndarrays must have matching dimensions");
    }
    
    // Determine result type through promotion
    NDAType result_type = ndarray_promote_types(A->dtype, B->dtype);
    size_t size = ndarray_size(A);
    
    // Type-aware addition using proper BLAS dispatch
    switch (result_type) {
        case NDA_REAL64:
            cblas_daxpy(size, 1.0, ((double*)B->data), 1, ((double*)A->data), 1);
            break;
        case NDA_REAL32:
            // Convert to REAL64, perform operation, convert back
            {
                double* temp_A = malloc(size * sizeof(double));
                double* temp_B = malloc(size * sizeof(double));
                for (size_t i = 0; i < size; i++) {
                    temp_A[i] = (double)((float*)A->data)[i];
                    temp_B[i] = (double)((float*)B->data)[i];
                }
                cblas_daxpy(size, 1.0, temp_B, 1, temp_A, 1);
                // Convert result back to REAL32
                for (size_t i = 0; i < size; i++) {
                    ((float*)A->data)[i] = (float)temp_A[i];
                }
                free(temp_A);
                free(temp_B);
            }
            break;
        case NDA_COMPLEX64:
            cblas_zaxpy(size, 1.0, (double complex*)B->data, 1, (double complex*)A->data, 1);
            break;
        case NDA_COMPLEX32:
            // Convert to COMPLEX64, perform operation, convert back
            {
                double complex* temp_A = malloc(size * sizeof(double complex));
                double complex* temp_B = malloc(size * sizeof(double complex));
                for (size_t i = 0; i < size; i++) {
                    temp_A[i] = (double complex)((float complex*)A->data)[i];
                    temp_B[i] = (double complex)((float complex*)B->data)[i];
                }
                cblas_zaxpy(size, 1.0, temp_B, 1, temp_A, 1);
                // Convert result back to COMPLEX32
                for (size_t i = 0; i < size; i++) {
                    ((float complex*)A->data)[i] = (float complex)temp_A[i];
                }
                free(temp_A);
                free(temp_B);
            }
            break;
        default:
            // Fallback: convert to REAL64
            {
                double* temp_A = malloc(size * sizeof(double));
                double* temp_B = malloc(size * sizeof(double));
                for (size_t i = 0; i < size; i++) {
                    temp_A[i] = ndarray_get(A, &(size_t){i, 0});
                    temp_B[i] = ndarray_get(B, &(size_t){i, 0});
                }
                cblas_daxpy(size, 1.0, temp_B, 1, temp_A, 1);
                // Convert result back to original type
                for (size_t i = 0; i < size; i++) {
                    ndarray_set(A, &(size_t){i, 0}, temp_A[i]);
                }
                free(temp_A);
                free(temp_B);
            }
            break;
    }
    return A;
}

NDArray ndarray_mul(const NDArray A, const NDArray B) {
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

NDArray ndarray_add_scalar(const NDArray A, double scalar) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    size_t size = ndarray_size(A);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        A->data[i] += scalar;
    }
    return A;
}

NDArray ndarray_mul_scalar(const NDArray A, double scalar) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    size_t size = ndarray_size(A);
    cblas_dscal(size, scalar, A->data, 1);
    return A;
}

NDArray ndarray_axpby(const NDArray A, double alpha, const NDArray B,
        double beta) {
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
    // Compute A = alpha*A + beta*B using BLAS routines
    // First: A = alpha*A (scale A in place)
    cblas_dscal(size, alpha, A->data, 1);
    // Second: A = 1.0*beta*B + A (add scaled B to A)
    cblas_daxpy(size, beta, B->data, 1, A->data, 1);
    return A;
}

NDArray ndarray_scale_shift(const NDArray A, double alpha, double beta) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    size_t size = ndarray_size(A);
    // First: A = alpha*A (scale)
    cblas_dscal(size, alpha, A->data, 1);
    // Second: A = A + beta (shift)
    if (beta != 0.0) {
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            A->data[i] += beta;
        }
    }
    return A;
}

NDArray ndarray_mul_scaled(const NDArray A, const NDArray B, double scalar) {
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
    // A = A * B * scalar (element-wise)
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        A->data[i] *= B->data[i] * scalar;
    }
    return A;
}

NDArray ndarray_map_mul(const NDArray A, double (*func)(double), 
                        const NDArray B, double alpha) {
    assert(A != NULL && B != NULL && "ndarrays cannot be NULL");
    assert(func != NULL && "function pointer cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    assert(A->ndim == B->ndim
            && "ndarrays must have same number of dimensions");
    for (size_t i = 0; i < A->ndim; ++i) {
        assert(A->dims[i] == B->dims[i]
                && "ndarrays must have matching dimensions");
    }
    size_t size = ndarray_size(A);
    // A = func(A) * B * alpha (element-wise)
    OMP_PRAGMA(omp parallel for)
    for (size_t i = 0; i < size; ++i) {
        A->data[i] = func(A->data[i]) * B->data[i] * alpha;
    }
    return A;
}

NDArray ndarray_mul_add(const NDArray A, const NDArray B, const NDArray C, 
                        double alpha, double beta) {
    assert(A != NULL && B != NULL && C != NULL 
           && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2 && C->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    assert(A->ndim == B->ndim && A->ndim == C->ndim
            && "ndarrays must have same number of dimensions");
    for (size_t i = 0; i < A->ndim; ++i) {
        assert(A->dims[i] == B->dims[i] && A->dims[i] == C->dims[i]
                && "ndarrays must have matching dimensions");
    }
    size_t size = ndarray_size(A);
    // C = alpha * (A * B) + beta * C (element-wise)
    if (beta == 0.0) {
        // Special case: just compute alpha * A * B
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            C->data[i] = alpha * A->data[i] * B->data[i];
        }
    } else if (beta == 1.0) {
        // Special case: C = alpha * A * B + C
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            C->data[i] += alpha * A->data[i] * B->data[i];
        }
    } else {
        // General case
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            C->data[i] = alpha * A->data[i] * B->data[i] + beta * C->data[i];
        }
    }
    return C;
}

/*
NDArray ndarray_matvec_mul(const NDArray y, const NDArray A, const NDArray x,
                           double alpha, double beta) {
    assert(y != NULL && A != NULL && x != NULL 
           && "ndarrays cannot be NULL");
    assert(A->ndim == 2 && "A must be a 2D matrix");
    assert(x->ndim == 2 && "x must be a 2D vector");
    assert(y->ndim == 2 && "y must be a 2D vector");
    int x_is_col = (x->dims[1] == 1);
    int x_is_row = (x->dims[0] == 1);
    assert((x_is_col || x_is_row) && "x must be a vector (one dimension = 1)");
    int y_is_col = (y->dims[1] == 1);
    int y_is_row = (y->dims[0] == 1);
    assert((y_is_col || y_is_row) && "y must be a vector (one dimension = 1)");
    size_t m = A->dims[0];  // rows of A
    size_t n = A->dims[1];  // cols of A
    size_t x_len = x_is_col ? x->dims[0] : x->dims[1];
    size_t y_len = y_is_col ? y->dims[0] : y->dims[1];
    assert(x_len == n && "x length must match number of columns in A");
    assert(y_len == m && "y length must match number of rows in A");
    // y = alpha * A * x + beta * y
    cblas_dgemv(CblasRowMajor, CblasNoTrans, 
                m, n, alpha, A->data, n, 
                x->data, 1, beta, y->data, 1);
    return y;
}
*/

NDArray ndarray_clip_min(const NDArray A, double min_val) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    size_t size = ndarray_size(A);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        if (A->data[i] < min_val) {
            A->data[i] = min_val;
        }
    }
    return A;
}

NDArray ndarray_clip_max(const NDArray A, double max_val) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    size_t size = ndarray_size(A);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        if (A->data[i] > max_val) {
            A->data[i] = max_val;
        }
    }
    return A;
}

NDArray ndarray_clip(const NDArray A, double min_val, double max_val) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    assert(min_val <= max_val && "min_val must be <= max_val");
    size_t size = ndarray_size(A);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        if (A->data[i] < min_val) {
            A->data[i] = min_val;
        } else if (A->data[i] > max_val) {
            A->data[i] = max_val;
        }
    }
    return A;
}

NDArray ndarray_abs(const NDArray A) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    size_t size = ndarray_size(A);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        A->data[i] = fabs(A->data[i]);
    }
    return A;
}

NDArray ndarray_sign(const NDArray A) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    size_t size = ndarray_size(A);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        if (A->data[i] > 0.0) {
            A->data[i] = 1.0;
        } else if (A->data[i] < 0.0) {
            A->data[i] = -1.0;
        } else {
            A->data[i] = 0.0;
        }
    }
    return A;
}
