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

NDArray ndarray_axpby(NDArray A, double alpha, NDArray B, double beta) {
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

NDArray ndarray_scale_shift(NDArray A, double alpha, double beta) {
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

NDArray ndarray_mul_scaled(NDArray A, NDArray B, double scalar) {
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

NDArray ndarray_map_mul(NDArray A, double (*func)(double), 
                        NDArray B, double alpha) {
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

NDArray ndarray_mul_add(NDArray A, NDArray B, NDArray C, 
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

void ndarray_gemv(NDArray A, NDArray x, double alpha, 
                  double beta, NDArray y) {
    assert(A != NULL && x != NULL && y != NULL 
           && "ndarrays cannot be NULL");
    assert(A->ndim == 2 && "A must be a 2D matrix");
    assert(x->ndim == 2 && "x must be a 2D vector");
    assert(y->ndim == 2 && "y must be a 2D vector");
    
    // Check if x is a column vector [n, 1] or row vector [1, n]
    int x_is_col = (x->dims[1] == 1);
    int x_is_row = (x->dims[0] == 1);
    assert((x_is_col || x_is_row) && "x must be a vector (one dimension = 1)");
    
    // Check if y is a column vector [m, 1] or row vector [1, m]
    int y_is_col = (y->dims[1] == 1);
    int y_is_row = (y->dims[0] == 1);
    assert((y_is_col || y_is_row) && "y must be a vector (one dimension = 1)");
    
    size_t m = A->dims[0];  // rows of A
    size_t n = A->dims[1];  // cols of A
    
    // Check dimensions compatibility
    size_t x_len = x_is_col ? x->dims[0] : x->dims[1];
    size_t y_len = y_is_col ? y->dims[0] : y->dims[1];
    
    assert(x_len == n && "x length must match number of columns in A");
    assert(y_len == m && "y length must match number of rows in A");
    
    // y = alpha * A * x + beta * y
    // Using CBLAS dgemv for optimized matrix-vector multiplication
    cblas_dgemv(CblasRowMajor, CblasNoTrans, 
                m, n, alpha, A->data, n, 
                x->data, 1, beta, y->data, 1);
}

NDArray ndarray_clip_min(NDArray A, double min_val) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    
    size_t size = ndarray_size(A);
    
    // Clip values below min_val
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        if (A->data[i] < min_val) {
            A->data[i] = min_val;
        }
    }
    
    return A;
}

NDArray ndarray_clip_max(NDArray A, double max_val) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    
    size_t size = ndarray_size(A);
    
    // Clip values above max_val
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        if (A->data[i] > max_val) {
            A->data[i] = max_val;
        }
    }
    
    return A;
}

NDArray ndarray_clip(NDArray A, double min_val, double max_val) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    assert(min_val <= max_val && "min_val must be <= max_val");
    
    size_t size = ndarray_size(A);
    
    // Clip values to [min_val, max_val] range
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

NDArray ndarray_abs(NDArray A) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    
    size_t size = ndarray_size(A);
    
    // Compute absolute value
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        A->data[i] = fabs(A->data[i]);
    }
    
    return A;
}

NDArray ndarray_sign(NDArray A) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    
    size_t size = ndarray_size(A);
    
    // Compute sign: -1, 0, or 1
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


