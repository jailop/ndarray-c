/**
 * Linear algebra operations
 */

#include "ndarray_internal.h"

static void matmul_2d_blocked(double *C, double *A, double *B, 
                              size_t m, size_t n, size_t p) {
    // Use CBLAS dgemm for matrix multiplication: C = alpha*A*B + beta*C
    // CblasRowMajor: row-major order
    // CblasNoTrans: no transpose
    // m: number of rows of A and C
    // p: number of columns of B and C
    // n: number of columns of A and rows of B
    // alpha=1.0, beta=0.0 for C = A*B
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, p, n, 1.0, A, n, B, p, 0.0, C, p);
}

NDArray ndarray_new_tensordot(NDArray A, NDArray B, 
                               int *axes_a, int *axes_b) {
    assert(A != NULL && B != NULL && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    // Count axes (terminated by -1)
    int n_axes = 0;
    if (axes_a != NULL && axes_b != NULL) {
        while (axes_a[n_axes] != -1) {
            assert(axes_b[n_axes] != -1
                    && "axes_a and axes_b must have same length");
            n_axes++;
        }
        assert(axes_b[n_axes] == -1
                && "axes_a and axes_b must have same length");
    }
    // Validate axes and check dimension compatibility
    for (int i = 0; i < n_axes; ++i) {
        assert(axes_a[i] >= 0 && axes_a[i] < (int)A->ndim && 
               "axes_a contains invalid axis");
        assert(axes_b[i] >= 0 && axes_b[i] < (int)B->ndim && 
               "axes_b contains invalid axis");
        assert(A->dims[axes_a[i]] == B->dims[axes_b[i]] && 
               "contracted axes must have matching dimensions");
    }
    // Mark which axes are being contracted
    int *a_contracted = (int*)calloc(A->ndim, sizeof(int));
    int *b_contracted = (int*)calloc(B->ndim, sizeof(int));
    for (int i = 0; i < n_axes; ++i) {
        a_contracted[axes_a[i]] = 1;
        b_contracted[axes_b[i]] = 1;
    }
    // Build result dimensions: free axes from A, then free axes from B
    size_t result_ndim = (A->ndim - n_axes) + (B->ndim - n_axes);
    if (result_ndim < 2) result_ndim = 2;
    size_t *result_dims = (size_t*)malloc(sizeof(size_t) * (result_ndim + 1));
    size_t idx = 0;
    // Add free axes from A
    for (size_t i = 0; i < A->ndim; ++i) {
        if (!a_contracted[i]) {
            result_dims[idx++] = A->dims[i];
        }
    }
    // Add free axes from B
    for (size_t i = 0; i < B->ndim; ++i) {
        if (!b_contracted[i]) {
            result_dims[idx++] = B->dims[i];
        }
    }
    // Pad to ensure ndim >= 2
    while (idx < result_ndim) {
        result_dims[idx++] = 1;
    }
    result_dims[idx] = 0;
    NDArray C = ndarray_new_zeros(result_dims);
    // Compute sizes for iteration
    size_t contract_size = 1;
    for (int i = 0; i < n_axes; ++i) {
        contract_size *= A->dims[axes_a[i]];
    }
    size_t a_free_size = 1;
    for (size_t i = 0; i < A->ndim; ++i) {
        if (!a_contracted[i]) a_free_size *= A->dims[i];
    }
    size_t b_free_size = 1;
    for (size_t i = 0; i < B->ndim; ++i) {
        if (!b_contracted[i]) b_free_size *= B->dims[i];
    }
    // Compute strides for A, B, and C
    size_t *a_strides = (size_t*)malloc(sizeof(size_t) * A->ndim);
    size_t *b_strides = (size_t*)malloc(sizeof(size_t) * B->ndim);
    size_t *c_strides = (size_t*)malloc(sizeof(size_t) * C->ndim);
    a_strides[A->ndim - 1] = 1;
    for (int i = A->ndim - 2; i >= 0; --i) {
        a_strides[i] = a_strides[i + 1] * A->dims[i + 1];
    }
    b_strides[B->ndim - 1] = 1;
    for (int i = B->ndim - 2; i >= 0; --i) {
        b_strides[i] = b_strides[i + 1] * B->dims[i + 1];
    }
    c_strides[C->ndim - 1] = 1;
    for (int i = C->ndim - 2; i >= 0; --i) {
        c_strides[i] = c_strides[i + 1] * C->dims[i + 1];
    }
    // Perform contraction
    size_t result_size = a_free_size * b_free_size;
    OMP_PRAGMA(omp parallel)
    {
        size_t *a_idx = (size_t*)malloc(sizeof(size_t) * A->ndim);
        size_t *b_idx = (size_t*)malloc(sizeof(size_t) * B->ndim);
        size_t *c_idx = (size_t*)malloc(sizeof(size_t) * C->ndim);
        OMP_PRAGMA(omp for)
        for (size_t out_idx = 0; out_idx < result_size; ++out_idx) {
            // Decode output index
            size_t temp = out_idx;
            idx = 0;
            // Get indices for free axes from A
            for (size_t i = 0; i < A->ndim; ++i) {
                if (!a_contracted[i]) {
                    size_t dim = A->dims[i];
                    size_t stride = 1;
                    for (size_t j = i + 1; j < A->ndim; ++j) {
                        if (!a_contracted[j]) stride *= A->dims[j];
                    }
                    for (size_t j = 0; j < B->ndim; ++j) {
                        if (!b_contracted[j]) stride *= B->dims[j];
                    }
                    a_idx[i] = (temp / stride) % dim;
                }
            }
            // Get indices for free axes from B
            for (size_t i = 0; i < B->ndim; ++i) {
                if (!b_contracted[i]) {
                    size_t dim = B->dims[i];
                    size_t stride = 1;
                    for (size_t j = i + 1; j < B->ndim; ++j) {
                        if (!b_contracted[j]) stride *= B->dims[j];
                    }
                    b_idx[i] = (temp / stride) % dim;
                }
            }
            // Perform contraction sum
            double sum = 0.0;
            for (size_t contract_idx = 0; contract_idx < contract_size;
                    ++contract_idx) {
                // Decode contraction indices
                size_t ctemp = contract_idx;
                for (int i = n_axes - 1; i >= 0; --i) {
                    size_t dim = A->dims[axes_a[i]];
                    a_idx[axes_a[i]] = ctemp % dim;
                    b_idx[axes_b[i]] = ctemp % dim;
                    ctemp /= dim;
                }
                // Compute flat indices
                size_t a_flat = 0;
                for (size_t i = 0; i < A->ndim; ++i) {
                    a_flat += a_idx[i] * a_strides[i];
                }
                size_t b_flat = 0;
                for (size_t i = 0; i < B->ndim; ++i) {
                    b_flat += b_idx[i] * b_strides[i];
                }
                sum += A->data[a_flat] * B->data[b_flat];
            }
            C->data[out_idx] = sum;
        }
        free(a_idx);
        free(b_idx);
        free(c_idx);
    }
    free(a_contracted);
    free(b_contracted);
    free(result_dims);
    free(a_strides);
    free(b_strides);
    free(c_strides);
    return C;
}

NDArray ndarray_new_matmul(NDArray A, NDArray B) {
    assert(A != NULL && B != NULL && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    size_t a_rows = A->dims[A->ndim - 2];
    size_t a_cols = A->dims[A->ndim - 1];
    size_t b_rows = B->dims[B->ndim - 2];
    size_t b_cols = B->dims[B->ndim - 1];
    assert(a_cols == b_rows
            && "matrix dimensions incompatible for multiplication");
    // Simple case: both 2D matrices
    if (A->ndim == 2 && B->ndim == 2) {
        size_t dims[] = {a_rows, b_cols, 0};
        NDArray C = ndarray_new_zeros(dims);
        matmul_2d_blocked(C->data, A->data, B->data, a_rows, a_cols, b_cols);
        return C;
    }
    // Batch matmul: determine result shape
    size_t max_ndim = (A->ndim > B->ndim) ? A->ndim : B->ndim;
    size_t *result_dims = (size_t*)malloc(sizeof(size_t) * (max_ndim + 1));
    // Broadcast batch dimensions
    for (size_t i = 0; i < max_ndim - 2; ++i) {
        size_t a_idx = (A->ndim - 2 > i) ? A->ndim - 2 - i - 1 : 0;
        size_t b_idx = (B->ndim - 2 > i) ? B->ndim - 2 - i - 1 : 0;
        size_t a_dim = (a_idx < A->ndim - 2) ? A->dims[a_idx] : 1;
        size_t b_dim = (b_idx < B->ndim - 2) ? B->dims[b_idx] : 1;
        assert((a_dim == b_dim || a_dim == 1 || b_dim == 1) && 
               "batch dimensions must be compatible for broadcasting");
        result_dims[max_ndim - 2 - i - 1] = (a_dim > b_dim) ? a_dim : b_dim;
    }
    result_dims[max_ndim - 2] = a_rows;
    result_dims[max_ndim - 1] = b_cols;
    result_dims[max_ndim] = 0;
    NDArray C = ndarray_new_zeros(result_dims);
    free(result_dims);
    // Calculate batch sizes
    size_t batch_size = 1;
    for (size_t i = 0; i < C->ndim - 2; ++i) {
        batch_size *= C->dims[i];
    }
    size_t a_batch_stride = a_rows * a_cols;
    size_t b_batch_stride = b_rows * b_cols;
    size_t c_batch_stride = a_rows * b_cols;
    size_t a_batch_size = 1;
    for (size_t i = 0; i < A->ndim - 2; ++i) {
        a_batch_size *= A->dims[i];
    }
    size_t b_batch_size = 1;
    for (size_t i = 0; i < B->ndim - 2; ++i) {
        b_batch_size *= B->dims[i];
    }
    OMP_PRAGMA(omp parallel for)
    for (size_t batch = 0; batch < batch_size; ++batch) {
        size_t a_batch_idx = (a_batch_size == 1) ? 0 : (batch % a_batch_size);
        size_t b_batch_idx = (b_batch_size == 1) ? 0 : (batch % b_batch_size);
        double *a_ptr = A->data + a_batch_idx * a_batch_stride;
        double *b_ptr = B->data + b_batch_idx * b_batch_stride;
        double *c_ptr = C->data + batch * c_batch_stride;
        matmul_2d_blocked(c_ptr, a_ptr, b_ptr, a_rows, a_cols, b_cols);
    }
    
    return C;
}
