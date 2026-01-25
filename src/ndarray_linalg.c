/**
 * Linear algebra operations on ndarrays
 */

#include "ndarray_internal.h"

NDArray ndarray_new_matmul_typed(const NDArray A, const NDArray B, NDAType result_dtype) {
    assert((A != NULL && B != NULL) && "ndarrays cannot be NULL");
    assert((A->ndim >= 2 && B->ndim >= 2) && "ndarrays must have at least 2 dimensions");
    assert((A->ndim == 2 && B->ndim == 2) && "matrix multiplication requires exactly 2D arrays");
    
    // Check matrix dimensions compatibility
    size_t m = A->dims[0];
    size_t k = A->dims[1];
    size_t n = B->dims[1];
    assert((k == B->dims[0]) && "inner matrix dimensions must agree");
    
    // Phase 4: Type-aware matrix multiplication
    // Determine result type through promotion
    NDAType result_type = (result_dtype != 0xFF) ? 
                          result_dtype : 
                          ndarray_promote_types(A->dtype, B->dtype);
    
    // Create result array
    NDArray C = ndarray_new_typed(NDA_DIMS(m, n), result_type);
    
    // Type-aware matrix multiplication dispatch
    switch (result_type) {
        case NDA_REAL64: {
            // REAL64 × REAL64 → REAL64: use DGEMM
            const double alpha = 1.0, beta = 0.0;
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       (int)m, (int)n, (int)k,
                       alpha,
                       (const double*)A->data, (int)A->dims[1],
                       (const double*)B->data, (int)B->dims[1],
                       beta,
                       (double*)C->data, (int)C->dims[1]);
            break;
        }
        
        case NDA_REAL32: {
            // REAL32 × REAL32 → REAL32: use SGEMM
            const float alpha = 1.0f, beta = 0.0f;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       (int)m, (int)n, (int)k,
                       alpha,
                       (const float*)A->data, (int)A->dims[1],
                       (const float*)B->data, (int)B->dims[1],
                       beta,
                       (float*)C->data, (int)C->dims[1]);
            break;
        }
        
        case NDA_COMPLEX64: {
            // COMPLEX64 × COMPLEX64 → COMPLEX64: use ZGEMM
            const double alpha[2] = {1.0, 0.0};
            const double beta[2] = {0.0, 0.0};
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       (int)m, (int)n, (int)k,
                       alpha,
                       (const void*)A->data, (int)A->dims[1],
                       (const void*)B->data, (int)B->dims[1],
                       beta,
                       (void*)C->data, (int)C->dims[1]);
            break;
        }
        
        case NDA_COMPLEX32: {
            // COMPLEX32 × COMPLEX32 → COMPLEX32: use CGEMM
            const float alpha[2] = {1.0f, 0.0f};
            const float beta[2] = {0.0f, 0.0f};
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       (int)m, (int)n, (int)k,
                       alpha,
                       (const void*)A->data, (int)A->dims[1],
                       (const void*)B->data, (int)B->dims[1],
                       beta,
                       (void*)C->data, (int)C->dims[1]);
            break;
        }
        
        default:
            fprintf(stderr, "Unsupported type for matrix multiplication\n");
            ndarray_free(C);
            return NULL;
    }
    
    return C;
}