/**
 * Comparison and logical operations on ndarrays
 */

#include "ndarray_internal.h"

NDArray ndarray_new_equal(NDArray A, NDArray B) {
    assert(A != NULL && B != NULL && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    assert(A->ndim == B->ndim
            && "ndarrays must have same number of dimensions");
    for (size_t i = 0; i < A->ndim; ++i) {
        assert(A->dims[i] == B->dims[i]
                && "ndarrays must have matching dimensions");
    }
    
    size_t result_dims[A->ndim + 1];
    for (size_t i = 0; i < A->ndim; ++i) {
        result_dims[i] = A->dims[i];
    }
    result_dims[A->ndim] = 0;
    
    NDArray result = ndarray_new(result_dims);
    size_t size = ndarray_size(A);
    double *result_data = (double*)result->data;
    
    if (A->dtype == NDA_REAL64) {
        const double *A_data = (const double*)A->data;
        const double *B_data = (const double*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] == B_data[i]) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_REAL32) {
        const float *A_data = (const float*)A->data;
        const float *B_data = (const float*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] == B_data[i]) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX64) {
        const double complex *A_data = (const double complex*)A->data;
        const double complex *B_data = (const double complex*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] == B_data[i]) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX32) {
        const float complex *A_data = (const float complex*)A->data;
        const float complex *B_data = (const float complex*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] == B_data[i]) ? 1.0 : 0.0;
        }
    }
    
    return result;
}

NDArray ndarray_new_less(NDArray A, NDArray B) {
    assert(A != NULL && B != NULL && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    assert(A->ndim == B->ndim
            && "ndarrays must have same number of dimensions");
    for (size_t i = 0; i < A->ndim; ++i) {
        assert(A->dims[i] == B->dims[i]
                && "ndarrays must have matching dimensions");
    }
    
    size_t result_dims[A->ndim + 1];
    for (size_t i = 0; i < A->ndim; ++i) {
        result_dims[i] = A->dims[i];
    }
    result_dims[A->ndim] = 0;
    
    NDArray result = ndarray_new(result_dims);
    size_t size = ndarray_size(A);
    double *result_data = (double*)result->data;
    
    if (A->dtype == NDA_REAL64) {
        const double *A_data = (const double*)A->data;
        const double *B_data = (const double*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] < B_data[i]) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_REAL32) {
        const float *A_data = (const float*)A->data;
        const float *B_data = (const float*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] < B_data[i]) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX64) {
        const double complex *A_data = (const double complex*)A->data;
        const double complex *B_data = (const double complex*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (cabs(A_data[i]) < cabs(B_data[i])) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX32) {
        const float complex *A_data = (const float complex*)A->data;
        const float complex *B_data = (const float complex*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (cabsf(A_data[i]) < cabsf(B_data[i])) ? 1.0 : 0.0;
        }
    }
    
    return result;
}

NDArray ndarray_new_greater(NDArray A, NDArray B) {
    assert(A != NULL && B != NULL && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    assert(A->ndim == B->ndim
            && "ndarrays must have same number of dimensions");
    for (size_t i = 0; i < A->ndim; ++i) {
        assert(A->dims[i] == B->dims[i]
                && "ndarrays must have matching dimensions");
    }
    
    size_t result_dims[A->ndim + 1];
    for (size_t i = 0; i < A->ndim; ++i) {
        result_dims[i] = A->dims[i];
    }
    result_dims[A->ndim] = 0;
    
    NDArray result = ndarray_new(result_dims);
    size_t size = ndarray_size(A);
    double *result_data = (double*)result->data;
    
    if (A->dtype == NDA_REAL64) {
        const double *A_data = (const double*)A->data;
        const double *B_data = (const double*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] > B_data[i]) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_REAL32) {
        const float *A_data = (const float*)A->data;
        const float *B_data = (const float*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] > B_data[i]) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX64) {
        const double complex *A_data = (const double complex*)A->data;
        const double complex *B_data = (const double complex*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (cabs(A_data[i]) > cabs(B_data[i])) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX32) {
        const float complex *A_data = (const float complex*)A->data;
        const float complex *B_data = (const float complex*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (cabsf(A_data[i]) > cabsf(B_data[i])) ? 1.0 : 0.0;
        }
    }
    
    return result;
}

NDArray ndarray_new_equal_scalar(NDArray A, double value) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    
    size_t result_dims[A->ndim + 1];
    for (size_t i = 0; i < A->ndim; ++i) {
        result_dims[i] = A->dims[i];
    }
    result_dims[A->ndim] = 0;
    
    NDArray result = ndarray_new(result_dims);
    size_t size = ndarray_size(A);
    double *result_data = (double*)result->data;
    
    if (A->dtype == NDA_REAL64) {
        const double *A_data = (const double*)A->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] == value) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_REAL32) {
        const float *A_data = (const float*)A->data;
        float fvalue = (float)value;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] == fvalue) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX64) {
        const double complex *A_data = (const double complex*)A->data;
        double complex cvalue = value + 0.0 * I;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] == cvalue) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX32) {
        const float complex *A_data = (const float complex*)A->data;
        float complex cvalue = (float)value + 0.0f * I;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] == cvalue) ? 1.0 : 0.0;
        }
    }
    
    return result;
}

NDArray ndarray_new_less_scalar(NDArray A, double value) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    
    size_t result_dims[A->ndim + 1];
    for (size_t i = 0; i < A->ndim; ++i) {
        result_dims[i] = A->dims[i];
    }
    result_dims[A->ndim] = 0;
    
    NDArray result = ndarray_new(result_dims);
    size_t size = ndarray_size(A);
    double *result_data = (double*)result->data;
    
    if (A->dtype == NDA_REAL64) {
        const double *A_data = (const double*)A->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] < value) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_REAL32) {
        const float *A_data = (const float*)A->data;
        float fvalue = (float)value;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] < fvalue) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX64) {
        const double complex *A_data = (const double complex*)A->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (cabs(A_data[i]) < value) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX32) {
        const float complex *A_data = (const float complex*)A->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (cabsf(A_data[i]) < (float)value) ? 1.0 : 0.0;
        }
    }
    
    return result;
}

NDArray ndarray_new_greater_scalar(NDArray A, double value) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    
    size_t result_dims[A->ndim + 1];
    for (size_t i = 0; i < A->ndim; ++i) {
        result_dims[i] = A->dims[i];
    }
    result_dims[A->ndim] = 0;
    
    NDArray result = ndarray_new(result_dims);
    size_t size = ndarray_size(A);
    double *result_data = (double*)result->data;
    
    if (A->dtype == NDA_REAL64) {
        const double *A_data = (const double*)A->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] > value) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_REAL32) {
        const float *A_data = (const float*)A->data;
        float fvalue = (float)value;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] > fvalue) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX64) {
        const double complex *A_data = (const double complex*)A->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (cabs(A_data[i]) > value) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX32) {
        const float complex *A_data = (const float complex*)A->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (cabsf(A_data[i]) > (float)value) ? 1.0 : 0.0;
        }
    }
    
    return result;
}

NDArray ndarray_new_logical_and(NDArray A, NDArray B) {
    assert(A != NULL && B != NULL && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    assert(A->ndim == B->ndim
            && "ndarrays must have same number of dimensions");
    for (size_t i = 0; i < A->ndim; ++i) {
        assert(A->dims[i] == B->dims[i]
                && "ndarrays must have matching dimensions");
    }
    
    size_t result_dims[A->ndim + 1];
    for (size_t i = 0; i < A->ndim; ++i) {
        result_dims[i] = A->dims[i];
    }
    result_dims[A->ndim] = 0;
    
    NDArray result = ndarray_new(result_dims);
    size_t size = ndarray_size(A);
    double *result_data = (double*)result->data;
    
    if (A->dtype == NDA_REAL64) {
        const double *A_data = (const double*)A->data;
        const double *B_data = (const double*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = ((A_data[i] != 0.0) && (B_data[i] != 0.0)) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_REAL32) {
        const float *A_data = (const float*)A->data;
        const float *B_data = (const float*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = ((A_data[i] != 0.0f) && (B_data[i] != 0.0f)) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX64) {
        const double complex *A_data = (const double complex*)A->data;
        const double complex *B_data = (const double complex*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = ((A_data[i] != 0.0) && (B_data[i] != 0.0)) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX32) {
        const float complex *A_data = (const float complex*)A->data;
        const float complex *B_data = (const float complex*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = ((A_data[i] != 0.0f) && (B_data[i] != 0.0f)) ? 1.0 : 0.0;
        }
    }
    
    return result;
}

NDArray ndarray_new_logical_or(NDArray A, NDArray B) {
    assert(A != NULL && B != NULL && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    assert(A->ndim == B->ndim
            && "ndarrays must have same number of dimensions");
    for (size_t i = 0; i < A->ndim; ++i) {
        assert(A->dims[i] == B->dims[i]
                && "ndarrays must have matching dimensions");
    }
    size_t result_dims[A->ndim + 1];
    for (size_t i = 0; i < A->ndim; ++i) {
        result_dims[i] = A->dims[i];
    }
    result_dims[A->ndim] = 0;
    NDArray result = ndarray_new(result_dims);
    size_t size = ndarray_size(A);
    double *result_data = (double*)result->data;
    
    if (A->dtype == NDA_REAL64) {
        const double *A_data = (const double*)A->data;
        const double *B_data = (const double*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = ((A_data[i] != 0.0) || (B_data[i] != 0.0)) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_REAL32) {
        const float *A_data = (const float*)A->data;
        const float *B_data = (const float*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = ((A_data[i] != 0.0f) || (B_data[i] != 0.0f)) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX64) {
        const double complex *A_data = (const double complex*)A->data;
        const double complex *B_data = (const double complex*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = ((A_data[i] != 0.0) || (B_data[i] != 0.0)) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX32) {
        const float complex *A_data = (const float complex*)A->data;
        const float complex *B_data = (const float complex*)B->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = ((A_data[i] != 0.0f) || (B_data[i] != 0.0f)) ? 1.0 : 0.0;
        }
    }
    
    return result;
}

NDArray ndarray_new_logical_not(NDArray A) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    size_t result_dims[A->ndim + 1];
    for (size_t i = 0; i < A->ndim; ++i) {
        result_dims[i] = A->dims[i];
    }
    result_dims[A->ndim] = 0;
    NDArray result = ndarray_new(result_dims);
    size_t size = ndarray_size(A);
    double *result_data = (double*)result->data;
    
    if (A->dtype == NDA_REAL64) {
        const double *A_data = (const double*)A->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] == 0.0) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_REAL32) {
        const float *A_data = (const float*)A->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] == 0.0f) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX64) {
        const double complex *A_data = (const double complex*)A->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] == 0.0) ? 1.0 : 0.0;
        }
    } else if (A->dtype == NDA_COMPLEX32) {
        const float complex *A_data = (const float complex*)A->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = (A_data[i] == 0.0f) ? 1.0 : 0.0;
        }
    }
    
    return result;
}

NDArray ndarray_new_where(NDArray condition, NDArray x, NDArray y) {
    assert(condition != NULL && x != NULL && y != NULL
           && "ndarrays cannot be NULL");
    assert(condition->ndim >= 2 && x->ndim >= 2 && y->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    assert(condition->ndim == x->ndim && x->ndim == y->ndim
            && "ndarrays must have same number of dimensions");
    for (size_t i = 0; i < condition->ndim; ++i) {
        assert(condition->dims[i] == x->dims[i] && x->dims[i] == y->dims[i]
                && "ndarrays must have matching dimensions");
    }
    size_t result_dims[x->ndim + 1];
    for (size_t i = 0; i < x->ndim; ++i) {
        result_dims[i] = x->dims[i];
    }
    result_dims[x->ndim] = 0;
    NDArray result = ndarray_new(result_dims);
    size_t size = ndarray_size(x);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        result->data[i] = (condition->data[i] != 0.0) ? x->data[i] : y->data[i];
    }
    return result;
}
