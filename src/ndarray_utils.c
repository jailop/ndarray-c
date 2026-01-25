/**
 * Utility functions for complex number operations and type conversions
 */

#include "ndarray_internal.h"

// Real and imaginary part extraction

NDArray ndarray_new_real_part(const NDArray arr) {
    if (!ndarray_is_complex_type(arr->dtype)) {
        fprintf(stderr, "Error: Input array must be complex type\n");
        return NULL;
    }
    
    NDArray result = ndarray_new_typed(arr->dims, 
                                   (arr->dtype == NDA_COMPLEX64) ? NDA_REAL64 : NDA_REAL32);
    size_t size = ndarray_size(arr);
    
    if (arr->dtype == NDA_COMPLEX64) {
        const double complex *src = (const double complex*)arr->data;
        double *dst = (double*)result->data;
        
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            dst[i] = creal(src[i]);
        }
    } else { // NDA_COMPLEX32
        const float complex *src = (const float complex*)arr->data;
        float *dst = (float*)result->data;
        
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            dst[i] = crealf(src[i]);
        }
    }
    
    return result;
}

NDArray ndarray_new_imag_part(const NDArray arr) {
    if (!ndarray_is_complex_type(arr->dtype)) {
        fprintf(stderr, "Error: Input array must be complex type\n");
        return NULL;
    }
    
    NDArray result = ndarray_new_typed(arr->dims, 
                                   (arr->dtype == NDA_COMPLEX64) ? NDA_REAL64 : NDA_REAL32);
    size_t size = ndarray_size(arr);
    
    if (arr->dtype == NDA_COMPLEX64) {
        const double complex *src = (const double complex*)arr->data;
        double *dst = (double*)result->data;
        
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            dst[i] = cimag(src[i]);
        }
    } else { // NDA_COMPLEX32
        const float complex *src = (const float complex*)arr->data;
        float *dst = (float*)result->data;
        
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            dst[i] = cimagf(src[i]);
        }
    }
    
    return result;
}

// Complex conjugation

NDArray ndarray_new_conjugate(const NDArray arr) {
    if (!ndarray_is_complex_type(arr->dtype)) {
        fprintf(stderr, "Error: Input array must be complex type\n");
        return NULL;
    }
    
    NDArray result = ndarray_new_typed(arr->dims, arr->dtype);
    size_t size = ndarray_size(arr);
    
    if (arr->dtype == NDA_COMPLEX64) {
        const double complex *src = (const double complex*)arr->data;
        double complex *dst = (double complex*)result->data;
        
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            dst[i] = conj(src[i]);
        }
    } else { // NDA_COMPLEX32
        const float complex *src = (const float complex*)arr->data;
        float complex *dst = (float complex*)result->data;
        
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            dst[i] = conjf(src[i]);
        }
    }
    
    return result;
}

// Magnitude and phase extraction

NDArray ndarray_new_magnitude(const NDArray arr) {
    if (!ndarray_is_complex_type(arr->dtype)) {
        fprintf(stderr, "Error: Input array must be complex type\n");
        return NULL;
    }
    
    NDArray result = ndarray_new_typed(arr->dims, 
                                   (arr->dtype == NDA_COMPLEX64) ? NDA_REAL64 : NDA_REAL32);
    size_t size = ndarray_size(arr);
    
    if (arr->dtype == NDA_COMPLEX64) {
        const double complex *src = (const double complex*)arr->data;
        double *dst = (double*)result->data;
        
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            dst[i] = cabs(src[i]);
        }
    } else { // NDA_COMPLEX32
        const float complex *src = (const float complex*)arr->data;
        float *dst = (float*)result->data;
        
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            dst[i] = cabsf(src[i]);
        }
    }
    
    return result;
}

NDArray ndarray_new_phase(const NDArray arr) {
    if (!ndarray_is_complex_type(arr->dtype)) {
        fprintf(stderr, "Error: Input array must be complex type\n");
        return NULL;
    }
    
    NDArray result = ndarray_new_typed(arr->dims, 
                                   (arr->dtype == NDA_COMPLEX64) ? NDA_REAL64 : NDA_REAL32);
    size_t size = ndarray_size(arr);
    
    if (arr->dtype == NDA_COMPLEX64) {
        const double complex *src = (const double complex*)arr->data;
        double *dst = (double*)result->data;
        
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            dst[i] = carg(src[i]);
        }
    } else { // NDA_COMPLEX32
        const float complex *src = (const float complex*)arr->data;
        float *dst = (float*)result->data;
        
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            dst[i] = cargf(src[i]);
        }
    }
    
    return result;
}

// Type conversion utilities

NDArray ndarray_convert_type(const NDArray arr, NDAType target_type) {
    if (arr->dtype == target_type) {
        // Same type - just copy
        return ndarray_new_copy(arr);
    }
    
    size_t size = ndarray_size(arr);
    NDArray result = ndarray_new_typed(arr->dims, target_type);
    
    if (result == NULL) return NULL;
    
    // Convert from source type
    if (arr->dtype == NDA_REAL64) {
        const double *src = (const double*)arr->data;
        
        if (target_type == NDA_REAL32) {
            float *dst = (float*)result->data;
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < size; ++i) {
                dst[i] = (float)src[i];
            }
        } else if (target_type == NDA_COMPLEX64) {
            double complex *dst = (double complex*)result->data;
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < size; ++i) {
                dst[i] = src[i] + 0.0 * I;
            }
        } else if (target_type == NDA_COMPLEX32) {
            float complex *dst = (float complex*)result->data;
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < size; ++i) {
                dst[i] = (float)src[i] + 0.0f * I;
            }
        }
    } else if (arr->dtype == NDA_REAL32) {
        const float *src = (const float*)arr->data;
        
        if (target_type == NDA_REAL64) {
            double *dst = (double*)result->data;
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < size; ++i) {
                dst[i] = (double)src[i];
            }
        } else if (target_type == NDA_COMPLEX64) {
            double complex *dst = (double complex*)result->data;
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < size; ++i) {
                dst[i] = (double)src[i] + 0.0 * I;
            }
        } else if (target_type == NDA_COMPLEX32) {
            float complex *dst = (float complex*)result->data;
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < size; ++i) {
                dst[i] = src[i] + 0.0f * I;
            }
        }
    } else if (arr->dtype == NDA_COMPLEX64) {
        const double complex *src = (const double complex*)arr->data;
        
        if (target_type == NDA_REAL64) {
            double *dst = (double*)result->data;
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < size; ++i) {
                dst[i] = creal(src[i]);
            }
        } else if (target_type == NDA_REAL32) {
            float *dst = (float*)result->data;
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < size; ++i) {
                dst[i] = (float)creal(src[i]);
            }
        } else if (target_type == NDA_COMPLEX32) {
            float complex *dst = (float complex*)result->data;
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < size; ++i) {
                dst[i] = (float)src[0] + (float)src[1] * I;
            }
        }
    } else if (arr->dtype == NDA_COMPLEX32) {
        const float complex *src = (const float complex*)arr->data;
        
        if (target_type == NDA_REAL64) {
            double *dst = (double*)result->data;
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < size; ++i) {
                dst[i] = (double)crealf(src[i]);
            }
        } else if (target_type == NDA_REAL32) {
            float *dst = (float*)result->data;
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < size; ++i) {
                dst[i] = crealf(src[i]);
            }
        } else if (target_type == NDA_COMPLEX64) {
            double complex *dst = (double complex*)result->data;
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < size; ++i) {
                dst[i] = (double)src[0] + (double)src[1] * I;
            }
        }
    }
    
    return result;
}