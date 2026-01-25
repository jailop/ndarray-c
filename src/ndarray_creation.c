#include "ndarray_internal.h"

NDArray ndarray_new_zeros_typed(const size_t *dims, NDAType dtype) {
    NDArray t = ndarray_new_typed(dims, dtype);
    size_t size = ndarray_size(t);
    size_t element_size = ndarray_element_size(dtype);
    memset(t->data, 0, element_size * size);
    return t;
}

NDArray ndarray_new_zeros(const size_t *dims) {
    return ndarray_new_zeros_typed(dims, NDA_REAL64);
}

NDArray ndarray_new_from_data_typed(const size_t *dims, const void *data, NDAType dtype) {
    NDArray t = ndarray_new_typed(dims, dtype);
    size_t size = ndarray_size(t);
    size_t element_size = ndarray_element_size(dtype);
    memcpy(t->data, data, element_size * size);
    return t;
}

NDArray ndarray_new_from_data(const size_t *dims, const double *data) {
    return ndarray_new_from_data_typed(dims, data, NDA_REAL64);
}

NDArray ndarray_new_ones_typed(const size_t *dims, NDAType dtype) {
    NDArray t = ndarray_new_typed(dims, dtype);
    size_t size = ndarray_size(t);
    const NDATypeInfo *type_info = ndarray_get_type_info(dtype);
    
    if (dtype == NDA_REAL64) {
        double *data = (double*)t->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            data[i] = 1.0;
        }
    } else if (dtype == NDA_REAL32) {
        float *data = (float*)t->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            data[i] = 1.0f;
        }
    } else if (dtype == NDA_COMPLEX64) {
        double complex *data = (double complex*)t->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            data[i] = 1.0 + 0.0 * I;
        }
    } else if (dtype == NDA_COMPLEX32) {
        float complex *data = (float complex*)t->data;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            data[i] = 1.0f + 0.0f * I;
        }
    }
    return t;
}

NDArray ndarray_new_ones(const size_t *dims) {
    return ndarray_new_ones_typed(dims, NDA_REAL64);
}

NDArray ndarray_new_full_typed(const size_t *dims, NDAType dtype, const void *value) {
    NDArray t = ndarray_new_typed(dims, dtype);
    size_t size = ndarray_size(t);
    
    if (dtype == NDA_REAL64) {
        double *data = (double*)t->data;
        double val = *(const double*)value;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            data[i] = val;
        }
    } else if (dtype == NDA_REAL32) {
        float *data = (float*)t->data;
        float val = *(const float*)value;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            data[i] = val;
        }
    } else if (dtype == NDA_COMPLEX64) {
        double complex *data = (double complex*)t->data;
        double complex val = *(const double complex*)value;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            data[i] = val;
        }
    } else if (dtype == NDA_COMPLEX32) {
        float complex *data = (float complex*)t->data;
        float complex val = *(const float complex*)value;
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < size; ++i) {
            data[i] = val;
        }
    }
    return t;
}

NDArray ndarray_new_full(const size_t *dims, double value) {
    return ndarray_new_full_typed(dims, NDA_REAL64, &value);
}

NDArray ndarray_new_arange_typed(const size_t *dims, NDAType dtype, double start, double stop, double step) {
    NDArray t = ndarray_new_typed(dims, dtype);
    size_t size = ndarray_size(t);
    size_t num_elements = 0;
    if (step > 0) {
        num_elements = (size_t)ceil((stop - start) / step);
    }
    if (num_elements > size) num_elements = size;
    
    if (dtype == NDA_REAL64) {
        double *data = (double*)t->data;
        if (num_elements >= OMP_THRESHOLD) {
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < num_elements; ++i) {
                data[i] = start + i * step;
            }
        } else {
            for (size_t i = 0; i < num_elements; ++i) {
                data[i] = start + i * step;
            }
        }
        if (num_elements < size) {
            memset(data + num_elements, 0, (size - num_elements) * sizeof(double));
        }
    } else if (dtype == NDA_REAL32) {
        float *data = (float*)t->data;
        if (num_elements >= OMP_THRESHOLD) {
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < num_elements; ++i) {
                data[i] = (float)(start + i * step);
            }
        } else {
            for (size_t i = 0; i < num_elements; ++i) {
                data[i] = (float)(start + i * step);
            }
        }
        if (num_elements < size) {
            memset(data + num_elements, 0, (size - num_elements) * sizeof(float));
        }
    } else if (dtype == NDA_COMPLEX64) {
        double complex *data = (double complex*)t->data;
        if (num_elements >= OMP_THRESHOLD) {
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < num_elements; ++i) {
                data[i] = (start + i * step) + 0.0 * I;
            }
        } else {
            for (size_t i = 0; i < num_elements; ++i) {
                data[i] = (start + i * step) + 0.0 * I;
            }
        }
        if (num_elements < size) {
            memset(data + num_elements, 0, (size - num_elements) * sizeof(double complex));
        }
    } else if (dtype == NDA_COMPLEX32) {
        float complex *data = (float complex*)t->data;
        if (num_elements >= OMP_THRESHOLD) {
            OMP_PRAGMA(omp parallel for simd)
            for (size_t i = 0; i < num_elements; ++i) {
                data[i] = (float)(start + i * step) + 0.0f * I;
            }
        } else {
            for (size_t i = 0; i < num_elements; ++i) {
                data[i] = (float)(start + i * step) + 0.0f * I;
            }
        }
        if (num_elements < size) {
            memset(data + num_elements, 0, (size - num_elements) * sizeof(float complex));
        }
    }
    return t;
}

NDArray ndarray_new_arange(const size_t *dims, double start, double stop, double step) {
    return ndarray_new_arange_typed(dims, NDA_REAL64, start, stop, step);
}

NDArray ndarray_new_linspace_typed(const size_t *dims, NDAType dtype, double start, double stop, size_t num) {
    NDArray t = ndarray_new_typed(dims, dtype);
    size_t size = ndarray_size(t);
    if (num <= 1) {
        if (dtype == NDA_REAL64) {
            ((double*)t->data)[0] = start;
        } else if (dtype == NDA_REAL32) {
            ((float*)t->data)[0] = (float)start;
        } else if (dtype == NDA_COMPLEX64) {
            ((double complex*)t->data)[0] = start + 0.0 * I;
        } else if (dtype == NDA_COMPLEX32) {
            ((float complex*)t->data)[0] = (float)start + 0.0f * I;
        }
        return t;
    }
    double step = (stop - start) / (num - 1);
    size_t max_idx = (size < num) ? size : num;
    
    if (dtype == NDA_REAL64) {
        double *data = (double*)t->data;
        OMP_PRAGMA(omp parallel for)
        for (size_t i = 0; i < max_idx; ++i) {
            data[i] = start + i * step;
        }
    } else if (dtype == NDA_REAL32) {
        float *data = (float*)t->data;
        OMP_PRAGMA(omp parallel for)
        for (size_t i = 0; i < max_idx; ++i) {
            data[i] = (float)(start + i * step);
        }
    } else if (dtype == NDA_COMPLEX64) {
        double complex *data = (double complex*)t->data;
        OMP_PRAGMA(omp parallel for)
        for (size_t i = 0; i < max_idx; ++i) {
            data[i] = (start + i * step) + 0.0 * I;
        }
    } else if (dtype == NDA_COMPLEX32) {
        float complex *data = (float complex*)t->data;
        OMP_PRAGMA(omp parallel for)
        for (size_t i = 0; i < max_idx; ++i) {
            data[i] = (float)(start + i * step) + 0.0f * I;
        }
    }
    return t;
}

NDArray ndarray_new_linspace(const size_t *dims, double start, double stop, size_t num) {
    return ndarray_new_linspace_typed(dims, NDA_REAL64, start, stop, num);
}

// Complex array creation utilities

NDArray ndarray_new_complex64(const size_t *dims) {
    return ndarray_new_typed(dims, NDA_COMPLEX64);
}

NDArray ndarray_new_complex32(const size_t *dims) {
    return ndarray_new_typed(dims, NDA_COMPLEX32);
}

NDArray ndarray_new_zeros_complex64(const size_t *dims) {
    return ndarray_new_zeros_typed(dims, NDA_COMPLEX64);
}

NDArray ndarray_new_zeros_complex32(const size_t *dims) {
    return ndarray_new_zeros_typed(dims, NDA_COMPLEX32);
}

NDArray ndarray_new_from_complex64(const size_t *dims, const void *data) {
    return ndarray_new_from_data_typed(dims, data, NDA_COMPLEX64);
}

NDArray ndarray_new_from_complex32(const size_t *dims, const void *data) {
    return ndarray_new_from_data_typed(dims, data, NDA_COMPLEX32);
}

NDArray ndarray_new_full_complex64(const size_t *dims, const void *value) {
    return ndarray_new_full_typed(dims, NDA_COMPLEX64, value);
}

NDArray ndarray_new_full_complex32(const size_t *dims, const void *value) {
    return ndarray_new_full_typed(dims, NDA_COMPLEX32, value);
}

NDArray ndarray_new_complex64_from_parts(const size_t *dims, const double *real, const double *imag) {
    NDArray t = ndarray_new_complex64(dims);
    size_t size = ndarray_size(t);
    double complex *data = (double complex*)t->data;
    
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        data[i] = real[i] + imag[i] * I;
    }
    return t;
}

NDArray ndarray_new_complex32_from_parts(const size_t *dims, const float *real, const float *imag) {
    NDArray t = ndarray_new_complex32(dims);
    size_t size = ndarray_size(t);
    float complex *data = (float complex*)t->data;
    
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        data[i] = real[i] + imag[i] * I;
    }
    return t;
}

NDArray ndarray_new_complex64_from_polar(const size_t *dims, const double *r, const double *theta) {
    NDArray t = ndarray_new_complex64(dims);
    size_t size = ndarray_size(t);
    double complex *data = (double complex*)t->data;
    
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        data[i] = r[i] * (cos(theta[i]) + sin(theta[i]) * I);
    }
    return t;
}

NDArray ndarray_new_complex32_from_polar(const size_t *dims, const float *r, const float *theta) {
    NDArray t = ndarray_new_complex32(dims);
    size_t size = ndarray_size(t);
    float complex *data = (float complex*)t->data;
    
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        data[i] = r[i] * (cosf(theta[i]) + sinf(theta[i]) * I);
    }
    return t;
}

// Convenience functions for type-specific array creation

NDArray ndarray_new_zeros_real32(const size_t *dims) {
    return ndarray_new_zeros_typed(dims, NDA_REAL32);
}

NDArray ndarray_new_zeros_real64(const size_t *dims) {
    return ndarray_new_zeros_typed(dims, NDA_REAL64);
}

NDArray ndarray_new_ones_real32(const size_t *dims) {
    return ndarray_new_ones_typed(dims, NDA_REAL32);
}

NDArray ndarray_new_ones_real64(const size_t *dims) {
    return ndarray_new_ones_typed(dims, NDA_REAL64);
}

NDArray ndarray_new_full_real32(const size_t *dims, float value) {
    return ndarray_new_full_typed(dims, NDA_REAL32, &value);
}

NDArray ndarray_new_full_real64(const size_t *dims, double value) {
    return ndarray_new_full_typed(dims, NDA_REAL64, &value);
}

NDArray ndarray_new_arange_real32(const size_t *dims, float start, float stop, float step) {
    return ndarray_new_arange_typed(dims, NDA_REAL32, (double)start, (double)stop, (double)step);
}

NDArray ndarray_new_arange_real64(const size_t *dims, double start, double stop, double step) {
    return ndarray_new_arange_typed(dims, NDA_REAL64, start, stop, step);
}

NDArray ndarray_new_linspace_real32(const size_t *dims, float start, float stop, size_t num) {
    return ndarray_new_linspace_typed(dims, NDA_REAL32, (double)start, (double)stop, num);
}

NDArray ndarray_new_linspace_real64(const size_t *dims, double start, double stop, size_t num) {
    return ndarray_new_linspace_typed(dims, NDA_REAL64, start, stop, num);
}
