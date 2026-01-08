#include "ndarray_internal.h"

NDArray ndarray_new_zeros(const size_t *dims) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    memset(t->data, 0, sizeof(double) * size);
    return t;
}

NDArray ndarray_new_from_data(const size_t *dims, const double *data) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    memcpy(t->data, data, sizeof(double) * size);
    return t;
}

NDArray ndarray_new_ones(const size_t *dims) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        t->data[i] = 1.0;
    }
    return t;
}

NDArray ndarray_new_full(const size_t *dims, double value) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        t->data[i] = value;
    }
    return t;
}

NDArray ndarray_new_arange(const size_t *dims, double start, double stop,
        double step) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    size_t num_elements = 0;
    if (step > 0) {
        num_elements = (size_t)ceil((stop - start) / step);
    }
    if (num_elements > size) num_elements = size;
    if (num_elements >= OMP_THRESHOLD) {
        OMP_PRAGMA(omp parallel for simd)
        for (size_t i = 0; i < num_elements; ++i) {
            t->data[i] = start + i * step;
        }
    } else {
        for (size_t i = 0; i < num_elements; ++i) {
            t->data[i] = start + i * step;
        }
    }
    if (num_elements < size) {
        memset(t->data + num_elements, 0, (size - num_elements) * sizeof(double));
    }
    return t;
}

NDArray ndarray_new_linspace(const size_t *dims, double start, double stop,
        size_t num) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    if (num <= 1) {
        t->data[0] = start;
        return t;
    }
    double step = (stop - start) / (num - 1);
    size_t max_idx = (size < num) ? size : num;
    OMP_PRAGMA(omp parallel for)
    for (size_t i = 0; i < max_idx; ++i) {
        t->data[i] = start + i * step;
    }
    return t;
}
