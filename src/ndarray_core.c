/**
 * Core ndarray operations: allocation, deallocation, and element access
 */

#include "ndarray_internal.h"

size_t ndarray_size(NDArray t) {
    size_t size = 1;
    for (size_t i = 0; i < t->ndim; ++i) {
        size *= t->dims[i];
    }
    return size;
}

NDArray ndarray_new(size_t* dims) {
    size_t ndim = 0;
    size_t size = 1;
    while (dims[ndim] != 0) {
        size *= dims[ndim];
        ndim++;
    }
#ifndef NDEBUG
    assert(ndim >= 2 && "ndarray must have at least 2 dimensions");
#endif
    NDArray t = (NDArray) malloc(sizeof(_NDArray));
    t->ndim = ndim;
    t->dims = (size_t*) malloc(sizeof(size_t) * ndim);
    for (size_t i = 0; i < ndim; ++i) {
        t->dims[i] = dims[i];
    }
    t->data = (double*) malloc(sizeof(double) * size);
    return t;
}

void ndarray_free(NDArray t) {
    if (t == NULL) return;
    if (t->dims != NULL) free(t->dims);
    if (t->data != NULL) free(t->data);
    free(t);
}

void ndarray_free_all(NDArray arr_list[]) {
    if (arr_list == NULL) return;
    for (NDArray* p = arr_list; *p != NULL; ++p) {
        ndarray_free(*p);
    }
}

size_t ndarray_offset(NDArray t, size_t *pos) {
    size_t offset = 0;
    size_t stride = 1;
    for (int i = t->ndim - 1; i >= 0; --i) {
        offset += pos[i] * stride;
        stride *= t->dims[i];
    }
    return offset;
}

void ndarray_set(NDArray t, size_t* pos, double value) {
    size_t p = ndarray_offset(t, pos);
    t->data[p] = value;
}

double ndarray_get(NDArray t, size_t* pos) {
    size_t p = ndarray_offset(t, pos);
    return t->data[p];
}

NDArray ndarray_new_copy(NDArray t) {
    assert(t != NULL && "ndarray cannot be NULL");
    assert(t->ndim >= 2 && "ndarray must have at least 2 dimensions");
    NDArray copy = (NDArray) malloc(sizeof(_NDArray));
    copy->ndim = t->ndim;
    copy->dims = (size_t*) malloc(sizeof(size_t) * t->ndim);
    memcpy(copy->dims, t->dims, sizeof(size_t) * t->ndim);
    size_t size = 1;
    for (size_t i = 0; i < t->ndim; ++i) {
        size *= t->dims[i];
    }
    copy->data = (double*) malloc(sizeof(double) * size);
    // Use CBLAS dcopy for efficient array copy
    cblas_dcopy(size, t->data, 1, copy->data, 1);
    return copy;
}

void ndarray_set_slice(NDArray arr, int axis, size_t index, double* values) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(values != NULL && "values cannot be NULL");
    assert(axis >= 0 && axis < (int)arr->ndim && "axis out of range");
    assert(index < arr->dims[axis] && "index exceeds dimension size");
    
    // Calculate sizes for iteration
    size_t before_axis_size = 1;
    for (int i = 0; i < axis; ++i) {
        before_axis_size *= arr->dims[i];
    }
    
    size_t after_axis_size = 1;
    for (size_t i = axis + 1; i < arr->ndim; ++i) {
        after_axis_size *= arr->dims[i];
    }
    
    // Copy data based on axis position
    if (axis == (int)(arr->ndim - 1)) {
        // Last axis: contiguous memory, simple iteration
        OMP_PRAGMA(omp parallel for)
        for (size_t outer = 0; outer < before_axis_size; ++outer) {
            size_t offset = outer * arr->dims[axis] + index;
            arr->data[offset] = values[outer];
        }
    } else if (axis == 0) {
        // First axis: each slice is contiguous
        size_t offset = index * after_axis_size;
        memcpy(arr->data + offset, values, after_axis_size * sizeof(double));
    } else {
        // Middle axis: strided access
        OMP_PRAGMA(omp parallel for)
        for (size_t outer = 0; outer < before_axis_size; ++outer) {
            size_t src_idx = outer * after_axis_size;
            size_t dst_offset = outer * arr->dims[axis] * after_axis_size +
                              index * after_axis_size;
            memcpy(arr->data + dst_offset, values + src_idx,
                   after_axis_size * sizeof(double));
        }
    }
}

void ndarray_fill_slice(NDArray arr, int axis, size_t index, double value) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(axis >= 0 && axis < (int)arr->ndim && "axis out of range");
    assert(index < arr->dims[axis] && "index exceeds dimension size");
    
    // Calculate sizes for iteration
    size_t before_axis_size = 1;
    for (int i = 0; i < axis; ++i) {
        before_axis_size *= arr->dims[i];
    }
    
    size_t after_axis_size = 1;
    for (size_t i = axis + 1; i < arr->ndim; ++i) {
        after_axis_size *= arr->dims[i];
    }
    
    // Fill data based on axis position
    if (axis == (int)(arr->ndim - 1)) {
        // Last axis: simple iteration
        OMP_PRAGMA(omp parallel for)
        for (size_t outer = 0; outer < before_axis_size; ++outer) {
            size_t offset = outer * arr->dims[axis] + index;
            arr->data[offset] = value;
        }
    } else if (axis == 0) {
        // First axis: fill contiguous block
        size_t offset = index * after_axis_size;
        for (size_t i = 0; i < after_axis_size; ++i) {
            arr->data[offset + i] = value;
        }
    } else {
        // Middle axis: strided fill
        OMP_PRAGMA(omp parallel for)
        for (size_t outer = 0; outer < before_axis_size; ++outer) {
            size_t dst_offset = outer * arr->dims[axis] * after_axis_size +
                              index * after_axis_size;
            for (size_t i = 0; i < after_axis_size; ++i) {
                arr->data[dst_offset + i] = value;
            }
        }
    }
}
