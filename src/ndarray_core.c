#include "ndarray_internal.h"

size_t ndarray_size(const NDArray t) {
    size_t size = 1;
    for (size_t i = 0; i < t->ndim; ++i) {
        size *= t->dims[i];
    }
    return size;
}

NDArray ndarray_new(const size_t* dims) {
    size_t ndim = 0;
    size_t size = 1;
    while (dims[ndim] != 0) {
        size *= dims[ndim];
        ndim++;
    }
#ifndef NDEBUG
    assert(ndim >= 2 && "ndarray must have at least 2 dimensions");
#endif
    NDArray t = (NDArray) malloc(sizeof(NDArray_));
    t->ndim = ndim;
    t->dims = (size_t*) malloc(sizeof(size_t) * ndim);
    for (size_t i = 0; i < ndim; ++i) {
        t->dims[i] = dims[i];
    }
    t->data = (double*) malloc(sizeof(double) * size);
    
    /* Initialize GSL-compatibility fields */
    t->stride = 1;                          /* unit stride for contiguous access */
    t->tda = (ndim >= 2) ? dims[1] : 1;    /* trailing dimension: row width for 2D */
    t->owner = 1;                           /* this ndarray owns its memory */
    
    return t;
}

void ndarray_free(NDArray t) {
    if (t == NULL) return;
    if (t->owner) {
        /* Only free memory if this ndarray owns it */
        if (t->dims != NULL) free(t->dims);
        if (t->data != NULL) free(t->data);
    }
    free(t);
}

void ndarray_free_all(NDArray arr_list[]) {
    if (arr_list == NULL) return;
    for (NDArray* p = arr_list; *p != NULL; ++p) {
        ndarray_free(*p);
    }
}

size_t ndarray_offset(const NDArray t, const size_t *pos) {
    size_t offset = 0;
    size_t stride = 1;
    for (int i = t->ndim - 1; i >= 0; --i) {
        offset += pos[i] * stride;
        stride *= t->dims[i];
    }
    return offset;
}

void ndarray_set(const NDArray t, const size_t* pos, double value) {
    size_t p = ndarray_offset(t, pos);
    t->data[p] = value;
}

double ndarray_get(const NDArray t, const size_t* pos) {
    size_t p = ndarray_offset(t, pos);
    return t->data[p];
}

NDArray ndarray_new_copy(const NDArray t) {
    assert(t != NULL && "ndarray cannot be NULL");
    assert(t->ndim >= 2 && "ndarray must have at least 2 dimensions");
    NDArray copy = (NDArray) malloc(sizeof(NDArray_));
    copy->ndim = t->ndim;
    copy->dims = (size_t*) malloc(sizeof(size_t) * t->ndim);
    memcpy(copy->dims, t->dims, sizeof(size_t) * t->ndim);
    size_t size = 1;
    for (size_t i = 0; i < t->ndim; ++i) {
        size *= t->dims[i];
    }
    copy->data = (double*) malloc(sizeof(double) * size);
    cblas_dcopy(size, t->data, 1, copy->data, 1);
    
    /* Copy GSL-compatibility fields */
    copy->stride = t->stride;
    copy->tda = t->tda;
    copy->owner = 1;  /* copy owns its own memory */
    
    return copy;
}

NDArray ndarray_set_slice(const NDArray arr, int axis, size_t index, 
        const double* values) {
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
    return arr;
}

NDArray ndarray_fill_slice(const NDArray arr, int axis, size_t index, double value) {
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
    return arr;
}

double* ndarray_get_slice_ptr(const NDArray arr, int axis, size_t index) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(axis >= 0 && axis < (int)arr->ndim && "axis out of range");
    assert(index < arr->dims[axis] && "index exceeds dimension size");
    // Calculate offset to the start of the slice
    size_t offset = 0;
    size_t stride = 1;
    // Calculate stride for dimensions after the target axis
    for (size_t i = axis + 1; i < arr->ndim; ++i) {
        stride *= arr->dims[i];
    }
    // Offset is index * stride
    offset = index * stride;
    return &arr->data[offset];
}

NDArray ndarray_copy_slice(const NDArray dst, int dst_axis, size_t dst_idx,
                       const NDArray src, int src_axis, size_t src_idx) {
    assert((dst != NULL && src != NULL) &&
            "Destination or source cannot be null");
    assert((dst_axis >= 0 && dst_axis < (int)dst->ndim) &&
            "Destination axis out of range");
    assert((src_axis >= 0 && src_axis < (int)src->ndim) &&
            "Source axis out of range");
    assert((dst_idx < dst->dims[dst_axis]) &&
            "Destination index out of range");
    assert((src_idx < src->dims[src_axis]) &&
            "Source index out of range");
    // Calculate slice sizes
    size_t dst_slice_size = ndarray_get_slice_size(dst, dst_axis);
    size_t src_slice_size = ndarray_get_slice_size(src, src_axis);
    assert((dst_slice_size == src_slice_size) &&
            "Slice sizes don't match");
    // Get pointers to the slices
    double* dst_ptr = ndarray_get_slice_ptr(dst, dst_axis, dst_idx);
    double* src_ptr = ndarray_get_slice_ptr(src, src_axis, src_idx);
    // Copy the data
    memcpy(dst_ptr, src_ptr, dst_slice_size * sizeof(double));
    return dst;
}

size_t ndarray_get_slice_size(const NDArray arr, int axis) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(axis >= 0 && axis < (int)arr->ndim && "axis out of range");
    // Size is the product of all dimensions except the target axis
    size_t size = 1;
    for (size_t i = 0; i < arr->ndim; ++i) {
        if ((int)i != axis) {
            size *= arr->dims[i];
        }
    }
    return size;
}
