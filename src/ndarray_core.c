#include "ndarray_internal.h"

// Type information table
static const NDATypeInfo type_info_table[] = {
    { sizeof(double), "real64", "d", 0 },        // NDA_REAL64
    { sizeof(double complex), "complex64", "z", 1 },  // NDA_COMPLEX64  
    { sizeof(float), "real32", "s", 0 },          // NDA_REAL32
    { sizeof(float complex), "complex32", "c", 1 },   // NDA_COMPLEX32
};

const NDATypeInfo* ndarray_get_type_info(NDAType dtype) {
    if (dtype >= 0 && dtype < 4) {
        return &type_info_table[dtype];
    }
    return NULL;
}

NDAType ndarray_promote_types(NDAType type1, NDAType type2) {
    // Promote to higher precision or complex type
    if (type1 == type2) return type1;
    
    // Complex always wins over real of same precision
    if (type1 == NDA_REAL64 && type2 == NDA_COMPLEX64) return NDA_COMPLEX64;
    if (type1 == NDA_COMPLEX64 && type2 == NDA_REAL64) return NDA_COMPLEX64;
    if (type1 == NDA_REAL32 && type2 == NDA_COMPLEX32) return NDA_COMPLEX32;
    if (type1 == NDA_COMPLEX32 && type2 == NDA_REAL32) return NDA_COMPLEX32;
    
    // Higher precision wins
    if (type1 == NDA_REAL64 && type2 == NDA_REAL32) return NDA_REAL64;
    if (type1 == NDA_REAL32 && type2 == NDA_REAL64) return NDA_REAL64;
    if (type1 == NDA_COMPLEX64 && type2 == NDA_COMPLEX32) return NDA_COMPLEX64;
    if (type1 == NDA_COMPLEX32 && type2 == NDA_COMPLEX64) return NDA_COMPLEX64;
    
    // Mixed precision: complex always wins with higher precision
    if ((type1 == NDA_REAL64 && type2 == NDA_COMPLEX32) ||
        (type1 == NDA_COMPLEX32 && type2 == NDA_REAL64)) {
        return NDA_COMPLEX64;
    }
    if ((type1 == NDA_REAL32 && type2 == NDA_COMPLEX64) ||
        (type1 == NDA_COMPLEX64 && type2 == NDA_REAL32)) {
        return NDA_COMPLEX64;
    }
    
    return NDA_REAL64; // fallback
}

// BLAS dispatch helpers
void ndarray_blas_axpy(size_t n, double alpha, const NDArray X, const NDArray Y, NDArray result) {
    const NDATypeInfo* x_info = ndarray_get_type_info(X->dtype);
    const NDATypeInfo* y_info = ndarray_get_type_info(Y->dtype);
    
    if (!x_info || !y_info) return;
    
    // For now, require same type
    if (X->dtype != Y->dtype) return;
    
    switch (X->dtype) {
        case NDA_REAL64:
            cblas_daxpy(n, alpha, (const double*)X->data, 1, (const double*)Y->data, 1);
            break;
        case NDA_REAL32:
            cblas_saxpy(n, alpha, (const float*)X->data, 1, (const float*)Y->data, 1);
            break;
        case NDA_COMPLEX64:
            cblas_zaxpy(n, alpha, (const double complex*)X->data, 1, (const double complex*)Y->data, 1);
            break;
        case NDA_COMPLEX32:
            cblas_caxpy(n, alpha, (const float complex*)X->data, 1, (const float complex*)Y->data, 1);
            break;
    }
}

void ndarray_blas_scal(size_t n, double alpha, const NDArray X) {
    const NDATypeInfo* x_info = ndarray_get_type_info(X->dtype);
    if (!x_info) return;
    
    switch (X->dtype) {
        case NDA_REAL64:
            cblas_dscal(n, alpha, (double*)X->data, 1);
            break;
        case NDA_REAL32:
            cblas_sscal(n, alpha, (float*)X->data, 1);
            break;
        case NDA_COMPLEX64:
            cblas_zscal(n, alpha, (double complex*)X->data, 1);
            break;
        case NDA_COMPLEX32:
            cblas_cscal(n, alpha, (float complex*)X->data, 1);
            break;
    }
}
}

size_t ndarray_element_size(NDAType dtype) {
    const NDATypeInfo* info = ndarray_get_type_info(dtype);
    return info ? info->element_size : sizeof(double);
}

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
