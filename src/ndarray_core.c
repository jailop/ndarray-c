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
