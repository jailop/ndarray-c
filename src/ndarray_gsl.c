/**
 * NDArray ↔ GSL Vector/Matrix Integration Implementation
 * 
 * Provides zero-copy conversions between NDArray and GSL structures.
 * All returned views point directly to NDArray's memory with no data copying.
 */

#include "ndarray_internal.h"

/**
 * Calculate the size of a slice along a given axis.
 * Returns the product of all dimensions except the given axis.
 */
static size_t ndarray_slice_size(const NDArray arr, int axis) {
    size_t size = 1;
    for (size_t i = 0; i < arr->ndim; ++i) {
        if ((int)i != axis) {
            size *= arr->dims[i];
        }
    }
    return size;
}

/**
 * Calculate the stride in elements for a given axis.
 * This is the number of elements to skip to move to the next element along that axis.
 */
static size_t ndarray_axis_stride(const NDArray arr, int axis) {
    size_t stride = 1;
    for (int i = arr->ndim - 1; i > axis; --i) {
        stride *= arr->dims[i];
    }
    return stride;
}

/* ============================================================
 * Vector View Functions
 * ============================================================ */

gsl_vector_view ndarray_to_gsl_vector(NDArray arr, int axis, size_t index) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(axis >= 0 && axis < (int)arr->ndim && "axis out of range");
    assert(index < arr->dims[axis] && "index exceeds dimension size");
    
    gsl_vector_view v = { { 0, 0, NULL, NULL, 0 } };
    
    /* For 2D arrays with one dimension == 1, treat as 1D vector */
    if (arr->ndim == 2 && ((arr->dims[0] == 1 && axis == 0) || 
                           (arr->dims[1] == 1 && axis == 1))) {
        /* Return the entire array as a vector */
        v.vector.size = (arr->dims[0] == 1) ? arr->dims[1] : arr->dims[0];
        v.vector.stride = arr->stride;
        v.vector.data = arr->data;
        v.vector.block = NULL;
        v.vector.owner = 0;  /* borrowed, don't free */
        return v;
    }
    
    /* General case: extract a slice */
    size_t size = ndarray_slice_size(arr, axis);
    size_t stride_elements = ndarray_axis_stride(arr, axis);
    
    /* Calculate pointer to the start of the slice */
    size_t offset = index * stride_elements;
    
    v.vector.size = size;
    v.vector.stride = arr->stride;  /* Use array's stride */
    v.vector.data = arr->data + offset;
    v.vector.block = NULL;
    v.vector.owner = 0;  /* borrowed, don't free */
    
    return v;
}

gsl_vector_const_view ndarray_to_gsl_vector_const(const NDArray arr, int axis, size_t index) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(axis >= 0 && axis < (int)arr->ndim && "axis out of range");
    assert(index < arr->dims[axis] && "index exceeds dimension size");
    
    /* We need to create a const view by initializing a mutable vector first,
     * then casting to const. This is a known pattern in GSL. */
    gsl_vector_const_view v;
    gsl_vector *temp_v = (gsl_vector*)&v.vector;
    
    /* For 2D arrays with one dimension == 1, treat as 1D vector */
    if (arr->ndim == 2 && ((arr->dims[0] == 1 && axis == 0) || 
                           (arr->dims[1] == 1 && axis == 1))) {
        /* Return the entire array as a vector */
        temp_v->size = (arr->dims[0] == 1) ? arr->dims[1] : arr->dims[0];
        temp_v->stride = arr->stride;
        temp_v->data = (double*)arr->data;
        temp_v->block = NULL;
        temp_v->owner = 0;  /* borrowed, don't free */
        return v;
    }
    
    /* General case: extract a slice */
    size_t size = ndarray_slice_size(arr, axis);
    size_t stride_elements = ndarray_axis_stride(arr, axis);
    
    /* Calculate pointer to the start of the slice */
    size_t offset = index * stride_elements;
    
    temp_v->size = size;
    temp_v->stride = arr->stride;  /* Use array's stride */
    temp_v->data = (double*)(arr->data + offset);
    temp_v->block = NULL;
    temp_v->owner = 0;  /* borrowed, don't free */
    
    return v;
}

/* ============================================================
 * Matrix View Functions
 * ============================================================ */

gsl_matrix_view ndarray_to_gsl_matrix(NDArray arr) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(arr->ndim == 2 && "ndarray must be 2D for matrix view");
    
    gsl_matrix_view m = { { 0, 0, 0, NULL, NULL, 0 } };
    
    m.matrix.size1 = arr->dims[0];      /* number of rows */
    m.matrix.size2 = arr->dims[1];      /* number of columns */
    m.matrix.tda = arr->tda;            /* trailing dimension (row pitch) */
    m.matrix.data = arr->data;
    m.matrix.block = NULL;
    m.matrix.owner = 0;                 /* borrowed, don't free */
    
    return m;
}

gsl_matrix_const_view ndarray_to_gsl_matrix_const(const NDArray arr) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(arr->ndim == 2 && "ndarray must be 2D for matrix view");
    
    /* We need to create a const view by initializing a mutable matrix first,
     * then casting to const. This is a known pattern in GSL. */
    gsl_matrix_const_view m;
    gsl_matrix *temp_m = (gsl_matrix*)&m.matrix;
    
    temp_m->size1 = arr->dims[0];       /* number of rows */
    temp_m->size2 = arr->dims[1];       /* number of columns */
    temp_m->tda = arr->tda;             /* trailing dimension (row pitch) */
    temp_m->data = (double*)arr->data;
    temp_m->block = NULL;
    temp_m->owner = 0;                  /* borrowed, don't free */
    
    return m;
}

/* ============================================================
 * Row and Column View Functions
 * ============================================================ */

gsl_vector_view ndarray_to_gsl_row(NDArray arr, size_t row_idx) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(arr->ndim == 2 && "ndarray must be 2D to extract rows");
    assert(row_idx < arr->dims[0] && "row index out of range");
    
    gsl_vector_view v = { { 0, 0, NULL, NULL, 0 } };
    
    /* Row is contiguous in row-major layout */
    v.vector.size = arr->dims[1];               /* number of columns */
    v.vector.stride = 1;                        /* contiguous */
    v.vector.data = arr->data + row_idx * arr->tda;
    v.vector.block = NULL;
    v.vector.owner = 0;                         /* borrowed, don't free */
    
    return v;
}

gsl_vector_const_view ndarray_to_gsl_row_const(const NDArray arr, size_t row_idx) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(arr->ndim == 2 && "ndarray must be 2D to extract rows");
    assert(row_idx < arr->dims[0] && "row index out of range");
    
    gsl_vector_const_view v;
    gsl_vector *temp_v = (gsl_vector*)&v.vector;
    
    /* Row is contiguous in row-major layout */
    temp_v->size = arr->dims[1];                /* number of columns */
    temp_v->stride = 1;                         /* contiguous */
    temp_v->data = (double*)(arr->data + row_idx * arr->tda);
    temp_v->block = NULL;
    temp_v->owner = 0;                          /* borrowed, don't free */
    
    return v;
}

gsl_vector_view ndarray_to_gsl_column(NDArray arr, size_t col_idx) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(arr->ndim == 2 && "ndarray must be 2D to extract columns");
    assert(col_idx < arr->dims[1] && "column index out of range");
    
    gsl_vector_view v = { { 0, 0, NULL, NULL, 0 } };
    
    /* Column requires strided access in row-major layout */
    v.vector.size = arr->dims[0];               /* number of rows */
    v.vector.stride = arr->tda;                 /* stride to next row */
    v.vector.data = arr->data + col_idx;
    v.vector.block = NULL;
    v.vector.owner = 0;                         /* borrowed, don't free */
    
    return v;
}

gsl_vector_const_view ndarray_to_gsl_column_const(const NDArray arr, size_t col_idx) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(arr->ndim == 2 && "ndarray must be 2D to extract columns");
    assert(col_idx < arr->dims[1] && "column index out of range");
    
    gsl_vector_const_view v;
    gsl_vector *temp_v = (gsl_vector*)&v.vector;
    
    /* Column requires strided access in row-major layout */
    temp_v->size = arr->dims[0];                /* number of rows */
    temp_v->stride = arr->tda;                  /* stride to next row */
    temp_v->data = (double*)(arr->data + col_idx);
    temp_v->block = NULL;
    temp_v->owner = 0;                          /* borrowed, don't free */
    
    return v;
}

/* ============================================================
 * Getter/Setter Functions
 * ============================================================ */

size_t ndarray_get_stride(const NDArray arr) {
    assert(arr != NULL && "ndarray cannot be NULL");
    return arr->stride;
}

NDArray ndarray_set_stride(NDArray arr, size_t stride) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(stride > 0 && "stride must be positive");
    arr->stride = stride;
    return arr;
}

size_t ndarray_get_tda(const NDArray arr) {
    assert(arr != NULL && "ndarray cannot be NULL");
    return arr->tda;
}

NDArray ndarray_set_tda(NDArray arr, size_t tda) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(tda > 0 && "tda must be positive");
    arr->tda = tda;
    return arr;
}

int ndarray_get_owner(const NDArray arr) {
    assert(arr != NULL && "ndarray cannot be NULL");
    return arr->owner;
}

NDArray ndarray_set_owner(NDArray arr, int owner) {
    assert(arr != NULL && "ndarray cannot be NULL");
    arr->owner = owner ? 1 : 0;
    return arr;
}
