/**
 * Shape manipulation operations
 */

#include "ndarray_internal.h"

NDArray ndarray_new_stack(int axis, NDArray* arr_list) {
    assert(arr_list != NULL && arr_list[0] != NULL 
           && "first ndarray cannot be NULL");
    size_t count = 0;
    for (NDArray* p = arr_list; *p != NULL; ++p) {
        count++;
    }
    // Build result dimensions: insert new axis
    size_t result_ndim = arr_list[0]->ndim + 1;
    size_t result_dims[result_ndim + 1];
    for (size_t i = 0; i < (size_t)axis; ++i) {
        result_dims[i] = arr_list[0]->dims[i];
    }
    result_dims[axis] = count;
    for (size_t i = axis; i < arr_list[0]->ndim; ++i) {
        result_dims[i + 1] = arr_list[0]->dims[i];
    }
    result_dims[result_ndim] = 0;
    NDArray result = ndarray_new(result_dims);
    size_t elem_per_array = 1;
    for (size_t i = 0; i < arr_list[0]->ndim; ++i) {
        elem_per_array *= arr_list[0]->dims[i];
    }
    // Copy data based on axis position
    if (axis == 0) {
        // Stack along first axis: simple sequential copy
        for (size_t k = 0; k < count; ++k) {
            memcpy(result->data + k * elem_per_array, arr_list[k]->data, 
                   elem_per_array * sizeof(double));
        }
    } else {
        // Stack along other axes: strided copy
        OMP_PRAGMA(omp parallel)
        {
            size_t src_idx[arr_list[0]->ndim];
            size_t dst_idx[result_ndim];
            OMP_PRAGMA(omp for)
            for (size_t i = 0; i < elem_per_array; ++i) {
                size_t temp = i;
                for (int d = arr_list[0]->ndim - 1; d >= 0; --d) {
                    src_idx[d] = temp % arr_list[0]->dims[d];
                    temp /= arr_list[0]->dims[d];
                }
                
                for (size_t d = 0; d < (size_t)axis; ++d) {
                    dst_idx[d] = src_idx[d];
                }
                for (size_t d = axis; d < arr_list[0]->ndim; ++d) {
                    dst_idx[d + 1] = src_idx[d];
                }
                
                for (size_t k = 0; k < count; ++k) {
                    dst_idx[axis] = k;
                    size_t dst_offset = ndarray_offset(result, dst_idx);
                    result->data[dst_offset] = arr_list[k]->data[i];
                }
            }
        }
    }
    return result;
}

NDArray ndarray_new_concat(int axis, NDArray* arr_list) {
    assert(arr_list != NULL && arr_list[0] != NULL 
           && "first ndarray cannot be NULL");
    assert(arr_list[0]->ndim >= 2 && "ndarray must have at least 2 dimensions");
    assert(axis >= 0 && axis < (int)arr_list[0]->ndim && "axis out of valid range");
    size_t count = 0;
    // Count and validate: all dims except axis must match
    for (NDArray* p = arr_list; *p != NULL; ++p) {
        assert((*p)->ndim == arr_list[0]->ndim && "all arrays must have same ndim");
        for (size_t i = 0; i < arr_list[0]->ndim; ++i) {
            if ((int)i != axis) {
                assert((*p)->dims[i] == arr_list[0]->dims[i] && 
                       "all dimensions except concat axis must match");
            }
        }
        count++;
    }
    // Calculate total size along concat axis
    size_t total_concat_size = 0;
    for (size_t i = 0; i < count; ++i) {
        total_concat_size += arr_list[i]->dims[axis];
    }
    
    // Build result dimensions (same ndim, extended along axis)
    size_t result_dims[arr_list[0]->ndim + 1];
    for (size_t i = 0; i < arr_list[0]->ndim; ++i) {
        result_dims[i] = (i == (size_t)axis)
            ? total_concat_size
            : arr_list[0]->dims[i];
    }
    result_dims[arr_list[0]->ndim] = 0;
    NDArray result = ndarray_new(result_dims);
    // Calculate slice sizes
    size_t before_axis_size = 1;
    for (size_t i = 0; i < (size_t)axis; ++i) {
        before_axis_size *= arr_list[0]->dims[i];
    }
    
    size_t after_axis_size = 1;
    for (size_t i = axis + 1; i < arr_list[0]->ndim; ++i) {
        after_axis_size *= arr_list[0]->dims[i];
    }
    
    // Copy data based on axis position
    if (axis == (int)(arr_list[0]->ndim - 1)) {
        // Concatenate along last axis: copy each "row" separately
        OMP_PRAGMA(omp parallel for)
        for (size_t outer = 0; outer < before_axis_size; ++outer) {
            size_t dst_offset = outer * total_concat_size;
            for (size_t k = 0; k < count; ++k) {
                size_t src_offset = outer * arr_list[k]->dims[axis];
                memcpy(result->data + dst_offset, 
                       arr_list[k]->data + src_offset,
                       arr_list[k]->dims[axis] * sizeof(double));
                dst_offset += arr_list[k]->dims[axis];
            }
        }
    } else if (axis == 0) {
        // Concatenate along first axis: copy sequentially
        size_t offset = 0;
        for (size_t k = 0; k < count; ++k) {
            size_t chunk_size = arr_list[k]->dims[0] * after_axis_size;
            memcpy(result->data + offset, arr_list[k]->data, 
                   chunk_size * sizeof(double));
            offset += chunk_size;
        }
    } else {
        // Concatenate along middle axis: strided copy
        size_t cumulative_offsets[count];
        cumulative_offsets[0] = 0;
        for (size_t i = 1; i < count; ++i) {
            cumulative_offsets[i] = cumulative_offsets[i-1] + arr_list[i-1]->dims[axis];
        }
        
        OMP_PRAGMA(omp parallel for)
        for (size_t outer = 0; outer < before_axis_size; ++outer) {
            for (size_t k = 0; k < count; ++k) {
                size_t src_offset = outer * arr_list[k]->dims[axis] * after_axis_size;
                size_t dst_offset = outer * total_concat_size * after_axis_size +
                                  cumulative_offsets[k] * after_axis_size;
                size_t chunk_size = arr_list[k]->dims[axis] * after_axis_size;
                
                memcpy(result->data + dst_offset, 
                       arr_list[k]->data + src_offset,
                       chunk_size * sizeof(double));
            }
        }
    }
    
    return result;
}

NDArray ndarray_new_take(NDArray arr, int axis, size_t start, size_t end) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(arr->ndim >= 2 && "ndarray must have at least 2 dimensions");
    assert(axis >= 0 && axis < (int)arr->ndim && "axis out of range");
    assert(start < end && "start must be less than end");
    assert(end <= arr->dims[axis] && "end exceeds dimension size");
    // Build result dimensions
    size_t result_dims[arr->ndim + 1];
    for (size_t i = 0; i < arr->ndim; ++i) {
        result_dims[i] = (i == (size_t)axis) ? (end - start) : arr->dims[i];
    }
    result_dims[arr->ndim] = 0;
    NDArray result = ndarray_new(result_dims);
    // Calculate slice sizes
    size_t before_axis_size = 1;
    for (size_t i = 0; i < (size_t)axis; ++i) {
        before_axis_size *= arr->dims[i];
    }
    size_t axis_slice_size = end - start;
    size_t after_axis_size = 1;
    for (size_t i = axis + 1; i < arr->ndim; ++i) {
        after_axis_size *= arr->dims[i];
    }
    
    // Copy data
    if (axis == 0) {
        // Simple case: copy contiguous block from start
        size_t offset = start * after_axis_size;
        size_t count = axis_slice_size * after_axis_size;
        memcpy(result->data, arr->data + offset, count * sizeof(double));
    } else if (axis == (int)(arr->ndim - 1)) {
        // Last axis: copy each row segment
        OMP_PRAGMA(omp parallel for)
        for (size_t outer = 0; outer < before_axis_size; ++outer) {
            size_t src_offset = outer * arr->dims[axis] + start;
            size_t dst_offset = outer * axis_slice_size;
            memcpy(result->data + dst_offset, 
                   arr->data + src_offset,
                   axis_slice_size * sizeof(double));
        }
    } else {
        // Middle axis: strided copy
        OMP_PRAGMA(omp parallel for)
        for (size_t outer = 0; outer < before_axis_size; ++outer) {
            for (size_t idx = 0; idx < axis_slice_size; ++idx) {
                size_t src_offset = outer * arr->dims[axis] * after_axis_size +
                                  (start + idx) * after_axis_size;
                size_t dst_offset = outer * axis_slice_size * after_axis_size +
                                  idx * after_axis_size;
                memcpy(result->data + dst_offset,
                       arr->data + src_offset,
                       after_axis_size * sizeof(double));
            }
        }
    }
    return result;
}

NDArray ndarray_new_transpose(NDArray A) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    size_t new_dims[A->ndim + 1];
    for (size_t i = 0; i < A->ndim; ++i) {
        new_dims[A->ndim - 1 - i] = A->dims[i];
    }
    new_dims[A->ndim] = 0;
    NDArray B = ndarray_new(new_dims);
    if (A->ndim == 2) {
        size_t rows = A->dims[0];
        size_t cols = A->dims[1];
        const size_t block_size = 32;
        OMP_PRAGMA(omp parallel for collapse(2))
        for (size_t i0 = 0; i0 < rows; i0 += block_size) {
            for (size_t j0 = 0; j0 < cols; j0 += block_size) {
                size_t i_max = (i0 + block_size < rows) ? i0 + block_size : rows;
                size_t j_max = (j0 + block_size < cols) ? j0 + block_size : cols;
                for (size_t i = i0; i < i_max; ++i) {
                    for (size_t j = j0; j < j_max; ++j) {
                        B->data[j * rows + i] = A->data[i * cols + j];
                    }
                }
            }
        }
    } else {
        size_t total_size = ndarray_size(A);
        OMP_PRAGMA(omp parallel)
        {
            size_t src_indices[A->ndim];
            size_t dst_indices[A->ndim];
            OMP_PRAGMA(omp for)
            for (size_t idx = 0; idx < total_size; ++idx) {
                size_t temp = idx;
                for (int d = A->ndim - 1; d >= 0; --d) {
                    src_indices[d] = temp % A->dims[d];
                    temp /= A->dims[d];
                }
                for (size_t d = 0; d < A->ndim; ++d) {
                    dst_indices[d] = src_indices[A->ndim - 1 - d];
                }
                size_t dst_idx = 0;
                size_t mult = 1;
                for (int d = B->ndim - 1; d >= 0; --d) {
                    dst_idx += dst_indices[d] * mult;
                    mult *= B->dims[d];
                }
                
                B->data[dst_idx] = A->data[idx];
            }
        }
    }
    return B;
}

