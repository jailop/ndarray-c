void ndarray_reshape(const NDArray arr, const size_t* new_dims) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(new_dims != NULL && "new_dims cannot be NULL");
    
    // Phase 4: Memory layout-aware type-aware reshaping
    // Count new dimensions and calculate total size
    size_t new_ndim = 0;
    size_t new_size = 1;
    int auto_dim_idx = -1;
    while (new_dims[new_ndim] != 0) {
        if ((int)new_dims[new_ndim] == -1) {
            assert(auto_dim_idx == -1 && "only one dimension can be -1");
            auto_dim_idx = new_ndim;
        } else {
            assert(auto_dim_idx == -1 && "cannot have multiple -1 dimensions");
        }
        new_size *= new_dims[new_ndim];
        new_ndim++;
    }
    assert(new_ndim >= 2 && "result ndarray must have at least 2 dimensions");
    
    // Calculate original and new total sizes
    size_t orig_size = ndarray_size(arr);
    size_t total_size = new_size;
    assert(orig_size == total_size && 
           "cannot reshape: total size must remain the same");
    
    // If -1 is used, calculate the inferred dimension
    size_t inferred_dim = 0;
    if (auto_dim_idx != -1) {
        assert(orig_size % new_size == 0 && 
               "cannot reshape: total size mismatch with inferred dimension");
        inferred_dim = orig_size / new_size;
    }
    
    // Check if we can optimize for contiguous memory layout
    bool can_reuse_data = false;
    if (orig_size == total_size) {
        // Only reuse data if memory layout is compatible (row-major preserved)
        // For now, assume compatible - more complex analysis could be added
        can_reuse_data = true;
    }
    
    // Phase 4: Type-aware result allocation (preserves original type)
    NDArray result;
    if (can_reuse_data) {
        // Same type, can reuse data pointer
        result = ndarray_new_from_data_typed(new_dims, arr->data, arr->dtype);
    } else {
        // Need to copy with type conversion if type would change
        // For now, create new array with original type
        result = ndarray_new_typed(new_dims, arr->dtype);
        
        if (result == NULL) {
            return;
        }
        
        size_t element_size = ndarray_element_size(arr->dtype);
        size_t copy_size = (orig_size < total_size) ? orig_size : total_size;
        
        // Phase 4: Optimized type-aware data copying
        if (copy_size >= OMP_THRESHOLD) {
            // Parallel copy for large arrays
            OMP_PRAGMA(omp parallel for)
            for (size_t flat_idx = 0; flat_idx < copy_size; ++flat_idx) {
                // Convert flat index to multi-dimensional source indices
                size_t src_indices[arr->ndim];
                size_t temp = flat_idx;
                for (size_t i = 0; i < arr->ndim; ++i) {
                    size_t stride = 1;
                    for (size_t j = i + 1; j < arr->ndim; ++j) {
                        stride *= arr->dims[j];
                    }
                    src_indices[i] = temp / stride;
                    temp %= stride;
                }
                
                // Convert source indices to destination flat index
                size_t dst_flat_idx = 0;
                if (auto_dim_idx != -1) {
                    // Reshaping with inferred dimension - map source to new layout
                    for (size_t i = 0; i < new_ndim; ++i) {
                        if (i == (size_t)auto_dim_idx) {
                            // Skip inferred dimension in calculation
                            dst_flat_idx += src_indices[arr->ndim - 1];
                        } else {
                            // Regular mapping for other dimensions
                            dst_flat_idx += src_indices[i];
                        }
                    }
                } else {
                    // Standard reshaping - map source to new layout directly
                    for (size_t i = 0; i < new_ndim; ++i) {
                        dst_flat_idx += src_indices[i];
                    }
                }
                
                // Phase 4: Type-aware element copying
                switch (arr->dtype) {
                    case NDA_REAL64: {
                        double* src_data = (double*)arr->data;
                        double* dst_data = (double*)result->data;
                        dst_data[flat_idx] = src_data[dst_flat_idx];
                        break;
                    }
                    case NDA_REAL32: {
                        float* src_data = (float*)arr->data;
                        float* dst_data = (float*)result->data;
                        dst_data[flat_idx] = src_data[dst_flat_idx];
                        break;
                    }
                    case NDA_COMPLEX64:
                    case NDA_COMPLEX32:
                        // For complex types, use memcpy to preserve both real and imag parts
                        memcpy((char*)result->data + flat_idx * element_size,
                               (char*)arr->data + dst_flat_idx * element_size,
                               element_size);
                        break;
                }
            }
        } else {
            // Sequential copy for small arrays
            for (size_t flat_idx = 0; flat_idx < copy_size; ++flat_idx) {
                // Convert flat index to multi-dimensional source indices
                size_t src_indices[arr->ndim];
                size_t temp = flat_idx;
                for (size_t i = 0; i < arr->ndim; ++i) {
                    size_t stride = 1;
                    for (size_t j = i + 1; j < arr->ndim; ++j) {
                        stride *= arr->dims[j];
                    }
                    src_indices[i] = temp / stride;
                    temp %= stride;
                }
                
                // Convert source indices to destination flat index
                size_t dst_flat_idx = 0;
                if (auto_dim_idx != -1) {
                    // Reshaping with inferred dimension - map source to new layout
                    for (size_t i = 0; i < new_ndim; ++i) {
                        if (i == (size_t)auto_dim_idx) {
                            // Skip inferred dimension in calculation
                            dst_flat_idx += src_indices[arr->ndim - 1];
                        } else {
                            // Regular mapping for other dimensions
                            dst_flat_idx += src_indices[i];
                        }
                    }
                } else {
                    // Standard reshaping - map source to new layout directly
                    for (size_t i = 0; i < new_ndim; ++i) {
                        dst_flat_idx += src_indices[i];
                    }
                }
                
                // Phase 4: Type-aware element copying
                switch (arr->dtype) {
                    case NDA_REAL64: {
                        double* src_data = (double*)arr->data;
                        double* dst_data = (double*)result->data;
                        dst_data[flat_idx] = src_data[dst_flat_idx];
                        break;
                    }
                    case NDA_REAL32: {
                        float* src_data = (float*)arr->data;
                        float* dst_data = (float*)result->data;
                        dst_data[flat_idx] = src_data[dst_flat_idx];
                        break;
                    }
                    case NDA_COMPLEX64:
                    case NDA_COMPLEX32:
                        // For complex types, use memcpy to preserve both real and imag parts
                        memcpy((char*)result->data + flat_idx * element_size,
                               (char*)arr->data + dst_flat_idx * element_size,
                               element_size);
                        break;
                }
            }
        }
    }
}