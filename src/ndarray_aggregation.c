#include "ndarray_internal.h"

size_t compute_stride(NDArray A, int axis) {
    size_t stride = 1;
    for (int i = axis + 1; i < (int)A->ndim; ++i) {
        stride *= A->dims[i];
    }
    return stride;
}

static double aggr_full_sum_mean(NDArray A, int aggr_type) {
    size_t size = ndarray_size(A);
    double acc = 0.0;
    
    // Phase 4: Type-aware sum/mean aggregation
    switch (A->dtype) {
        case NDA_REAL64: {
            const double *data = (const double*)A->data;
            OMP_PRAGMA(omp parallel for reduction(+:acc))
            for (size_t i = 0; i < size; ++i) {
                acc += data[i];
            }
            break;
        }
        case NDA_REAL32: {
            const float *data = (const float*)A->data;
            OMP_PRAGMA(omp parallel for reduction(+:acc))
            for (size_t i = 0; i < size; ++i) {
                acc += (double)data[i];
            }
            break;
        }
        case NDA_COMPLEX64:
        case NDA_COMPLEX32:
            // For complex types, use real part for now
            // TODO: Implement proper complex aggregation
            const double *complex_data = (const double*)A->data;
            OMP_PRAGMA(omp parallel for reduction(+:acc))
            for (size_t i = 0; i < size * 2; i += 2) {  // Only real parts
                acc += complex_data[i];
            }
            break;
    }
    
    return (aggr_type == NDA_AGGR_MEAN) ? acc / size : acc;
}

static double aggr_full_max(NDArray A) {
    size_t size = ndarray_size(A);
    double acc;
    
    // Phase 4: Type-aware max aggregation
    switch (A->dtype) {
        case NDA_REAL64: {
            const double *data = (const double*)A->data;
            acc = data[0];
            OMP_PRAGMA(omp parallel for reduction(max:acc))
            for (size_t i = 1; i < size; ++i) {
                if (data[i] > acc) acc = data[i];
            }
            break;
        }
        case NDA_REAL32: {
            const float *data = (const float*)A->data;
            acc = (double)data[0];
            OMP_PRAGMA(omp parallel for reduction(max:acc))
            for (size_t i = 1; i < size; ++i) {
                if ((double)data[i] > acc) acc = (double)data[i];
            }
            break;
        }
        case NDA_COMPLEX64:
        case NDA_COMPLEX32:
            // For complex types, use real part magnitude
            const double *complex_data = (const double*)A->data;
            acc = fabs(complex_data[0]);  // Real part of first element
            OMP_PRAGMA(omp parallel for reduction(max:acc))
            for (size_t i = 2; i < size * 2; i += 2) {  // Real parts only
                if (fabs(complex_data[i]) > acc) acc = fabs(complex_data[i]);
            }
            break;
    }
    return acc;
}

static double aggr_full_min(NDArray A) {
    size_t size = ndarray_size(A);
    double acc;
    
    // Phase 4: Type-aware min aggregation
    switch (A->dtype) {
        case NDA_REAL64: {
            const double *data = (const double*)A->data;
            acc = data[0];
            OMP_PRAGMA(omp parallel for reduction(min:acc))
            for (size_t i = 1; i < size; ++i) {
                if (data[i] < acc) acc = data[i];
            }
            break;
        }
        case NDA_REAL32: {
            const float *data = (const float*)A->data;
            acc = (double)data[0];
            OMP_PRAGMA(omp parallel for reduction(min:acc))
            for (size_t i = 1; i < size; ++i) {
                if ((double)data[i] < acc) acc = (double)data[i];
            }
            break;
        }
        case NDA_COMPLEX64:
        case NDA_COMPLEX32:
            // For complex types, use real part magnitude
            const double *complex_data = (const double*)A->data;
            acc = fabs(complex_data[0]);  // Real part of first element
            OMP_PRAGMA(omp parallel for reduction(min:acc))
            for (size_t i = 2; i < size * 2; i += 2) {  // Real parts only
                if (fabs(complex_data[i]) < acc) acc = fabs(complex_data[i]);
            }
            break;
    }
    return acc;
}

static double aggr_full_std(NDArray A) {
    size_t size = ndarray_size(A);
    double mean = 0.0;
    
    // Phase 4: Type-aware std aggregation - first pass for mean
    switch (A->dtype) {
        case NDA_REAL64: {
            const double *data = (const double*)A->data;
            OMP_PRAGMA(omp parallel for reduction(+:mean))
            for (size_t i = 0; i < size; ++i) {
                mean += data[i];
            }
            break;
        }
        case NDA_REAL32: {
            const float *data = (const float*)A->data;
            OMP_PRAGMA(omp parallel for reduction(+:mean))
            for (size_t i = 0; i < size; ++i) {
                mean += (double)data[i];
            }
            break;
        }
        case NDA_COMPLEX64:
        case NDA_COMPLEX32:
            // For complex types, use real part
            const double *complex_data = (const double*)A->data;
            OMP_PRAGMA(omp parallel for reduction(+:mean))
            for (size_t i = 0; i < size * 2; i += 2) {  // Real parts only
                mean += complex_data[i];
            }
            break;
    }
    mean /= size;
    
    // Second pass for variance
    double variance = 0.0;
    switch (A->dtype) {
        case NDA_REAL64: {
            const double *data = (const double*)A->data;
            OMP_PRAGMA(omp parallel for reduction(+:variance))
            for (size_t i = 0; i < size; ++i) {
                double diff = data[i] - mean;
                variance += diff * diff;
            }
            break;
        }
        case NDA_REAL32: {
            const float *data = (const float*)A->data;
            OMP_PRAGMA(omp parallel for reduction(+:variance))
            for (size_t i = 0; i < size; ++i) {
                double diff = (double)data[i] - mean;
                variance += diff * diff;
            }
            break;
        }
        case NDA_COMPLEX64:
        case NDA_COMPLEX32:
            // For complex types, use real part
            const double *complex_data = (const double*)A->data;
            OMP_PRAGMA(omp parallel for reduction(+:variance))
            for (size_t i = 0; i < size * 2; i += 2) {  // Real parts only
                double diff = complex_data[i] - mean;
                variance += diff * diff;
            }
            break;
    }
    return sqrt(variance / size);
}

static void aggr_axis_sum_mean(NDArray result, NDArray A, int axis,
        int aggr_type) {
    size_t result_size = ndarray_size(result);
    size_t axis_dim = A->dims[axis];
    size_t stride = compute_stride(A, axis);
    
    // Initialize result to zeros
    memset(result->data, 0, result_size * ndarray_element_size(result->dtype));
    
    OMP_PRAGMA(omp parallel for)
    for (size_t i = 0; i < result_size; ++i) {
        size_t outer_idx = i / stride;
        size_t inner_idx = i % stride;
        double sum = 0.0;
        
        for (size_t j = 0; j < axis_dim; ++j) {
            size_t idx = outer_idx * (axis_dim * stride) + j * stride + inner_idx;
            
            // Phase 4: Type-aware axis aggregation
            switch (A->dtype) {
                case NDA_REAL64: {
                    const double *data = (const double*)A->data;
                    sum += data[idx];
                    break;
                }
                case NDA_REAL32: {
                    const float *data = (const float*)A->data;
                    sum += (double)data[idx];
                    break;
                }
                case NDA_COMPLEX64:
                case NDA_COMPLEX32:
                    // For complex types, use real part
                    const double *complex_data = (const double*)A->data;
                    sum += complex_data[idx * 2];  // Real part
                    break;
            }
        }
        
        // Store result based on result type
        double final_val = (aggr_type == NDA_AGGR_MEAN) ? sum / axis_dim : sum;
        switch (result->dtype) {
            case NDA_REAL64: {
                double *res_data = (double*)result->data;
                res_data[i] = final_val;
                break;
            }
            case NDA_REAL32: {
                float *res_data = (float*)result->data;
                res_data[i] = (float)final_val;
                break;
            }
            case NDA_COMPLEX64:
            case NDA_COMPLEX32:
                // For complex results, store in real part
                double *complex_res = (double*)result->data;
                complex_res[i * 2] = final_val;
                complex_res[i * 2 + 1] = 0.0;
                break;
        }
    }
}

static void aggr_axis_max(NDArray result, NDArray A, int axis) {
    size_t result_size = ndarray_size(result);
    size_t axis_dim = A->dims[axis];
    size_t stride = compute_stride(A, axis);
    
    OMP_PRAGMA(omp parallel for)
    for (size_t i = 0; i < result_size; ++i) {
        size_t outer_idx = i / stride;
        size_t inner_idx = i % stride;
        double max_val;
        
        // Initialize with first element
        size_t first_idx = outer_idx * (axis_dim * stride) + inner_idx;
        switch (A->dtype) {
            case NDA_REAL64: {
                const double *data = (const double*)A->data;
                max_val = data[first_idx];
                break;
            }
            case NDA_REAL32: {
                const float *data = (const float*)A->data;
                max_val = (double)data[first_idx];
                break;
            }
            case NDA_COMPLEX64:
            case NDA_COMPLEX32:
                const double *complex_data = (const double*)A->data;
                max_val = fabs(complex_data[first_idx * 2]);  // Real part
                break;
        }
        
        // Find maximum
        for (size_t j = 1; j < axis_dim; ++j) {
            size_t idx = outer_idx * (axis_dim * stride) + j * stride + inner_idx;
            switch (A->dtype) {
                case NDA_REAL64: {
                    const double *data = (const double*)A->data;
                    if (data[idx] > max_val) max_val = data[idx];
                    break;
                }
                case NDA_REAL32: {
                    const float *data = (const float*)A->data;
                    if ((double)data[idx] > max_val) max_val = (double)data[idx];
                    break;
                }
                case NDA_COMPLEX64:
                case NDA_COMPLEX32:
                    const double *complex_data = (const double*)A->data;
                    if (fabs(complex_data[idx * 2]) > max_val) max_val = fabs(complex_data[idx * 2]);
                    break;
            }
        }
        
        // Store result based on result type
        switch (result->dtype) {
            case NDA_REAL64: {
                double *res_data = (double*)result->data;
                res_data[i] = max_val;
                break;
            }
            case NDA_REAL32: {
                float *res_data = (float*)result->data;
                res_data[i] = (float)max_val;
                break;
            }
            case NDA_COMPLEX64:
            case NDA_COMPLEX32:
                double *complex_res = (double*)result->data;
                complex_res[i * 2] = max_val;
                complex_res[i * 2 + 1] = 0.0;
                break;
        }
    }
}

static void aggr_axis_min(NDArray result, NDArray A, int axis) {
    size_t result_size = ndarray_size(result);
    size_t axis_dim = A->dims[axis];
    size_t stride = compute_stride(A, axis);
    
    OMP_PRAGMA(omp parallel for)
    for (size_t i = 0; i < result_size; ++i) {
        size_t outer_idx = i / stride;
        size_t inner_idx = i % stride;
        double min_val;
        
        // Initialize with first element
        size_t first_idx = outer_idx * (axis_dim * stride) + inner_idx;
        switch (A->dtype) {
            case NDA_REAL64: {
                const double *data = (const double*)A->data;
                min_val = data[first_idx];
                break;
            }
            case NDA_REAL32: {
                const float *data = (const float*)A->data;
                min_val = (double)data[first_idx];
                break;
            }
            case NDA_COMPLEX64:
            case NDA_COMPLEX32:
                const double *complex_data = (const double*)A->data;
                min_val = fabs(complex_data[first_idx * 2]);  // Real part
                break;
        }
        
        // Find minimum
        for (size_t j = 1; j < axis_dim; ++j) {
            size_t idx = outer_idx * (axis_dim * stride) + j * stride + inner_idx;
            switch (A->dtype) {
                case NDA_REAL64: {
                    const double *data = (const double*)A->data;
                    if (data[idx] < min_val) min_val = data[idx];
                    break;
                }
                case NDA_REAL32: {
                    const float *data = (const float*)A->data;
                    if ((double)data[idx] < min_val) min_val = (double)data[idx];
                    break;
                }
                case NDA_COMPLEX64:
                case NDA_COMPLEX32:
                    const double *complex_data = (const double*)A->data;
                    if (fabs(complex_data[idx * 2]) < min_val) min_val = fabs(complex_data[idx * 2]);
                    break;
            }
        }
        
        // Store result based on result type
        switch (result->dtype) {
            case NDA_REAL64: {
                double *res_data = (double*)result->data;
                res_data[i] = min_val;
                break;
            }
            case NDA_REAL32: {
                float *res_data = (float*)result->data;
                res_data[i] = (float)min_val;
                break;
            }
            case NDA_COMPLEX64:
            case NDA_COMPLEX32:
                double *complex_res = (double*)result->data;
                complex_res[i * 2] = min_val;
                complex_res[i * 2 + 1] = 0.0;
                break;
        }
    }
}

static void aggr_axis_std(NDArray result, NDArray A, int axis) {
    size_t result_size = ndarray_size(result);
    size_t axis_dim = A->dims[axis];
    size_t stride = compute_stride(A, axis);
    
    OMP_PRAGMA(omp parallel for)
    for (size_t i = 0; i < result_size; ++i) {
        size_t outer_idx = i / stride;
        size_t inner_idx = i % stride;
        
        // Use Welford's algorithm for numerical stability
        double mean = 0.0;
        double m2 = 0.0;
        
        for (size_t j = 0; j < axis_dim; ++j) {
            size_t idx = outer_idx * (axis_dim * stride) + j * stride + inner_idx;
            double val;
            
            // Extract value based on type
            switch (A->dtype) {
                case NDA_REAL64: {
                    const double *data = (const double*)A->data;
                    val = data[idx];
                    break;
                }
                case NDA_REAL32: {
                    const float *data = (const float*)A->data;
                    val = (double)data[idx];
                    break;
                }
                case NDA_COMPLEX64:
                case NDA_COMPLEX32:
                    const double *complex_data = (const double*)A->data;
                    val = complex_data[idx * 2];  // Real part
                    break;
            }
            
            double delta = val - mean;
            mean += delta / (double)(j + 1);
            double delta2 = val - mean;
            m2 += delta * delta2;
        }
        
        double std_val = sqrt(m2 / axis_dim);
        
        // Store result based on result type
        switch (result->dtype) {
            case NDA_REAL64: {
                double *res_data = (double*)result->data;
                res_data[i] = std_val;
                break;
            }
            case NDA_REAL32: {
                float *res_data = (float*)result->data;
                res_data[i] = (float)std_val;
                break;
            }
            case NDA_COMPLEX64:
            case NDA_COMPLEX32:
                double *complex_res = (double*)result->data;
                complex_res[i * 2] = std_val;
                complex_res[i * 2 + 1] = 0.0;
                break;
        }
    }
}

NDArray ndarray_new_aggr(NDArray A, int axis, int aggr_type) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    assert((axis == -1 || (axis >= 0 && axis < (int)A->ndim)) 
            && "axis must be in range [0, ndim-1] or -1 (NDA_ALL_AXES) for all axes");
    if (axis == -1) {
        size_t dims[] = {1, 1, 0};
        NDArray result = ndarray_new_typed(dims, A->dtype);
        double value;
        switch (aggr_type) {
            case NDA_AGGR_SUM:
            case NDA_AGGR_MEAN:
                value = aggr_full_sum_mean(A, aggr_type);
                break;
            case NDA_AGGR_MAX:
                value = aggr_full_max(A);
                break;
            case NDA_AGGR_MIN:
                value = aggr_full_min(A);
                break;
            case NDA_AGGR_STD:
                value = aggr_full_std(A);
                break;
        }
        
        // Store result based on type
        switch (result->dtype) {
            case NDA_REAL64: {
                double *res_data = (double*)result->data;
                res_data[0] = value;
                break;
            }
            case NDA_REAL32: {
                float *res_data = (float*)result->data;
                res_data[0] = (float)value;
                break;
            }
            case NDA_COMPLEX64:
            case NDA_COMPLEX32:
                double *complex_res = (double*)result->data;
                complex_res[0] = value;      // Real part
                complex_res[1] = 0.0;        // Imaginary part
                break;
        }
        return result;
    }
    // result dimensions
    size_t result_ndim = A->ndim - 1;
    if (result_ndim < 2) {
        result_ndim = 2;
    }
    size_t result_dims[result_ndim + 1];
    size_t idx = 0;
    // result dimensions, inserting 1 for aggregated axis if needed
    for (size_t i = 0; i < A->ndim; ++i) {
        if ((int)i == axis) {
            if (result_ndim == 2 && A->ndim == 2) {
                result_dims[idx++] = 1;
            }
        } else {
            result_dims[idx++] = A->dims[i];
        }
    }
    // still don't have 2 dimensions, pad at the end
    while (idx < result_ndim) {
        result_dims[idx++] = 1;
    }
    result_dims[idx] = 0;
    NDArray result = ndarray_new_typed(result_dims, A->dtype);
    switch (aggr_type) {
        case NDA_AGGR_SUM:
        case NDA_AGGR_MEAN:
            aggr_axis_sum_mean(result, A, axis, aggr_type);
            break;
        case NDA_AGGR_MAX:
            aggr_axis_max(result, A, axis);
            break;
        case NDA_AGGR_MIN:
            aggr_axis_min(result, A, axis);
            break;
        case NDA_AGGR_STD:
            aggr_axis_std(result, A, axis);
            break;
        default:
            assert(0 && "invalid aggregation type");
    }
    return result;
}

double ndarray_scalar_aggr(const NDArray A, int aggr_type) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    switch (aggr_type) {
        case NDA_AGGR_SUM:
        case NDA_AGGR_MEAN:
            return aggr_full_sum_mean(A, aggr_type);
        case NDA_AGGR_MAX:
            return aggr_full_max(A);
        case NDA_AGGR_MIN:
            return aggr_full_min(A);
        case NDA_AGGR_STD:
            return aggr_full_std(A);
        default:
            assert(0 && "invalid aggregation type");
            return 0.0;
    }
}

// Phase 4: Convenience statistical functions
double ndarray_sum(const NDArray A) {
    return ndarray_scalar_aggr(A, NDA_AGGR_SUM);
}

double ndarray_mean(const NDArray A) {
    return ndarray_scalar_aggr(A, NDA_AGGR_MEAN);
}

double ndarray_var(const NDArray A) {
    // Variance = standard deviation squared
    double std_val = ndarray_scalar_aggr(A, NDA_AGGR_STD);
    return std_val * std_val;
}

double ndarray_std(const NDArray A) {
    return ndarray_scalar_aggr(A, NDA_AGGR_STD);
}

double ndarray_min(const NDArray A) {
    return ndarray_scalar_aggr(A, NDA_AGGR_MIN);
}

double ndarray_max(const NDArray A) {
    return ndarray_scalar_aggr(A, NDA_AGGR_MAX);
}

static size_t aggr_full_argmin(NDArray A) {
    size_t size = ndarray_size(A);
    size_t min_idx = 0;
    
    // Phase 4: Type-aware argmin aggregation
    switch (A->dtype) {
        case NDA_REAL64: {
            const double *data = (const double*)A->data;
            double min_val = data[0];
            for (size_t i = 1; i < size; ++i) {
                if (data[i] < min_val) {
                    min_val = data[i];
                    min_idx = i;
                }
            }
            break;
        }
        case NDA_REAL32: {
            const float *data = (const float*)A->data;
            float min_val = data[0];
            for (size_t i = 1; i < size; ++i) {
                if (data[i] < min_val) {
                    min_val = data[i];
                    min_idx = i;
                }
            }
            break;
        }
        case NDA_COMPLEX64:
        case NDA_COMPLEX32:
            // For complex types, use real part magnitude
            const double *complex_data = (const double*)A->data;
            double min_val = fabs(complex_data[0]);  // Real part of first element
            for (size_t i = 1; i < size; ++i) {
                if (fabs(complex_data[i * 2]) < min_val) {
                    min_val = fabs(complex_data[i * 2]);
                    min_idx = i;
                }
            }
            break;
    }
    return min_idx;
}

static size_t aggr_full_argmax(NDArray A) {
    size_t size = ndarray_size(A);
    size_t max_idx = 0;
    
    // Phase 4: Type-aware argmax aggregation
    switch (A->dtype) {
        case NDA_REAL64: {
            const double *data = (const double*)A->data;
            double max_val = data[0];
            for (size_t i = 1; i < size; ++i) {
                if (data[i] > max_val) {
                    max_val = data[i];
                    max_idx = i;
                }
            }
            break;
        }
        case NDA_REAL32: {
            const float *data = (const float*)A->data;
            float max_val = data[0];
            for (size_t i = 1; i < size; ++i) {
                if (data[i] > max_val) {
                    max_val = data[i];
                    max_idx = i;
                }
            }
            break;
        }
        case NDA_COMPLEX64:
        case NDA_COMPLEX32:
            // For complex types, use real part magnitude
            const double *complex_data = (const double*)A->data;
            double max_val = fabs(complex_data[0]);  // Real part of first element
            for (size_t i = 1; i < size; ++i) {
                if (fabs(complex_data[i * 2]) > max_val) {
                    max_val = fabs(complex_data[i * 2]);
                    max_idx = i;
                }
            }
            break;
    }
    return max_idx;
}

size_t ndarray_argmin(const NDArray A) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    return aggr_full_argmin(A);
}

size_t ndarray_argmax(const NDArray A) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    return aggr_full_argmax(A);
}