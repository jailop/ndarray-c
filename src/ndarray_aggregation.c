/**
 * Aggregation operations
 */

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
    // Use CBLAS dasum for sum of absolute values or manual sum
    // For sum, we use CBLAS ddot with a vector of ones
    double acc = 0.0;
    OMP_PRAGMA(omp parallel for reduction(+:acc))
    for (size_t i = 0; i < size; ++i) {
        acc += A->data[i];
    }
    return (aggr_type == NDARRAY_AGGR_MEAN) ? acc / size : acc;
}

static double aggr_full_max(NDArray A) {
    size_t size = ndarray_size(A);
    double acc = A->data[0];
    OMP_PRAGMA(omp parallel for reduction(max:acc))
    for (size_t i = 1; i < size; ++i) {
        if (A->data[i] > acc) acc = A->data[i];
    }
    return acc;
}

static double aggr_full_min(NDArray A) {
    size_t size = ndarray_size(A);
    double acc = A->data[0];
    OMP_PRAGMA(omp parallel for reduction(min:acc))
    for (size_t i = 1; i < size; ++i) {
        if (A->data[i] < acc) acc = A->data[i];
    }
    return acc;
}

static double aggr_full_std(NDArray A) {
    size_t size = ndarray_size(A);
    double mean = 0.0;
    OMP_PRAGMA(omp parallel for reduction(+:mean))
    for (size_t i = 0; i < size; ++i) {
        mean += A->data[i];
    }
    mean /= size;
    double variance = 0.0;
    OMP_PRAGMA(omp parallel for reduction(+:variance))
    for (size_t i = 0; i < size; ++i) {
        double diff = A->data[i] - mean;
        variance += diff * diff;
    }
    return sqrt(variance / size);
}

static void aggr_axis_sum_mean(NDArray result, NDArray A, int axis,
        int aggr_type) {
    size_t result_size = ndarray_size(result);
    size_t axis_dim = A->dims[axis];
    size_t stride = compute_stride(A, axis);
    memset(result->data, 0, sizeof(double) * result_size);
    OMP_PRAGMA(omp parallel for)
    for (size_t i = 0; i < result_size; ++i) {
        size_t outer_idx = i / stride;
        size_t inner_idx = i % stride;
        double sum = 0.0;
        for (size_t j = 0; j < axis_dim; ++j) {
            size_t idx = outer_idx * (axis_dim * stride) + j * stride
                + inner_idx;
            sum += A->data[idx];
        }
        result->data[i] = (aggr_type == NDARRAY_AGGR_MEAN) 
            ? sum / axis_dim 
            : sum;
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
        size_t idx = outer_idx * (axis_dim * stride) + inner_idx;
        double max_val = A->data[idx];
        for (size_t j = 1; j < axis_dim; ++j) {
            idx = outer_idx * (axis_dim * stride) + j * stride + inner_idx;
            if (A->data[idx] > max_val) {
                max_val = A->data[idx];
            }
        }
        result->data[i] = max_val;
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
        size_t idx = outer_idx * (axis_dim * stride) + inner_idx;
        double min_val = A->data[idx];
        for (size_t j = 1; j < axis_dim; ++j) {
            idx = outer_idx * (axis_dim * stride) + j * stride + inner_idx;
            if (A->data[idx] < min_val) {
                min_val = A->data[idx];
            }
        }
        result->data[i] = min_val;
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
        double mean = 0.0;
        for (size_t j = 0; j < axis_dim; ++j) {
            size_t idx = outer_idx * (axis_dim * stride) + j * stride + inner_idx;
            mean += A->data[idx];
        }
        mean /= axis_dim;
        double variance = 0.0;
        for (size_t j = 0; j < axis_dim; ++j) {
            size_t idx = outer_idx * (axis_dim * stride) + j * stride + inner_idx;
            double diff = A->data[idx] - mean;
            variance += diff * diff;
        }
        result->data[i] = sqrt(variance / axis_dim);
    }
}

NDArray ndarray_new_axis_aggr(NDArray A, int axis, int aggr_type) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    assert((axis == -1 || (axis >= 0 && axis < (int)A->ndim)) 
            && "axis must be in range [0, ndim-1] or -1 (NDA_AXES_ALL) for all axes");
    if (axis == -1) {
        size_t dims[] = {1, 1, 0};
        NDArray result = ndarray_new(dims);
        switch (aggr_type) {
            case NDARRAY_AGGR_SUM:
            case NDARRAY_AGGR_MEAN:
                result->data[0] = aggr_full_sum_mean(A, aggr_type);
                break;
            case NDARRAY_AGGR_MAX:
                result->data[0] = aggr_full_max(A);
                break;
            case NDARRAY_AGGR_MIN:
                result->data[0] = aggr_full_min(A);
                break;
            case NDARRAY_AGGR_STD:
                result->data[0] = aggr_full_std(A);
                break;
        }
        return result;
    }
    // Compute result dimensions
    size_t result_ndim = A->ndim - 1;
    if (result_ndim < 2) {
        result_ndim = 2;
    }
    size_t result_dims[result_ndim + 1];
    size_t idx = 0;
    // Build result dimensions, inserting 1 for aggregated axis if needed
    for (size_t i = 0; i < A->ndim; ++i) {
        if ((int)i == axis) {
            // Insert dimension of 1 for the aggregated axis
            if (result_ndim == 2 && A->ndim == 2) {
                result_dims[idx++] = 1;
            }
        } else {
            result_dims[idx++] = A->dims[i];
        }
    }
    // If we still don't have 2 dimensions, pad at the end
    while (idx < result_ndim) {
        result_dims[idx++] = 1;
    }
    result_dims[idx] = 0;
    NDArray result = ndarray_new(result_dims);
    switch (aggr_type) {
        case NDARRAY_AGGR_SUM:
        case NDARRAY_AGGR_MEAN:
            aggr_axis_sum_mean(result, A, axis, aggr_type);
            break;
        case NDARRAY_AGGR_MAX:
            aggr_axis_max(result, A, axis);
            break;
        case NDARRAY_AGGR_MIN:
            aggr_axis_min(result, A, axis);
            break;
        case NDARRAY_AGGR_STD:
            aggr_axis_std(result, A, axis);
            break;
        default:
            assert(0 && "invalid aggregation type");
    }
    return result;
}
