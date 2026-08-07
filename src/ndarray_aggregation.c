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
    OMP_PRAGMA(omp parallel for reduction(+:acc))
    for (size_t i = 0; i < size; ++i) {
        acc += A->data[i];
    }
    return (aggr_type == NDA_AGGR_MEAN) ? acc / size : acc;
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
        result->data[i] = (aggr_type == NDA_AGGR_MEAN) 
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
        // Use Welford's algorithm
        double mean = 0.0;
        double m2 = 0.0;
        
        for (size_t j = 0; j < axis_dim; ++j) {
            size_t idx = outer_idx * (axis_dim * stride) + j * stride + inner_idx;
            double delta = A->data[idx] - mean;
            mean += delta / (double)(j + 1);
            double delta2 = A->data[idx] - mean;
            m2 += delta * delta2;
        }
        
        result->data[i] = sqrt(m2 / axis_dim);
    }
}

NDArray ndarray_new_aggr(NDArray A, int axis, int aggr_type) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    assert((axis == -1 || (axis >= 0 && axis < (int)A->ndim)) 
            && "axis must be in range [0, ndim-1] or -1 (NDA_ALL_AXES) for all axes");
    if (axis == -1) {
        size_t dims[] = {1, 1, 0};
        NDArray result = ndarray_new(dims);
        switch (aggr_type) {
            case NDA_AGGR_SUM:
            case NDA_AGGR_MEAN:
                result->data[0] = aggr_full_sum_mean(A, aggr_type);
                break;
            case NDA_AGGR_MAX:
                result->data[0] = aggr_full_max(A);
                break;
            case NDA_AGGR_MIN:
                result->data[0] = aggr_full_min(A);
                break;
            case NDA_AGGR_STD:
                result->data[0] = aggr_full_std(A);
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
    NDArray result = ndarray_new(result_dims);
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

double ndarray_scalar_covariance(const NDArray A, const NDArray B) {
    assert(A != NULL && B != NULL && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2 && "ndarrays must have at least 2 dimensions");
    size_t size_a = ndarray_size(A);
    size_t size_b = ndarray_size(B);
    assert(size_a == size_b && "ndarrays must have the same size");
    assert(size_a >= 2 && "ndarrays must have at least 2 elements");
    double mean_a = 0.0, mean_b = 0.0;
    OMP_PRAGMA(omp parallel for reduction(+:mean_a) reduction(+:mean_b))
    for (size_t i = 0; i < size_a; ++i) {
        mean_a += A->data[i];
        mean_b += B->data[i];
    }
    mean_a /= size_a;
    mean_b /= size_b;
    double cov = 0.0;
    OMP_PRAGMA(omp parallel for reduction(+:cov))
    for (size_t i = 0; i < size_a; ++i) {
        cov += (A->data[i] - mean_a) * (B->data[i] - mean_b);
    }
    return cov / (size_a - 1);
}

double ndarray_scalar_correlation(const NDArray A, const NDArray B) {
    assert(A != NULL && B != NULL && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2 && "ndarrays must have at least 2 dimensions");
    size_t size_a = ndarray_size(A);
    size_t size_b = ndarray_size(B);
    assert(size_a == size_b && "ndarrays must have the same size");
    assert(size_a >= 2 && "ndarrays must have at least 2 elements");
    double mean_a = 0.0, mean_b = 0.0;
    OMP_PRAGMA(omp parallel for reduction(+:mean_a) reduction(+:mean_b))
    for (size_t i = 0; i < size_a; ++i) {
        mean_a += A->data[i];
        mean_b += B->data[i];
    }
    mean_a /= size_a;
    mean_b /= size_b;
    double cov = 0.0, var_a = 0.0, var_b = 0.0;
    OMP_PRAGMA(omp parallel for reduction(+:cov) reduction(+:var_a) reduction(+:var_b))
    for (size_t i = 0; i < size_a; ++i) {
        double da = A->data[i] - mean_a;
        double db = B->data[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    double denom = sqrt(var_a * var_b);
    return (denom == 0.0) ? 0.0 : cov / denom;
}

NDArray ndarray_new_covariance(const NDArray A) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    size_t n = A->dims[0];
    size_t p = A->dims[1];
    assert(n >= 2 && "need at least 2 observations");
    NDArray result = ndarray_new_zeros(NDA_DIMS(p, p));
    size_t stride = compute_stride(A, 0);
    OMP_PRAGMA(omp parallel for collapse(2))
    for (size_t j = 0; j < p; ++j) {
        for (size_t k = j; k < p; ++k) {
            double mean_j = 0.0, mean_k = 0.0;
            for (size_t i = 0; i < n; ++i) {
                mean_j += A->data[i * stride + j];
                mean_k += A->data[i * stride + k];
            }
            mean_j /= n;
            mean_k /= n;

            double cov = 0.0;
            for (size_t i = 0; i < n; ++i) {
                cov += (A->data[i * stride + j] - mean_j) *
                       (A->data[i * stride + k] - mean_k);
            }
            cov /= (n - 1);
            result->data[j * p + k] = cov;
            result->data[k * p + j] = cov;
        }
    }
    return result;
}

NDArray ndarray_new_correlation(const NDArray A) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    size_t n = A->dims[0];
    size_t p = A->dims[1];
    assert(n >= 2 && "need at least 2 observations");
    NDArray result = ndarray_new_zeros(NDA_DIMS(p, p));
    size_t stride = compute_stride(A, 0);
    OMP_PRAGMA(omp parallel for collapse(2))
    for (size_t j = 0; j < p; ++j) {
        for (size_t k = j; k < p; ++k) {
            double mean_j = 0.0, mean_k = 0.0;
            for (size_t i = 0; i < n; ++i) {
                mean_j += A->data[i * stride + j];
                mean_k += A->data[i * stride + k];
            }
            mean_j /= n;
            mean_k /= n;

            double cov = 0.0, var_j = 0.0, var_k = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double dj = A->data[i * stride + j] - mean_j;
                double dk = A->data[i * stride + k] - mean_k;
                cov += dj * dk;
                var_j += dj * dj;
                var_k += dk * dk;
            }
            double denom = sqrt(var_j * var_k);
            double corr = (denom == 0.0) ? 0.0 : cov / denom;
            result->data[j * p + k] = corr;
            result->data[k * p + j] = corr;
        }
    }
    return result;
}
