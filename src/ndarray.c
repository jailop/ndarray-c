/**
 * N-dimensional array implementation with OpenMP optimization
 * 
 * All ndarrays must have ndim >= 2
 * 
 * This implementation uses assert() for input validation.
 * - In debug builds: assertions are enabled and will abort on
 *   violation
 * - In release builds: compile with -DNDEBUG to disable assertions
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "ndarray.h"

#define BLOCK_SIZE 64

#ifdef USE_OPENMP
    #include <omp.h>
    #define OMP_PRAGMA(x) _Pragma(#x)
#else
    #define OMP_PRAGMA(x)
#endif

/**
 * Generate a Gaussian random number using Box-Muller transform
 * with given mean and standard deviation
 */
static double generate_gaussian(double mean, double std) {
    static int have_spare = 0;
    static double spare;
    if (have_spare) {
        have_spare = 0;
        return mean + std * spare;
    }
    have_spare = 1;
    double u, v, s;
    do {
        u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + std * (u * s);
}

static size_t ndarray_size(NDArray t) {
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

// Helper function for recursive printing of multidimensional arrays
static void print_recursive_helper(NDArray arr, size_t *indices, int precision,
                                    size_t depth, size_t max_items_per_dim) {
    size_t dim_size = arr->dims[depth];
    int should_truncate = (dim_size > 2 * max_items_per_dim);
    if (depth == arr->ndim - 1) {
        // Print innermost dimension
        printf("[");
        for (size_t i = 0; i < dim_size; ++i) {
            if (should_truncate && i == max_items_per_dim) {
                printf("...");
                i = dim_size - max_items_per_dim - 1;
                continue;
            }
            indices[depth] = i;
            printf("%.*f", precision, ndarray_get(arr, indices));
            if (i < dim_size - 1 && 
                !(should_truncate && i == dim_size - max_items_per_dim - 1)) {
                printf(", ");
            }
        }
        printf("]");
    } else {
        printf("[");
        for (size_t i = 0; i < dim_size; ++i) {
            if (should_truncate && i == max_items_per_dim) {
                if (i > 0) {
                    printf("\n");
                    for (size_t d = 0; d <= depth; ++d) printf(" ");
                }
                printf("...");
                i = dim_size - max_items_per_dim - 1;
                continue;
            }
            
            indices[depth] = i;
            if (i > 0) {
                printf("\n");
                for (size_t d = 0; d <= depth; ++d) printf(" ");
            }
            print_recursive_helper(arr, indices, precision, depth + 1,
                    max_items_per_dim);
            if (i < dim_size - 1
                    && !(should_truncate
                    && i == dim_size - max_items_per_dim - 1)) {
                printf(",");
            }
        }
        printf("]");
    }
}

void ndarray_print(NDArray arr, const char *name, int precision) {
    assert(arr != NULL && "ndarray cannot be NULL");
    
    if (precision < 0) precision = 4;
    
    // Get terminal width
    struct winsize w;
    int term_width = 80; // Default fallback
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) != -1 && w.ws_col > 0) {
        term_width = w.ws_col;
    }
    
    // Print header
    if (name != NULL) {
        printf("Array '%s' [", name);
    } else {
        printf("Array [");
    }
    for (size_t i = 0; i < arr->ndim; ++i) {
        printf("%zu%s", arr->dims[i], i < arr->ndim - 1 ? ", " : "");
    }
    printf("]:\n");
    
    if (arr->ndim == 2) {
        // 2D: Pretty matrix format with smart truncation
        size_t rows = arr->dims[0];
        size_t cols = arr->dims[1];
        
        int elem_width = precision + 6;
        int available_width = term_width - 6; // Account for brackets and spaces
        size_t max_cols = available_width / (elem_width + 1);
        
        // Decide how many rows/cols to show
        size_t show_rows_head = 3, show_rows_tail = 3;
        size_t show_cols_head = 3, show_cols_tail = 3;
        
        int truncate_rows = (rows > show_rows_head + show_rows_tail + 1);
        int truncate_cols = (cols > max_cols);
        
        if (truncate_cols && max_cols > 6) {
            show_cols_head = max_cols / 2;
            show_cols_tail = max_cols - show_cols_head;
        } else if (!truncate_cols) {
            show_cols_head = cols;
            show_cols_tail = 0;
        }
        
        printf("[");
        for (size_t i = 0; i < rows; ++i) {
            if (truncate_rows && i == show_rows_head) {
                if (i > 0) printf(" ");
                printf("...\n");
                i = rows - show_rows_tail - 1;
                continue;
            }
            
            if (i > 0) printf(" ");
            printf("[");
            
            for (size_t j = 0; j < cols; ++j) {
                if (truncate_cols && j == show_cols_head) {
                    printf("  ...");
                    j = cols - show_cols_tail - 1;
                    continue;
                }
                
                size_t pos[] = {i, j};
                printf("%*.*f", elem_width, precision, ndarray_get(arr, pos));
                if (j < cols - 1 && !(truncate_cols && j == cols - show_cols_tail - 1)) {
                    printf(" ");
                }
            }
            printf("]");
            if (i < rows - 1 && !(truncate_rows && i == show_rows_head - 1)) {
                printf("\n");
            }
        }
        printf("]\n");
    } else {
        // 3D+: Nested bracket notation with truncation
        size_t indices[arr->ndim];
        for (size_t i = 0; i < arr->ndim; ++i) {
            indices[i] = 0;
        }
        size_t max_items_per_dim = 3;
        
        print_recursive_helper(arr, indices, precision, 0, max_items_per_dim);
        printf("\n");
    }
}

NDArray ndarray_copy(NDArray t) {
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
    memcpy(copy->data, t->data, sizeof(double) * size);
    return copy;
}


NDArray ndarray_new_zeros(size_t *dims) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    memset(t->data, 0, sizeof(double) * size);
    return t;
}

NDArray ndarray_new_ones(size_t *dims) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        t->data[i] = 1.0;
    }
    return t;
}

NDArray ndarray_new_full(size_t *dims, double value) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        t->data[i] = value;
    }
    return t;
}

NDArray ndarray_new_arange(size_t *dims, double start, double stop,
        double step) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    for (size_t i = 0; i < size; ++i) {
        double val = start + i * step;
        if (val >= stop) break;
        t->data[i] = val;
    }
    return t;
}

NDArray ndarray_new_linspace(size_t *dims, double start, double stop,
        size_t num) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    if (num <= 1) {
        t->data[0] = start;
        return t;
    }
    double step = (stop - start) / (num - 1);
    size_t max_idx = (size < num) ? size : num;
    OMP_PRAGMA(omp parallel for)
    for (size_t i = 0; i < max_idx; ++i) {
        t->data[i] = start + i * step;
    }
    return t;
}

NDArray ndarray_new_randnorm(size_t *dims, double mean, double stddev) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    for (size_t i = 0; i < size; ++i) {
        t->data[i] = generate_gaussian(mean, stddev);
    }
    return t;
}

NDArray ndarray_new_randunif(size_t *dims, double low, double high) {
    NDArray t = ndarray_new(dims);
    size_t size = ndarray_size(t);
    for (size_t i = 0; i < size; ++i) {
        t->data[i] = low + (high - low) * ((double)rand() / RAND_MAX);
    }
    return t;
}

NDArray ndarray_add(NDArray A, NDArray B) {
    assert(A != NULL && B != NULL && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    assert(A->ndim == B->ndim
            && "ndarrays must have same number of dimensions");
    for (size_t i = 0; i < A->ndim; ++i) {
        assert(A->dims[i] == B->dims[i]
                && "ndarrays must have matching dimensions");
    }
    size_t size = ndarray_size(A);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        A->data[i] += B->data[i];
    }
    return A;
}

NDArray ndarray_mul(NDArray A, NDArray B) {
    assert(A != NULL && B != NULL
            && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    assert(A->ndim == B->ndim
            && "ndarrays must have same number of dimensions");
    for (size_t i = 0; i < A->ndim; ++i) {
        assert(A->dims[i] == B->dims[i]
                && "ndarrays must have matching dimensions");
    }
    size_t size = ndarray_size(A);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        A->data[i] *= B->data[i];
    }
    return A;
}

NDArray ndarray_add_scalar(NDArray A, double scalar) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    size_t size = ndarray_size(A);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        A->data[i] += scalar;
    }
    return A;
}

NDArray ndarray_mul_scalar(NDArray A, double scalar) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    size_t size = ndarray_size(A);
    OMP_PRAGMA(omp parallel for simd)
    for (size_t i = 0; i < size; ++i) {
        A->data[i] *= scalar;
    }
    return A;
}

NDArray ndarray_mapfnc(NDArray A, double (*func)(double)) {
    assert(A != NULL && "ndarray cannot be NULL");
    assert(A->ndim >= 2 && "ndarray must have at least 2 dimensions");
    assert(func != NULL && "function pointer cannot be NULL");
    size_t size = ndarray_size(A);
    OMP_PRAGMA(omp parallel for)
    for (size_t i = 0; i < size; ++i) {
        A->data[i] = func(A->data[i]);
    }
    return A;
}

static void matmul_2d_blocked(double *C, double *A, double *B, 
                              size_t m, size_t n, size_t p) {
    for (size_t i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
        for (size_t j0 = 0; j0 < p; j0 += BLOCK_SIZE) {
            for (size_t k0 = 0; k0 < n; k0 += BLOCK_SIZE) {
                size_t i_max = (i0 + BLOCK_SIZE < m) ? i0 + BLOCK_SIZE : m;
                size_t j_max = (j0 + BLOCK_SIZE < p) ? j0 + BLOCK_SIZE : p;
                size_t k_max = (k0 + BLOCK_SIZE < n) ? k0 + BLOCK_SIZE : n;
                for (size_t i = i0; i < i_max; ++i) {
                    for (size_t k = k0; k < k_max; ++k) {
                        double a_ik = A[i * n + k];
                        OMP_PRAGMA(omp simd)
                        for (size_t j = j0; j < j_max; ++j) {
                            C[i * p + j] += a_ik * B[k * p + j];
                        }
                    }
                }
            }
        }
    }
}

NDArray ndarray_new_tensordot(NDArray A, NDArray B, 
                               int *axes_a, int *axes_b) {
    assert(A != NULL && B != NULL && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    // Count axes (terminated by -1)
    int n_axes = 0;
    if (axes_a != NULL && axes_b != NULL) {
        while (axes_a[n_axes] != -1) {
            assert(axes_b[n_axes] != -1
                    && "axes_a and axes_b must have same length");
            n_axes++;
        }
        assert(axes_b[n_axes] == -1
                && "axes_a and axes_b must have same length");
    }
    // Validate axes and check dimension compatibility
    for (int i = 0; i < n_axes; ++i) {
        assert(axes_a[i] >= 0 && axes_a[i] < (int)A->ndim && 
               "axes_a contains invalid axis");
        assert(axes_b[i] >= 0 && axes_b[i] < (int)B->ndim && 
               "axes_b contains invalid axis");
        assert(A->dims[axes_a[i]] == B->dims[axes_b[i]] && 
               "contracted axes must have matching dimensions");
    }
    // Mark which axes are being contracted
    int *a_contracted = (int*)calloc(A->ndim, sizeof(int));
    int *b_contracted = (int*)calloc(B->ndim, sizeof(int));
    for (int i = 0; i < n_axes; ++i) {
        a_contracted[axes_a[i]] = 1;
        b_contracted[axes_b[i]] = 1;
    }
    // Build result dimensions: free axes from A, then free axes from B
    size_t result_ndim = (A->ndim - n_axes) + (B->ndim - n_axes);
    if (result_ndim < 2) result_ndim = 2;
    size_t *result_dims = (size_t*)malloc(sizeof(size_t) * (result_ndim + 1));
    size_t idx = 0;
    // Add free axes from A
    for (size_t i = 0; i < A->ndim; ++i) {
        if (!a_contracted[i]) {
            result_dims[idx++] = A->dims[i];
        }
    }
    // Add free axes from B
    for (size_t i = 0; i < B->ndim; ++i) {
        if (!b_contracted[i]) {
            result_dims[idx++] = B->dims[i];
        }
    }
    // Pad to ensure ndim >= 2
    while (idx < result_ndim) {
        result_dims[idx++] = 1;
    }
    result_dims[idx] = 0;
    NDArray C = ndarray_new_zeros(result_dims);
    // Compute sizes for iteration
    size_t contract_size = 1;
    for (int i = 0; i < n_axes; ++i) {
        contract_size *= A->dims[axes_a[i]];
    }
    size_t a_free_size = 1;
    for (size_t i = 0; i < A->ndim; ++i) {
        if (!a_contracted[i]) a_free_size *= A->dims[i];
    }
    size_t b_free_size = 1;
    for (size_t i = 0; i < B->ndim; ++i) {
        if (!b_contracted[i]) b_free_size *= B->dims[i];
    }
    // Compute strides for A, B, and C
    size_t *a_strides = (size_t*)malloc(sizeof(size_t) * A->ndim);
    size_t *b_strides = (size_t*)malloc(sizeof(size_t) * B->ndim);
    size_t *c_strides = (size_t*)malloc(sizeof(size_t) * C->ndim);
    a_strides[A->ndim - 1] = 1;
    for (int i = A->ndim - 2; i >= 0; --i) {
        a_strides[i] = a_strides[i + 1] * A->dims[i + 1];
    }
    b_strides[B->ndim - 1] = 1;
    for (int i = B->ndim - 2; i >= 0; --i) {
        b_strides[i] = b_strides[i + 1] * B->dims[i + 1];
    }
    c_strides[C->ndim - 1] = 1;
    for (int i = C->ndim - 2; i >= 0; --i) {
        c_strides[i] = c_strides[i + 1] * C->dims[i + 1];
    }
    // Perform contraction
    size_t result_size = a_free_size * b_free_size;
    OMP_PRAGMA(omp parallel)
    {
        size_t *a_idx = (size_t*)malloc(sizeof(size_t) * A->ndim);
        size_t *b_idx = (size_t*)malloc(sizeof(size_t) * B->ndim);
        size_t *c_idx = (size_t*)malloc(sizeof(size_t) * C->ndim);
        OMP_PRAGMA(omp for)
        for (size_t out_idx = 0; out_idx < result_size; ++out_idx) {
            // Decode output index
            size_t temp = out_idx;
            idx = 0;
            // Get indices for free axes from A
            for (size_t i = 0; i < A->ndim; ++i) {
                if (!a_contracted[i]) {
                    size_t dim = A->dims[i];
                    size_t stride = 1;
                    for (size_t j = i + 1; j < A->ndim; ++j) {
                        if (!a_contracted[j]) stride *= A->dims[j];
                    }
                    for (size_t j = 0; j < B->ndim; ++j) {
                        if (!b_contracted[j]) stride *= B->dims[j];
                    }
                    a_idx[i] = (temp / stride) % dim;
                }
            }
            // Get indices for free axes from B
            for (size_t i = 0; i < B->ndim; ++i) {
                if (!b_contracted[i]) {
                    size_t dim = B->dims[i];
                    size_t stride = 1;
                    for (size_t j = i + 1; j < B->ndim; ++j) {
                        if (!b_contracted[j]) stride *= B->dims[j];
                    }
                    b_idx[i] = (temp / stride) % dim;
                }
            }
            // Perform contraction sum
            double sum = 0.0;
            for (size_t contract_idx = 0; contract_idx < contract_size;
                    ++contract_idx) {
                // Decode contraction indices
                size_t ctemp = contract_idx;
                for (int i = n_axes - 1; i >= 0; --i) {
                    size_t dim = A->dims[axes_a[i]];
                    a_idx[axes_a[i]] = ctemp % dim;
                    b_idx[axes_b[i]] = ctemp % dim;
                    ctemp /= dim;
                }
                // Compute flat indices
                size_t a_flat = 0;
                for (size_t i = 0; i < A->ndim; ++i) {
                    a_flat += a_idx[i] * a_strides[i];
                }
                size_t b_flat = 0;
                for (size_t i = 0; i < B->ndim; ++i) {
                    b_flat += b_idx[i] * b_strides[i];
                }
                sum += A->data[a_flat] * B->data[b_flat];
            }
            C->data[out_idx] = sum;
        }
        free(a_idx);
        free(b_idx);
        free(c_idx);
    }
    free(a_contracted);
    free(b_contracted);
    free(result_dims);
    free(a_strides);
    free(b_strides);
    free(c_strides);
    return C;
}

NDArray ndarray_new_matmul(NDArray A, NDArray B) {
    assert(A != NULL && B != NULL && "ndarrays cannot be NULL");
    assert(A->ndim >= 2 && B->ndim >= 2
            && "ndarrays must have at least 2 dimensions");
    size_t a_rows = A->dims[A->ndim - 2];
    size_t a_cols = A->dims[A->ndim - 1];
    size_t b_rows = B->dims[B->ndim - 2];
    size_t b_cols = B->dims[B->ndim - 1];
    assert(a_cols == b_rows
            && "matrix dimensions incompatible for multiplication");
    // Simple case: both 2D matrices
    if (A->ndim == 2 && B->ndim == 2) {
        size_t dims[] = {a_rows, b_cols, 0};
        NDArray C = ndarray_new_zeros(dims);
        OMP_PRAGMA(omp parallel)
        {
            OMP_PRAGMA(omp single)
            matmul_2d_blocked(C->data, A->data, B->data, a_rows, a_cols, b_cols);
        }
        return C;
    }
    // Batch matmul: determine result shape
    size_t max_ndim = (A->ndim > B->ndim) ? A->ndim : B->ndim;
    size_t *result_dims = (size_t*)malloc(sizeof(size_t) * (max_ndim + 1));
    // Broadcast batch dimensions
    for (size_t i = 0; i < max_ndim - 2; ++i) {
        size_t a_idx = (A->ndim - 2 > i) ? A->ndim - 2 - i - 1 : 0;
        size_t b_idx = (B->ndim - 2 > i) ? B->ndim - 2 - i - 1 : 0;
        size_t a_dim = (a_idx < A->ndim - 2) ? A->dims[a_idx] : 1;
        size_t b_dim = (b_idx < B->ndim - 2) ? B->dims[b_idx] : 1;
        assert((a_dim == b_dim || a_dim == 1 || b_dim == 1) && 
               "batch dimensions must be compatible for broadcasting");
        result_dims[max_ndim - 2 - i - 1] = (a_dim > b_dim) ? a_dim : b_dim;
    }
    result_dims[max_ndim - 2] = a_rows;
    result_dims[max_ndim - 1] = b_cols;
    result_dims[max_ndim] = 0;
    NDArray C = ndarray_new_zeros(result_dims);
    free(result_dims);
    // Calculate batch sizes
    size_t batch_size = 1;
    for (size_t i = 0; i < C->ndim - 2; ++i) {
        batch_size *= C->dims[i];
    }
    size_t a_batch_stride = a_rows * a_cols;
    size_t b_batch_stride = b_rows * b_cols;
    size_t c_batch_stride = a_rows * b_cols;
    size_t a_batch_size = 1;
    for (size_t i = 0; i < A->ndim - 2; ++i) {
        a_batch_size *= A->dims[i];
    }
    size_t b_batch_size = 1;
    for (size_t i = 0; i < B->ndim - 2; ++i) {
        b_batch_size *= B->dims[i];
    }
    OMP_PRAGMA(omp parallel for)
    for (size_t batch = 0; batch < batch_size; ++batch) {
        size_t a_batch_idx = (a_batch_size == 1) ? 0 : (batch % a_batch_size);
        size_t b_batch_idx = (b_batch_size == 1) ? 0 : (batch % b_batch_size);
        double *a_ptr = A->data + a_batch_idx * a_batch_stride;
        double *b_ptr = B->data + b_batch_idx * b_batch_stride;
        double *c_ptr = C->data + batch * c_batch_stride;
        matmul_2d_blocked(c_ptr, a_ptr, b_ptr, a_rows, a_cols, b_cols);
    }
    
    return C;
}

NDArray ndarray_stack(int axis, NDArray* arr_list) {
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

NDArray ndarray_concat(int axis, NDArray* arr_list) {
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

NDArray ndarray_take(NDArray arr, int axis, size_t start, size_t end) {
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

static double aggr_full_sum_mean(NDArray A, int aggr_type) {
    size_t size = ndarray_size(A);
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

static size_t compute_stride(NDArray A, int axis) {
    size_t stride = 1;
    for (int i = axis + 1; i < (int)A->ndim; ++i) {
        stride *= A->dims[i];
    }
    return stride;
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
            && "axis out of bounds");
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
