#ifndef _NDARRAY_H
#define _NDARRAY_H

#include <stddef.h>

/**
 * All ndarrays in this library must have ndim >= 2.
 * 1D arrays are not supported - use 2D arrays with one dimension set to 1.
 */

/**
 * Error codes for ndarray operations.
 */
#define NDA_SUCCESS           0   /**< Operation completed successfully */
#define NDA_ERROR_NULL        -1  /**< NULL pointer argument */
#define NDA_ERROR_AXIS        -2  /**< Axis out of range */
#define NDA_ERROR_INDEX       -3  /**< Index out of bounds */
#define NDA_ERROR_SIZE        -4  /**< Size mismatch */

/**
 * An structure to represent and operate over
 * multi-dimensional arrays of doubles.
 * 
 * This structure contains:
 *
 * - data: pointer to the array elements stored in row-major order
 * - dims: array of dimension sizes
 * - ndim: number of dimensions (must be >= 2)
 * - stride: element stride for linear access (GSL compatibility)
 * - tda: trailing dimension (physical row width for 2D matrices)
 * - owner: ownership flag (1 if this struct owns the data block)
 */
typedef struct {
    double *data;
    size_t *dims;
    size_t ndim;
    size_t stride;    /* stride for 1D/vector access (default: 1) */
    size_t tda;       /* trailing dimension for 2D matrices (default: dims[1] or 1) */
    int owner;        /* ownership flag: 1 if owns data, 0 if borrowed */
} NDArray_;

/**
 * A handle to a ndarray structure.
 * 
 * It is expected that users will interact with ndarrays
 * through this handle rather than directly manipulating
 * the underlying structure. All ndarray functions accept
 * and return this handle type.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new(NDA_DIMS(3, 4));
 * ndarray_set(arr, NDA_POS(0, 0), 5.0);
 * ndarray_free(arr);
 * ```
 */
typedef NDArray_* NDArray;

/**
 * Macro to create a dimensions array for ndarray creation.
 * 
 * The last element (0) is automatically added as a sentinel to indicate
 * the end of the dimensions.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr2d = ndarray_new(NDA_DIMS(3, 4));        // 3x4 matrix
 * NDArray arr3d = ndarray_new(NDA_DIMS(2, 3, 4));     // 2x3x4 tensor
 * NDArray arr4d = ndarray_new(NDA_DIMS(2, 2, 3, 2));  // 2x2x3x2 tensor
 * ```
 */
#define NDA_DIMS(...) ((size_t[]){__VA_ARGS__, 0})

/**
 * Macro to create a position array for ndarray access.
 * 
 * Used with ndarray_set() and ndarray_get() to specify element positions.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new(NDA_DIMS(3, 4));
 * ndarray_set(arr, NDA_POS(0, 0), 5.0);  // Set element at (0,0)
 * double val = ndarray_get(arr, NDA_POS(2, 3));  // Get element at (2,3)
 * ```
 */
#define NDA_POS(...) ((size_t[]){__VA_ARGS__})

/**
 * Macro to create an axis array for tensor operations.
 * 
 * The last element (-1) is automatically added as a sentinel.
 * For no contraction (outer product), use NDA_NO_AXES instead.
 * 
 * Example:
 * 
 * ```c
 * // Contract on single axis
 * NDArray C = ndarray_new_tensordot(A, B, NDA_AXES(1), NDA_AXES(0));
 * 
 * // Contract on multiple axes
 * NDArray D = ndarray_new_tensordot(A, B, NDA_AXES(2, 3), NDA_AXES(0, 1));
 * ```
 */
#define NDA_AXES(...) ((int[]){__VA_ARGS__, -1})

/**
 * Macro to indicate no axes for tensor operations.
 * 
 * Used for outer products in tensor contractions.
 * 
 * Example:
 * 
 * ```c
 * // Outer product: A[i,j] ⊗ B[k,l] -> C[i,j,k,l]
 * NDArray C = ndarray_new_tensordot(A, B, NDA_NO_AXES, NDA_NO_AXES);
 * ```
 */
#define NDA_NO_AXES ((int[]){-1})

/**
 * Macro to create a list of ndarrays for functions
 * that accept multiple arrays.
 * 
 * The NULL sentinel is automatically added at the end.
 * 
 * Example:
 * 
 * ```c
 * // Stack multiple arrays
 * NDArray stacked = ndarray_new_stack(0, NDA_LIST(A, B, C));
 * 
 * // Concatenate arrays
 * NDArray concatenated = ndarray_new_concat(1, NDA_LIST(A, B, C));
 * 
 * // Free multiple arrays
 * ndarray_free_all(NDA_LIST(A, B, C));
 * ```
 */
#define NDA_LIST(...) ((NDArray[]){__VA_ARGS__, NULL})

/**
 * Constant to indicate operations on all axes.
 * 
 * Used with functions like ndarray_new_aggr to perform aggregation
 * over all dimensions at once.
 * 
 * Example:
 * 
 * ```c
 * // Sum all elements
 * NDArray total = ndarray_new_aggr(A, NDA_ALL_AXES, NDA_AGGR_SUM);
 * 
 * // Mean of all elements
 * NDArray avg = ndarray_new_aggr(A, NDA_ALL_AXES, NDA_AGGR_MEAN);
 * ```
 */
#define NDA_ALL_AXES (-1)

/**
 * Creates a new ndarray with the specified dimensions.  The dimensions
 * are provided as a variable number of arguments, using the last one as
 * a sentinel (0).  `ndim` must be >= 2 (at least two dimensions
 * required).
 *
 * @param dims Array of dimensions, ending with 0.
 * @return A handle to the newly created ndarray, or NULL if ndim < 2.
 *
 * Example:
 *
 * ```c
 * NDArray arr = ndarray_new(NDA_DIMS(3, 4));
 * // ...
 * ndarray_free(arr);
 * ```
 */
NDArray ndarray_new(const size_t *dims);

/**
 * Frees the memory allocated for the ndarray.
 * 
 * This releases all memory associated with the ndarray, including
 * the data array, dimensions array, and the structure itself.
 * After calling this function, the handle should not be used.
 * 
 * @param t The ndarray to free.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new(NDA_DIMS(3, 4));
 * // ... use arr ...
 * ndarray_free(arr);
 * ```
 */
void ndarray_free(NDArray t);

/**
 * Frees the memory allocated for multiple ndarrays.
 * 
 * The ndarrays are provided as a NULL-terminated array.
 * Use the NDA_LIST macro for convenience.
 * 
 * @param arr_list NULL-terminated list of arrays to free.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new(NDA_DIMS(3, 4));
 * NDArray B = ndarray_new(NDA_DIMS(3, 4));
 * NDArray C = ndarray_new(NDA_DIMS(3, 4));
 * // ... use arrays ...
 * ndarray_free_all(NDA_LIST(A, B, C));
 * ```
 */
void ndarray_free_all(NDArray* arr_list);

/**
 * Computes the offset in the data array for the given position.
 * 
 * Converts multi-dimensional indices to a linear offset in the
 * underlying data array using row-major (C-order) layout.
 * 
 * @param t The ndarray.
 * @param pos An array of size_t representing the position in each dimension.
 * @return The computed offset in the data array.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new(NDA_DIMS(3, 4));
 * size_t offset = ndarray_offset(arr, NDA_POS(1, 2));  // Returns 6
 * double val = arr->data[offset];  // Direct access (not recommended)
 * ```
 */
size_t ndarray_offset(const NDArray t, const size_t *pos);

/**
 * Sets the value at the specified position in the ndarray.
 * 
 * @param t The ndarray.
 * @param pos An array of size_t representing the position in each dimension.
 * @param value The value to set.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new(NDA_DIMS(3, 4));
 * ndarray_set(arr, NDA_POS(0, 0), 5.0);
 * ndarray_set(arr, NDA_POS(2, 3), 10.0);
 * ```
 */
void ndarray_set(const NDArray t, const size_t* pos, double value);

/**
 * Gets the value at the specified position in the ndarray.
 * 
 * @param t The ndarray.
 * @param pos An array of size_t representing the position in each dimension.
 * @return The value at the specified position.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new(NDA_DIMS(3, 4));
 * ndarray_set(arr, NDA_POS(1, 2), 7.5);
 * double val = ndarray_get(arr, NDA_POS(1, 2));  // Returns 7.5
 * ```
 */
double ndarray_get(const NDArray t, const size_t* pos);

/**
 * Sets values along a slice of the array at a specific index on an axis.
 * 
 * For a 2D array: axis=0 sets a row, axis=1 sets a column.
 * For higher dimensions, sets the hyperplane perpendicular to the axis.
 * 
 * @param arr The ndarray to modify.
 * @param axis The axis along which to set the slice (0 to ndim-1).
 * @param index The index along the axis where to set values.
 * @param values Array of values to set (size must match the slice size).
 * @return handler of the array (for chaining)
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new(NDA_DIMS(3, 4));
 * double row_data[] = {1.0, 2.0, 3.0, 4.0};
 * ndarray_set_slice(arr, 0, 0, row_data);  // Set first row
 * ```
 */
NDArray ndarray_set_slice(const NDArray arr, int axis, size_t index,
        const double* values);

/**
 * Fills a slice of the array with a scalar value at a specific index on an axis.
 * 
 * For a 2D array: axis=0 fills a row, axis=1 fills a column.
 * For higher dimensions, fills the hyperplane perpendicular to the axis.
 * 
 * @param arr The ndarray to modify.
 * @param axis The axis along which to fill the slice (0 to ndim-1).
 * @param index The index along the axis where to fill.
 * @param value The scalar value to fill with.
 * @return handler of the array (for chaining)
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new(NDA_DIMS(3, 4));
 * ndarray_fill_slice(arr, 0, 1, 5.0);  // Fill second row with 5.0
 * ndarray_fill_slice(arr, 1, 2, 0.0);  // Fill third column with 0.0
 * ```
 */
NDArray ndarray_fill_slice(const NDArray arr, int axis, size_t index,
        double value);

/**
 * Get pointer to a slice along an axis.
 * 
 * Returns a pointer to the beginning of the slice. User must not access
 * beyond the slice bounds. Pointer is valid as long as the array exists.
 * 
 * @param arr The ndarray.
 * @param axis The axis along which to get the slice.
 * @param index The index along the axis.
 * @return Pointer to the start of the slice data.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new(NDA_DIMS(3, 4));
 * double* row_ptr = ndarray_get_slice_ptr(arr, 0, 1);  // Pointer to row 1
 * row_ptr[0] = 1.0;  // Modify first element of row 1
 * // For 3D [2,3,4]: axis=1, index=1 -> pointer to middle plane (2*4=8 elements)
 * ```
 */
double* ndarray_get_slice_ptr(const NDArray arr, int axis, size_t index);

/**
 * Copy a slice from one array to another.
 * 
 * Copies data from a slice in the source array to a slice in the destination.
 * The slice sizes must match.
 * 
 * @param dst Destination ndarray.
 * @param dst_axis Axis in destination.
 * @param dst_idx Index along destination axis.
 * @param src Source ndarray.
 * @param src_axis Axis in source.
 * @param src_idx Index along source axis.
 * @return handler of the destination array (for chaining)
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new(NDA_DIMS(3, 4));
 * NDArray B = ndarray_new(NDA_DIMS(3, 4));
 * ndarray_copy_slice(B, 0, 2, A, 0, 0);  // Copy row 0 from A to row 2 of B
 * ```
 */
NDArray ndarray_copy_slice(const NDArray dst, int dst_axis, size_t dst_idx,
                           const NDArray src, int src_axis, size_t src_idx);

/**
 * Get the size of a slice along an axis.
 * 
 * Returns the number of elements in a slice perpendicular to the given axis.
 * 
 * @param arr The ndarray.
 * @param axis The axis.
 * @return Number of elements in a slice.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new(NDA_DIMS(3, 4));
 * size_t row_size = ndarray_get_slice_size(arr, 0);  // Returns 4
 * size_t col_size = ndarray_get_slice_size(arr, 1);  // Returns 3
 * // For 3D [2,3,4]: axis=1 -> 8 (2*4 elements per plane)
 * ```
 */
size_t ndarray_get_slice_size(const NDArray arr, int axis);

/**
 * Pretty-prints an ndarray to stdout.
 * 
 * Automatically formats output based on dimensionality:
 * - 2D: matrix format with aligned columns
 * - 3D+: nested bracket notation with proper indentation
 * 
 * @param arr The ndarray to print.
 * @param name Optional name to display (can be NULL).
 * @param precision Number of decimal places (default 4 if < 0).
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new(NDA_DIMS(3, 4));
 * ndarray_print(arr, "A", 4);
 * // Output:
 * // Array 'A' [3, 4]:
 * // [[  1.0000   2.0000   3.0000   4.0000]
 * //  [  5.0000   6.0000   7.0000   8.0000]
 * //  [  9.0000  10.0000  11.0000  12.0000]]
 * ```
 */
void ndarray_print(const NDArray arr, const char *name, int precision);

/**
 * Creates a copy of the given ndarray.
 * 
 * Allocates a new ndarray with the same dimensions and copies all data.
 * 
 * @param t The ndarray to copy.
 * @return A handle to the newly created copy of the ndarray.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new(NDA_DIMS(3, 4));
 * NDArray copy = ndarray_new_copy(arr);
 * ndarray_free_all(NDA_LIST(arr, copy));
 * ```
 */
NDArray ndarray_new_copy(const NDArray t);

/**
 * Creates a new ndarray filled with zeros.
 * 
 * @param dims Array of dimensions, ending with 0.
 * @return A handle to the newly created ndarray filled with zeros.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_zeros(NDA_DIMS(3, 4));
 * ndarray_free(arr);
 * ```
 */
NDArray ndarray_new_zeros(const size_t *dims);

/**
 * Creates a new ndarray from existing data.
 * 
 * The data is copied into the new ndarray.
 * Supports any number of dimensions (ndim >= 2).
 * The size is automatically calculated from the dimensions.
 * 
 * @param dims Array of dimensions, ending with 0.
 * @param data Pointer to the data array to copy.
 * @return A handle to the newly created ndarray.
 * 
 * Example:
 * 
 * ```c
 * // 2D array: 2x3
 * double data2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
 * NDArray a = ndarray_new_from_data(NDA_DIMS(2, 3), (double*)data2d);
 * 
 * // 3D array: 2x3x4
 * double data3d[2][3][4] = {...};
 * NDArray b = ndarray_new_from_data(NDA_DIMS(2, 3, 4), (double*)data3d);
 * 
 * ndarray_free_all(NDA_LIST(a, b));
 * ```
 */
NDArray ndarray_new_from_data(const size_t *dims, const double *data);

/**
 * Creates a new ndarray filled with ones.
 * 
 * @param dims Array of dimensions, ending with 0.
 * @return A handle to the newly created ndarray filled with ones.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_ones(NDA_DIMS(3, 4));
 * ndarray_free(arr);
 * ```
 */
NDArray ndarray_new_ones(const size_t *dims);

/**
 * Creates a new ndarray filled with the specified value.
 * 
 * @param dims Array of dimensions, ending with 0.
 * @param value The value to fill the ndarray with.
 * @return A handle to the newly created ndarray filled with the specified value.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), 5.0);  // All elements are 5.0
 * ndarray_free(arr);
 * ```
 */
NDArray ndarray_new_full(const size_t *dims, double value);

/**
 * Creates a new ndarray with values in the range [start, stop) with the given step.
 * 
 * Values are generated sequentially and filled in row-major order.
 * 
 * @param dims Array of dimensions, ending with 0.
 * @param start The starting value of the sequence.
 * @param stop The ending value of the sequence (exclusive).
 * @param step The step size between values.
 * @return A handle to the newly created ndarray with the specified range of values.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_arange(NDA_DIMS(2, 5), 0.0, 10.0, 1.0);
 * // Creates [2,5] array with values [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 * ndarray_free(arr);
 * ```
 */
NDArray ndarray_new_arange(const size_t *dims, double start, double stop,
        double step);

/**
 * Creates a new ndarray with linearly spaced values between start and stop.
 * 
 * Values are evenly distributed and filled in row-major order.
 * 
 * @param dims Array of dimensions, ending with 0.
 * @param start The starting value of the sequence.
 * @param stop The ending value of the sequence (inclusive).
 * @param num The number of values to generate.
 * @return A handle to the newly created ndarray with linearly spaced values.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_linspace(NDA_DIMS(2, 5), 0.0, 1.0, 10);
 * // Creates [2,5] array with 10 values evenly spaced from 0.0 to 1.0
 * ndarray_free(arr);
 * ```
 */
NDArray ndarray_new_linspace(const size_t *dims, double start, double stop,
        size_t num);

/**
 * Creates a new ndarray with random values normally distributed.
 * 
 * Values follow a Gaussian distribution with the specified mean and
 * standard deviation.
 * 
 * @param dims Array of dimensions, ending with 0.
 * @param mean The mean of the normal distribution.
 * @param stddev The standard deviation of the normal distribution.
 * @return A handle to the newly created ndarray with random values.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_randnorm(NDA_DIMS(3, 4), 0.0, 1.0);
 * // Standard normal distribution (mean=0, stddev=1)
 * ndarray_free(arr);
 * ```
 */
NDArray ndarray_new_randnorm(const size_t *dims, double mean, double stddev);

/**
 * Creates a new ndarray with random values uniformly distributed.
 * 
 * Values are uniformly distributed between low (inclusive) and
 * high (exclusive).
 * 
 * @param dims Array of dimensions, ending with 0.
 * @param low The lower bound of the uniform distribution (inclusive).
 * @param high The upper bound of the uniform distribution (exclusive).
 * @return A handle to the newly created ndarray with random values.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_randunif(NDA_DIMS(3, 4), 0.0, 1.0);
 * // Random values between 0.0 and 1.0
 * ndarray_free(arr);
 * ```
 */
NDArray ndarray_new_randunif(const size_t *dims, double low, double high);

/**
 * Creates a new ndarray with random values from a Poisson distribution.
 * 
 * Uses Knuth's algorithm to generate Poisson-distributed random integers.
 * The Poisson distribution models the number of events occurring in a fixed
 * interval when events occur at a constant average rate (lambda).
 * 
 * @param dims Array of dimensions, ending with 0.
 * @param lambda The expected number of events (rate parameter, must be > 0).
 * @return A handle to the newly created ndarray with Poisson random values.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_randpoisson(NDA_DIMS(3, 4), 5.0);
 * // Random counts with average of 5 events per interval
 * ndarray_free(arr);
 * ```
 */
NDArray ndarray_new_randpoisson(const size_t *dims, double lambda);

/**
 * Adds two ndarrays element-wise.
 * 
 * The result is stored in the first ndarray (A).
 * This function modifies the input ndarray A in place.
 * Arrays must have identical dimensions.
 * 
 * @param A The first input ndarray (modified in place).
 * @param B The second input ndarray.
 * @return A handle to the same ndarray A with the result of the addition.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_ones(NDA_DIMS(3, 4));
 * NDArray B = ndarray_new_full(NDA_DIMS(3, 4), 2.0);
 * ndarray_add(A, B);  // A = A + B (A now contains all 3.0)
 * ndarray_free_all(NDA_LIST(A, B));
 * ```
 */
NDArray ndarray_add(const NDArray A, const NDArray B);

/**
 * Multiplies two ndarrays element-wise.
 * 
 * The result is stored in the first ndarray (A).
 * This function modifies the input ndarray A in place.
 * Arrays must have identical dimensions.
 * 
 * @param A The first input ndarray (modified in place).
 * @param B The second input ndarray.
 * @return A handle to the same ndarray A with the result of the multiplication.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_full(NDA_DIMS(3, 4), 3.0);
 * NDArray B = ndarray_new_full(NDA_DIMS(3, 4), 2.0);
 * ndarray_mul(A, B);  // A = A * B (A now contains all 6.0)
 * ndarray_free_all(NDA_LIST(A, B));
 * ```
 */
NDArray ndarray_mul(const NDArray A, const NDArray B);

/**
 * Adds a scalar value to each element of the input ndarray.
 * 
 * This function modifies the input ndarray in place.
 * 
 * @param A The input ndarray (modified in place).
 * @param scalar The scalar value to add to each element.
 * @return A handle to the same ndarray with each element increased by the scalar value.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_ones(NDA_DIMS(3, 4));
 * ndarray_add_scalar(arr, 5.0);  // All elements become 6.0
 * ndarray_free(arr);
 * ```
 */
NDArray ndarray_add_scalar(const NDArray A, double scalar);

/**
 * Multiplies each element of the input ndarray by a scalar value.
 * 
 * This function modifies the input ndarray in place.
 * 
 * @param A The input ndarray (modified in place).
 * @param scalar The scalar value to multiply with each element.
 * @return A handle to the same ndarray with each element multiplied by the scalar value.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), 2.0);
 * ndarray_mul_scalar(arr, 3.0);  // All elements become 6.0
 * ndarray_free(arr);
 * ```
 */
NDArray ndarray_mul_scalar(const NDArray A, double scalar);

/**
 * Applies a function element-wise over the input ndarray.
 * 
 * This function modifies the input ndarray in place.
 * 
 * @param A The input ndarray (modified in place).
 * @param func A pointer to the function to apply to each element.
 * @return A handle to the same ndarray with the function applied to each element.
 * 
 * Example:
 * 
 * ```c
 * #include <math.h>
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), 4.0);
 * ndarray_mapfnc(arr, sqrt);  // All elements become 2.0
 * ndarray_mapfnc(arr, exp);   // Apply exponential function
 * ndarray_free(arr);
 * ```
 */
NDArray ndarray_mapfnc(const NDArray A, double (*func)(double));

/**
 * Applies a binary function element-wise over the input ndarray with a
 * constant second argument.
 * 
 * This function modifies the input ndarray in place by applying
 * func(A[i], v) to each element.  Useful for operations like power,
 * hypot, fmod, etc.
 * 
 * @param A The input ndarray (modified in place).
 * @param func A pointer to the binary function to apply to each element.
 * @param v The constant second argument to pass to the function.
 * @return A handle to the same ndarray with the function applied to each element.
 * 
 * Example:
 * 
 * ```c
 * #include <math.h>
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), 2.0);
 * // Compute pow(arr, 3.0) for each element
 * ndarray_mapfnc_par(arr, pow, 3.0);  // All elements become 8.0
 * ndarray_free(arr);
 * ```
 */
NDArray ndarray_mapfnc_par(const NDArray A, double (*func)(double, double), double v);

/**
 * Computes the exponential function element-wise.
 * 
 * Returns a new ndarray where each element is exp(x) for the
 * corresponding element x in A. Uses the standard math library exp()
 * function.
 * 
 * @param A The input ndarray.
 * @return A new ndarray with exp(A[i]) for each element.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), 1.0);
 * NDArray result = ndarray_exp(arr);  // All elements become e ≈ 2.71828
 * ndarray_free_all(NDA_LIST(arr, result));
 * ```
 */
NDArray ndarray_exp(const NDArray A);

/**
 * Computes the natural logarithm element-wise.
 * 
 * Returns a new ndarray where each element is log(x) for the
 * corresponding element x in A.  Uses the standard math library log()
 * function.
 * 
 * @param A The input ndarray (elements must be positive).
 * @return A new ndarray with log(A[i]) for each element.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), exp(1.0));  // Fill with e
 * NDArray result = ndarray_log(arr);  // All elements become 1.0
 * ndarray_free_all(NDA_LIST(arr, result));
 * ```
 */
NDArray ndarray_log(const NDArray A);

/**
 * Computes the square root element-wise.
 * 
 * Returns a new ndarray where each element is sqrt(x) for the
 * corresponding element x in A.  Uses the standard math library sqrt()
 * function.
 * 
 * @param A The input ndarray (elements must be non-negative).
 * @return A new ndarray with sqrt(A[i]) for each element.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), 4.0);
 * NDArray result = ndarray_sqrt(arr);  // All elements become 2.0
 * ndarray_free_all(NDA_LIST(arr, result));
 * ```
 */
NDArray ndarray_sqrt(const NDArray A);

/**
 * Computes the sine function element-wise.
 * 
 * Returns a new ndarray where each element is sin(x) for the
 * corresponding element x in A.  Angles are in radians. Uses the
 * standard math library sin() function.
 * 
 * @param A The input ndarray (angles in radians).
 * @return A new ndarray with sin(A[i]) for each element.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), M_PI / 2.0);  // π/2
 * NDArray result = ndarray_sin(arr);  // All elements become 1.0
 * ndarray_free_all(NDA_LIST(arr, result));
 * ```
 */
NDArray ndarray_sin(const NDArray A);

/**
 * Computes the cosine function element-wise.
 * 
 * Returns a new ndarray where each element is cos(x) for the
 * corresponding element x in A.  Angles are in radians. Uses the
 * standard math library cos() function.
 * 
 * @param A The input ndarray (angles in radians).
 * @return A new ndarray with cos(A[i]) for each element.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), 0.0);
 * NDArray result = ndarray_cos(arr);  // All elements become 1.0
 * ndarray_free_all(NDA_LIST(arr, result));
 * ```
 */
NDArray ndarray_cos(const NDArray A);

/**
 * Computes the tangent function element-wise.
 * 
 * Returns a new ndarray where each element is tan(x) for the
 * corresponding element x in A.  Angles are in radians. Uses the
 * standard math library tan() function.
 * 
 * @param A The input ndarray (angles in radians).
 * @return A new ndarray with tan(A[i]) for each element.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), M_PI / 4.0);  // π/4
 * NDArray result = ndarray_tan(arr);  // All elements become 1.0
 * ndarray_free_all(NDA_LIST(arr, result));
 * ```
 */
NDArray ndarray_tan(const NDArray A);

/**
 * Computes the hyperbolic sine function element-wise.
 * 
 * Returns a new ndarray where each element is sinh(x) for the
 * corresponding element x in A.  Uses the standard math library sinh()
 * function.
 * 
 * @param A The input ndarray.
 * @return A new ndarray with sinh(A[i]) for each element.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), 1.0);
 * NDArray result = ndarray_sinh(arr);  // All elements become sinh(1) ≈ 1.17520
 * ndarray_free_all(NDA_LIST(arr, result));
 * ```
 */
NDArray ndarray_sinh(const NDArray A);

/**
 * Computes the hyperbolic cosine function element-wise.
 * 
 * Returns a new ndarray where each element is cosh(x) for the
 * corresponding element x in A.  Uses the standard math library cosh()
 * function.
 * 
 * @param A The input ndarray.
 * @return A new ndarray with cosh(A[i]) for each element.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), 1.0);
 * NDArray result = ndarray_cosh(arr);  // All elements become cosh(1) ≈ 1.54308
 * ndarray_free_all(NDA_LIST(arr, result));
 * ```
 */
NDArray ndarray_cosh(const NDArray A);

/**
 * Computes the hyperbolic tangent function element-wise.
 * 
 * Returns a new ndarray where each element is tanh(x) for the
 * corresponding element x in A.  Uses the standard math library tanh()
 * function.
 * 
 * @param A The input ndarray.
 * @return A new ndarray with tanh(A[i]) for each element.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), 1.0);
 * NDArray result = ndarray_tanh(arr);  // All elements become tanh(1) ≈ 0.76159
 * ndarray_free_all(NDA_LIST(arr, result));
 * ```
 */
NDArray ndarray_tanh(const NDArray A);

/**
 * Computes the inverse sine function element-wise.
 * 
 * Returns a new ndarray where each element is asin(x) for the
 * corresponding element x in A.  Input elements must be in [-1, 1].
 * Output is in radians in [-π/2, π/2].  Uses the standard math library
 * asin() function.
 * 
 * @param A The input ndarray (elements must be in [-1, 1]).
 * @return A new ndarray with asin(A[i]) for each element (radians).
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), 1.0);
 * NDArray result = ndarray_asin(arr);  // All elements become π/2
 * ndarray_free_all(NDA_LIST(arr, result));
 * ```
 */
NDArray ndarray_asin(const NDArray A);

/**
 * Computes the inverse cosine function element-wise.
 * 
 * Returns a new ndarray where each element is acos(x) for the
 * corresponding element x in A.  Input elements must be in [-1, 1].
 * Output is in radians in [0, π].  Uses the standard math library
 * acos() function.
 * 
 * @param A The input ndarray (elements must be in [-1, 1]).
 * @return A new ndarray with acos(A[i]) for each element (radians).
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), 1.0);
 * NDArray result = ndarray_acos(arr);  // All elements become 0.0
 * ndarray_free_all(NDA_LIST(arr, result));
 * ```
 */
NDArray ndarray_acos(const NDArray A);

/**
 * Computes the inverse tangent function element-wise.
 * 
 * Returns a new ndarray where each element is atan(x) for the
 * corresponding element x in A.  Output is in radians in (-π/2, π/2).
 * Uses the standard math library atan() function.
 * 
 * @param A The input ndarray.
 * @return A new ndarray with atan(A[i]) for each element (radians).
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), 1.0);
 * NDArray result = ndarray_atan(arr);  // All elements become π/4
 * ndarray_free_all(NDA_LIST(arr, result));
 * ```
 */
NDArray ndarray_atan(const NDArray A);

/**
 * Computes element-wise exponentiation.
 * 
 * Returns a new ndarray where each element is pow(x, exponent) for the
 * corresponding element x in A.  Uses the standard math library pow()
 * function.
 * 
 * @param A The base ndarray.
 * @param exponent The exponent value.
 * @return A new ndarray with pow(A[i], exponent) for each element.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_full(NDA_DIMS(3, 4), 2.0);
 * NDArray result = ndarray_pow(arr, 3.0);  // All elements become 8.0 (2^3)
 * ndarray_free_all(NDA_LIST(arr, result));
 * ```
 */
NDArray ndarray_pow(const NDArray A, double exponent);

/**
 * Linear combination: A = alpha*A + beta*B
 * 
 * Computes a scaled linear combination of two arrays and stores the result in A.
 * Uses optimized BLAS routines (dscal and daxpy) for efficiency.
 * Arrays must have identical dimensions.
 * 
 * @param A The first ndarray (modified in place).
 * @param alpha Scalar coefficient for A.
 * @param B The second ndarray.
 * @param beta Scalar coefficient for B.
 * @return A handle to the modified ndarray A.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_ones(NDA_DIMS(3, 4));
 * NDArray B = ndarray_new_full(NDA_DIMS(3, 4), 2.0);
 * ndarray_axpby(A, 2.0, B, 3.0);  // A = 2*A + 3*B = 8.0
 * ndarray_axpby(A, 1.0, B, -1.0); // A = A - B (subtraction)
 * ndarray_free_all(NDA_LIST(A, B));
 * ```
 */
NDArray ndarray_axpby(const NDArray A, double alpha, const NDArray B,
        double beta);

/**
 * Scale and shift: A = alpha*A + beta
 * 
 * Efficiently computes an affine transformation on array elements.
 * Uses optimized BLAS dscal for scaling.
 * 
 * @param A The ndarray to modify (modified in place).
 * @param alpha Scaling factor.
 * @param beta Shift value.
 * @return A handle to the modified ndarray A.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_ones(NDA_DIMS(3, 4));
 * ndarray_scale_shift(A, 2.0, 3.0);  // A = 2*A + 3 = 5.0
 * ndarray_scale_shift(A, 0.5, 0.0);  // A = A / 2 = 2.5
 * ndarray_free(A);
 * ```
 */
NDArray ndarray_scale_shift(const NDArray A, double alpha, double beta);

/**
 * Element-wise multiply then scale: A = A * B * scalar
 * 
 * Computes element-wise product of A and B, then scales by scalar.
 * All operations are fused for better performance.
 * Arrays must have identical dimensions.
 * 
 * @param A The first ndarray (modified in place).
 * @param B The second ndarray.
 * @param scalar Scaling factor.
 * @return A handle to the modified ndarray A.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_full(NDA_DIMS(3, 4), 2.0);
 * NDArray B = ndarray_new_full(NDA_DIMS(3, 4), 3.0);
 * ndarray_mul_scaled(A, B, 2.0);  // A = 2 * A * B = 12.0
 * ndarray_free_all(NDA_LIST(A, B));
 * ```
 */
NDArray ndarray_mul_scaled(const NDArray A, const NDArray B, double scalar);

/**
 * Map function, then multiply: A = func(A) * B * alpha
 * 
 * Applies a function element-wise to A, then multiplies by B and scales.
 * Fused operation for better performance than separate map + multiply.
 * Arrays must have identical dimensions.
 * 
 * @param A The first ndarray (modified in place).
 * @param func Function to apply to each element of A.
 * @param B The second ndarray.
 * @param alpha Scaling factor.
 * @return A handle to the modified ndarray A.
 * 
 * Example:
 * 
 * ```c
 * #include <math.h>
 * NDArray A = ndarray_new_full(NDA_DIMS(3, 4), 4.0);
 * NDArray B = ndarray_new_full(NDA_DIMS(3, 4), 2.0);
 * ndarray_map_mul(A, sqrt, B, 0.5);  // A = sqrt(A) * B * 0.5 = 2.0
 * ndarray_free_all(NDA_LIST(A, B));
 * ```
 */
NDArray ndarray_map_mul(const NDArray A, double (*func)(double), 
                        const NDArray B, double alpha);

/**
 * Fused multiply-add: C = alpha * (A * B) + beta * C
 * 
 * Element-wise multiply A and B, scale by alpha, add to beta*C.
 * Efficiently computes weighted element-wise product with accumulation.
 * All arrays must have identical dimensions.
 * 
 * @param A The first source ndarray.
 * @param B The second source ndarray.
 * @param C The destination ndarray (modified in place).
 * @param alpha Scaling factor for A*B.
 * @param beta Scaling factor for existing C values.
 * @return A handle to the modified ndarray C.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_full(NDA_DIMS(3, 4), 2.0);
 * NDArray B = ndarray_new_full(NDA_DIMS(3, 4), 3.0);
 * NDArray C = ndarray_new_ones(NDA_DIMS(3, 4));
 * ndarray_mul_add(A, B, C, 1.0, 1.0);  // C = C + A*B = 7.0
 * ndarray_free_all(NDA_LIST(A, B, C));
 * ```
 */
NDArray ndarray_mul_add(const NDArray A, const NDArray B, const NDArray C, 
                        double alpha, double beta);

/**
 * Matrix-vector multiply with accumulation: y = alpha * A * x + beta * y
 * 
 * Computes a linear combination of matrix-vector product and vector using optimized BLAS dgemv.
 * A must be 2D, x and y must be vectors (one dimension = 1).
 * 
 * @param y The output vector (modified in place).
 * @param A The matrix (2D ndarray).
 * @param x The input vector (column [n,1] or row [1,n]).
 * @param alpha Scaling factor for A*x.
 * @param beta Scaling factor for existing y values.
 * @return A handle to the modified ndarray y for chaining.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new(NDA_DIMS(3, 4));
 * NDArray x = ndarray_new(NDA_DIMS(4, 1));
 * NDArray y = ndarray_new(NDA_DIMS(3, 1));
 * ndarray_matvec_mul(y, A, x, 1.0, 0.0);  // y = A * x
 * ndarray_free_all(NDA_LIST(A, x, y));
 * ```
 */
// NDArray ndarray_matvec_mul(const NDArray y, const NDArray A, const NDArray x,
//                            double alpha, double beta);

/**
 * Clip values below a minimum threshold: A = max(A, min_val)
 * 
 * All elements less than min_val are set to min_val.
 * SIMD-optimized for performance.
 * 
 * @param A The ndarray to modify (modified in place).
 * @param min_val Minimum value threshold.
 * @return A handle to the modified ndarray A.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_randunif(NDA_DIMS(3, 4), -2.0, 2.0);
 * ndarray_clip_min(A, 0.0);  // Non-negativity constraint (ReLU)
 * ndarray_free(A);
 * ```
 */
NDArray ndarray_clip_min(const NDArray A, double min_val);

/**
 * Clip values above a maximum threshold: A = min(A, max_val)
 * 
 * All elements greater than max_val are set to max_val.
 * SIMD-optimized for performance.
 * 
 * @param A The ndarray to modify (modified in place).
 * @param max_val Maximum value threshold.
 * @return A handle to the modified ndarray A.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_randunif(NDA_DIMS(3, 4), -2.0, 2.0);
 * ndarray_clip_max(A, 1.0);  // Cap values at 1.0
 * ndarray_free(A);
 * ```
 */
NDArray ndarray_clip_max(const NDArray A, double max_val);

/**
 * Clip values to a range: A = clamp(A, min_val, max_val)
 * 
 * All elements are constrained to [min_val, max_val].
 * SIMD-optimized for performance.
 * 
 * @param A The ndarray to modify (modified in place).
 * @param min_val Minimum value threshold.
 * @param max_val Maximum value threshold.
 * @return A handle to the modified ndarray A.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_randunif(NDA_DIMS(3, 4), -2.0, 2.0);
 * ndarray_clip(A, 0.0, 1.0);  // Clip to [0, 1] range (probabilities)
 * ndarray_free(A);
 * ```
 */
NDArray ndarray_clip(const NDArray A, double min_val, double max_val);

/**
 * Absolute value: A = |A|
 * 
 * Computes element-wise absolute value.
 * 
 * @param A The ndarray to modify (modified in place).
 * @return A handle to the modified ndarray A.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_randunif(NDA_DIMS(3, 4), -2.0, 2.0);
 * ndarray_abs(A);  // All values become non-negative
 * ndarray_free(A);
 * ```
 */
NDArray ndarray_abs(const NDArray A);

/**
 * Sign function: A = sign(A)
 * 
 * Returns -1 for negative values, 0 for zero, +1 for positive values.
 * 
 * @param A The ndarray to modify (modified in place).
 * @return A handle to the modified ndarray A.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_randunif(NDA_DIMS(3, 4), -2.0, 2.0);
 * ndarray_sign(A);  // Extract sign information: -1, 0, or 1
 * ndarray_free(A);
 * ```
 */
NDArray ndarray_sign(const NDArray A);

/**
 * Element-wise equality comparison: result = (A == B)
 * 
 * Returns 1.0 where elements are equal, 0.0 otherwise.
 * Arrays must have identical dimensions.
 * 
 * @param A First ndarray.
 * @param B Second ndarray.
 * @return New ndarray with comparison results.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_ones(NDA_DIMS(3, 4));
 * NDArray B = ndarray_new_ones(NDA_DIMS(3, 4));
 * NDArray result = ndarray_new_equal(A, B);  // All 1.0
 * ndarray_free_all(NDA_LIST(A, B, result));
 * ```
 */
NDArray ndarray_new_equal(const NDArray A, const NDArray B);

/**
 * Element-wise less-than comparison: result = (A < B)
 * 
 * Returns 1.0 where A < B, 0.0 otherwise.
 * Arrays must have identical dimensions.
 * 
 * @param A First ndarray.
 * @param B Second ndarray.
 * @return New ndarray with comparison results.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_ones(NDA_DIMS(3, 4));
 * NDArray B = ndarray_new_full(NDA_DIMS(3, 4), 2.0);
 * NDArray result = ndarray_new_less(A, B);  // All 1.0 (A < B)
 * ndarray_free_all(NDA_LIST(A, B, result));
 * ```
 */
NDArray ndarray_new_less(const NDArray A, const NDArray B);

/**
 * Element-wise greater-than comparison: result = (A > B)
 * 
 * Returns 1.0 where A > B, 0.0 otherwise.
 * Arrays must have identical dimensions.
 * 
 * @param A First ndarray.
 * @param B Second ndarray.
 * @return New ndarray with comparison results.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_full(NDA_DIMS(3, 4), 2.0);
 * NDArray B = ndarray_new_ones(NDA_DIMS(3, 4));
 * NDArray result = ndarray_new_greater(A, B);  // All 1.0 (A > B)
 * ndarray_free_all(NDA_LIST(A, B, result));
 * ```
 */
NDArray ndarray_new_greater(const NDArray A, const NDArray B);

/**
 * Scalar equality comparison: result = (A == value)
 * 
 * Returns 1.0 where elements equal value, 0.0 otherwise.
 * 
 * @param A The ndarray.
 * @param value Scalar value to compare against.
 * @return New ndarray with comparison results.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_zeros(NDA_DIMS(3, 4));
 * NDArray mask = ndarray_new_equal_scalar(A, 0.0);  // Find all zeros
 * ndarray_free_all(NDA_LIST(A, mask));
 * ```
 */
NDArray ndarray_new_equal_scalar(const NDArray A, double value);

/**
 * Scalar less-than comparison: result = (A < value)
 * 
 * Returns 1.0 where elements are less than value, 0.0 otherwise.
 * 
 * @param A The ndarray.
 * @param value Scalar value to compare against.
 * @return New ndarray with comparison results.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_randunif(NDA_DIMS(3, 4), 0.0, 10.0);
 * NDArray mask = ndarray_new_less_scalar(A, 5.0);  // Find elements < 5.0
 * ndarray_free_all(NDA_LIST(A, mask));
 * ```
 */
NDArray ndarray_new_less_scalar(const NDArray A, double value);

/**
 * Scalar greater-than comparison: result = (A > value)
 * 
 * Returns 1.0 where elements are greater than value, 0.0 otherwise.
 * 
 * @param A The ndarray.
 * @param value Scalar value to compare against.
 * @return New ndarray with comparison results.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_randunif(NDA_DIMS(3, 4), 0.0, 10.0);
 * NDArray mask = ndarray_new_greater_scalar(A, 5.0);  // Find elements > 5.0
 * ndarray_free_all(NDA_LIST(A, mask));
 * ```
 */
NDArray ndarray_new_greater_scalar(const NDArray A, double value);

/**
 * Logical AND: result = (A && B)
 * 
 * Returns 1.0 where both elements are non-zero, 0.0 otherwise.
 * Arrays must have identical dimensions.
 * 
 * @param A First ndarray (treated as boolean).
 * @param B Second ndarray (treated as boolean).
 * @return New ndarray with logical AND results.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_greater_scalar(data, 0.0);
 * NDArray B = ndarray_new_less_scalar(data, 10.0);
 * NDArray mask = ndarray_new_logical_and(A, B);  // 0 < data < 10
 * ndarray_free_all(NDA_LIST(A, B, mask));
 * ```
 */
NDArray ndarray_new_logical_and(const NDArray A, const NDArray B);

/**
 * Logical OR: result = (A || B)
 * 
 * Returns 1.0 where at least one element is non-zero, 0.0 otherwise.
 * Arrays must have identical dimensions.
 * 
 * @param A First ndarray (treated as boolean).
 * @param B Second ndarray (treated as boolean).
 * @return New ndarray with logical OR results.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_less_scalar(data, 0.0);
 * NDArray B = ndarray_new_greater_scalar(data, 10.0);
 * NDArray mask = ndarray_new_logical_or(A, B);  // data < 0 OR data > 10
 * ndarray_free_all(NDA_LIST(A, B, mask));
 * ```
 */
NDArray ndarray_new_logical_or(const NDArray A, const NDArray B);

/**
 * Logical NOT: result = !A
 * 
 * Returns 1.0 where element is zero, 0.0 where non-zero.
 * 
 * @param A The ndarray (treated as boolean).
 * @return New ndarray with logical NOT results.
 * 
 * Example:
 * 
 * ```c
 * NDArray mask = ndarray_new_equal_scalar(data, 0.0);
 * NDArray not_mask = ndarray_new_logical_not(mask);  // Invert mask
 * ndarray_free_all(NDA_LIST(mask, not_mask));
 * ```
 */
NDArray ndarray_new_logical_not(const NDArray A);

/**
 * Element-wise ternary operator: result = condition ? x : y
 * 
 * NumPy-style where function. Selects elements from x or y based on condition.
 * All arrays must have identical dimensions.
 * 
 * @param condition Boolean array (non-zero = true).
 * @param x Array to select from when condition is true.
 * @param y Array to select from when condition is false.
 * @return New ndarray with selected values.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_randunif(NDA_DIMS(3, 4), -2.0, 2.0);
 * NDArray positive = ndarray_new_greater_scalar(A, 0.0);
 * NDArray zeros = ndarray_new_zeros(NDA_DIMS(3, 4));
 * NDArray result = ndarray_new_where(positive, A, zeros);  // Keep only positive
 * ndarray_free_all(NDA_LIST(A, positive, zeros, result));
 * ```
 */
NDArray ndarray_new_where(const NDArray condition, const NDArray x,
        const NDArray y);

/**
 * Tensor contraction (generalized tensor product).
 * 
 * Contracts specified axes between two tensors using Einstein summation convention.
 * This is the most general tensor operation. For standard matrix multiplication,
 * use ndarray_new_matmul which is optimized for that case.
 * 
 * @param A First tensor (must have ndim >= 2).
 * @param B Second tensor (must have ndim >= 2).
 * @param axes_a Array of axes from A to contract, terminated by -1.
 * @param axes_b Array of axes from B to contract, terminated by -1 (must match axes_a).
 * @return New tensor with contracted dimensions (ndim >= 2).
 * 
 * Example:
 * 
 * ```c
 * // Single axis contraction: A[i,j,k] * B[k,l] -> C[i,j,l]
 * NDArray A = ndarray_new(NDA_DIMS(2, 3, 4));
 * NDArray B = ndarray_new(NDA_DIMS(4, 5));
 * NDArray C = ndarray_new_tensordot(A, B, NDA_AXES(2), NDA_AXES(0));
 * 
 * // Outer product: A[i,j] * B[k,l] -> C[i,j,k,l]
 * NDArray D = ndarray_new_tensordot(A, B, NDA_NO_AXES, NDA_NO_AXES);
 * ndarray_free_all(NDA_LIST(A, B, C, D));
 * ```
 */
NDArray ndarray_new_tensordot(const NDArray A, const NDArray B, 
                               int *axes_a, int *axes_b);

/**
 * Batched matrix multiplication with broadcasting.
 * 
 * Operates on the last two dimensions and broadcasts over leading dimensions.
 * For A with shape [..., m, n] and B with shape [..., n, p], result is [..., m, p].
 * This is a special case of tensor contraction optimized for matrix multiplication.
 * Uses cache-optimized blocked algorithm for better performance.
 * 
 * @param A The first input ndarray (must have ndim >= 2).
 * @param B The second input ndarray (must have ndim >= 2).
 * @return A handle to the newly created ndarray, or NULL if dimensions are incompatible.
 * 
 * Example:
 * 
 * ```c
 * // 2D matrix multiplication
 * NDArray A = ndarray_new(NDA_DIMS(3, 4));
 * NDArray B = ndarray_new(NDA_DIMS(4, 5));
 * NDArray C = ndarray_new_matmul(A, B);  // Shape: [3, 5]
 * 
 * // 3D batched multiplication
 * NDArray D = ndarray_new(NDA_DIMS(2, 3, 4));
 * NDArray E = ndarray_new(NDA_DIMS(2, 4, 5));
 * NDArray F = ndarray_new_matmul(D, E);  // Shape: [2, 3, 5]
 * ndarray_free_all(NDA_LIST(A, B, C, D, E, F));
 * ```
 */
NDArray ndarray_new_matmul(const NDArray A, const NDArray B);

/**
 * Stacks ndarrays along a new axis (all shapes must be identical).
 * 
 * Creates a new dimension at the specified position.
 * All arrays must have the same shape.
 * 
 * @param axis Position for the new dimension (0 to ndim).
 * @param arr_list NULL-terminated array of ndarrays (same shape required).
 * @return New ndarray with ndim+1 dimensions.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new(NDA_DIMS(2, 3));
 * NDArray B = ndarray_new(NDA_DIMS(2, 3));
 * NDArray C = ndarray_new(NDA_DIMS(2, 3));
 * 
 * // Stack along new axis 0: [2,3] -> [3, 2, 3]
 * NDArray stacked = ndarray_new_stack(0, NDA_LIST(A, B, C));
 * ndarray_free_all(NDA_LIST(A, B, C, stacked));
 * ```
 */
NDArray ndarray_new_stack(int axis, NDArray* arr_list);

/**
 * Concatenates ndarrays along an existing axis.
 * 
 * All dimensions except the concatenation axis must match.
 * 
 * @param axis Axis along which to concatenate (0 to ndim-1).
 * @param arr_list NULL-terminated list of ndarrays to concatenate.
 * @return New ndarray with extended dimension along axis.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new(NDA_DIMS(2, 3, 4));
 * NDArray B = ndarray_new(NDA_DIMS(2, 5, 4));
 * 
 * // Concatenate along axis 1: [2,3,4] + [2,5,4] -> [2, 8, 4]
 * NDArray concatenated = ndarray_new_concat(1, NDA_LIST(A, B));
 * ndarray_free_all(NDA_LIST(A, B, concatenated));
 * ```
 */
NDArray ndarray_new_concat(int axis, NDArray* arr_list);

/**
 * Extract a subregion from an ndarray.
 * 
 * Creates a new ndarray containing elements from start to end indices (exclusive).
 * 
 * @param arr Source ndarray.
 * @param axis Axis along which to take the subregion.
 * @param start Starting index (inclusive).
 * @param end Ending index (exclusive).
 * @return New ndarray with copied subregion.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new(NDA_DIMS(4, 5));
 * 
 * // Extract rows 1 and 2: [4,5] -> [2,5]
 * NDArray rows = ndarray_new_take(A, 0, 1, 3);
 * 
 * // Extract columns 2, 3, 4: [3,6] -> [3,3]
 * NDArray cols = ndarray_new_take(A, 1, 2, 5);
 * ndarray_free_all(NDA_LIST(A, rows, cols));
 * ```
 */
NDArray ndarray_new_take(const NDArray arr, int axis, size_t start, size_t end);

/**
 * Creates a new ndarray that is the transpose of the given ndarray.
 * 
 * For N-dimensional arrays, reverses all axes (e.g., shape [2,3,4]
 * becomes [4,3,2]).
 *
 * @param A The input ndarray.
 * @return A handle to the newly created transposed ndarray.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new(NDA_DIMS(2, 3, 4));  // Shape: [2, 3, 4]
 * NDArray T = ndarray_new_transpose(A);         // Shape: [4, 3, 2]
 * ndarray_free_all(NDA_LIST(A, T));
 * ```
 */
NDArray ndarray_new_transpose(const NDArray A);

/**
 * Reshapes an ndarray in-place to new dimensions.
 * 
 * The total number of elements must remain the same.
 * Data remains in row-major (C-order) layout.
 * 
 * @param arr The ndarray to reshape (modified in place).
 * @param new_dims New dimensions (terminated by 0). Use -1 for one dimension
 *                 to be automatically inferred from the total size.
 * 
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new(NDA_DIMS(2, 6));     // Shape: [2, 6]
 * ndarray_reshape(arr, NDA_DIMS(3, 4));          // Shape: [3, 4]
 * ndarray_reshape(arr, NDA_DIMS(2, 2, 3));       // Shape: [2, 2, 3]
 * ndarray_free(arr);
 * ```
 */
void ndarray_reshape(const NDArray arr, const size_t* new_dims);

/**
 * Aggregation types for ndarray_new_aggr and ndarray_scalar_aggr functions.
 * 
 * NDA_AGGR_SUM:  Sum of elements
 * NDA_AGGR_MEAN: Mean (average) of elements
 * NDA_AGGR_STD:  Standard deviation of elements
 * NDA_AGGR_MAX:  Maximum element value
 * NDA_AGGR_MIN:  Minimum element value
 */
enum {
    NDA_AGGR_SUM = 0,
    NDA_AGGR_MEAN,
    NDA_AGGR_STD,
    NDA_AGGR_MAX,
    NDA_AGGR_MIN
};

/**
 * Creates a new ndarray by aggregating over a specified axis.
 *
 * Result maintains ndim >= 2 constraint:
 * - axis == NDA_ALL_AXES: returns shape [1, 1] with scalar result
 * - axis in [0, ndim-1]: if result would be 1D, adds dimension of 1
 *
 * @param A The input ndarray.
 * @param axis The axis to aggregate over (0 to ndim-1), or NDA_ALL_AXES for all axes.
 * @param aggr_type The type of aggregation to perform (NDA_AGGR_*).
 * @return A handle to the result ndarray (ndim >= 2).
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new(NDA_DIMS(3, 4));
 * 
 * // Sum along axis 0: [3, 4] -> [1, 4]
 * NDArray sum0 = ndarray_new_aggr(A, 0, NDA_AGGR_SUM);
 * 
 * // Mean of all elements: [3, 4] -> [1, 1]
 * NDArray mean_all = ndarray_new_aggr(A, NDA_ALL_AXES, NDA_AGGR_MEAN);
 * ndarray_free_all(NDA_LIST(A, sum0, mean_all));
 * ```
 */
NDArray ndarray_new_aggr(const NDArray A, int axis, int aggr_type);

/**
 * Performs an aggregation over all elements of the array.
 *
 * Returns a scalar value representing the aggregation result.
 *
 * @param A The input ndarray.
 * @param aggr_type The type of aggregation to perform (NDA_AGGR_*).
 * @return The aggregation value as a scalar.
 * 
 * Example:
 * 
 * ```c
 * NDArray A = ndarray_new_randunif(NDA_DIMS(3, 4), 0.0, 10.0);
 * 
 * double sum = ndarray_scalar_aggr(A, NDA_AGGR_SUM);
 * double mean = ndarray_scalar_aggr(A, NDA_AGGR_MEAN);
 * double max = ndarray_scalar_aggr(A, NDA_AGGR_MAX);
 * 
 * printf("Sum: %f, Mean: %f, Max: %f\n", sum, mean, max);
 * ndarray_free(A);
 * ```
 */
double ndarray_scalar_aggr(const NDArray A, int aggr_type);

/**
 * Saves an ndarray to a binary file.
 * 
 * Creates a binary file with a custom format including magic number,
 * version, dimensions, and data in row-major order.
 *
 * @param arr The ndarray to save.
 * @param filename Path to the output file (use .bin extension).
 * @return 0 on success, -1 on error.
 *
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_randunif(NDA_DIMS(3, 4), 0.0, 1.0);
 * if (ndarray_save(arr, "mydata.bin") == 0) {
 *     printf("Array saved successfully\n");
 * }
 * ndarray_free(arr);
 * ```
 */
int ndarray_save(const NDArray arr, const char *filename);

/**
 * Loads an ndarray from a binary file.
 * 
 * Reads the file format created by ndarray_save().
 * Returns NULL if the file cannot be read or is corrupted.
 *
 * @param filename Path to the input file (.bin extension).
 * @return A newly allocated ndarray, or NULL on error.
 *
 * Example:
 * 
 * ```c
 * NDArray arr = ndarray_new_load("mydata.bin");
 * if (arr == NULL) {
 *     fprintf(stderr, "Failed to load array\n");
 *     return 1;
 * }
 * ndarray_print(arr, "Loaded", 4);
 * ndarray_free(arr);
 * ```
 */
NDArray ndarray_new_load(const char *filename);

/**
 * Get the stride value for linear access (in elements, not bytes).
 * Default is 1 for contiguous data.
 * 
 * @param arr NDArray
 * @return Stride in elements
 */
size_t ndarray_get_stride(const NDArray arr);

/**
 * Set the stride value (advanced usage).
 * 
 * @param arr NDArray to modify
 * @param stride New stride in elements (must be > 0)
 * @return Handler for chaining
 */
NDArray ndarray_set_stride(NDArray arr, size_t stride);

/**
 * Get the trailing dimension (tda) - physical row width in memory.
 * For 2D matrices, this is the number of elements between row starts.
 * 
 * @param arr NDArray
 * @return tda value
 */
size_t ndarray_get_tda(const NDArray arr);

/**
 * Set the trailing dimension (advanced usage).
 * 
 * @param arr NDArray to modify
 * @param tda New trailing dimension (must be > 0)
 * @return Handler for chaining
 */
NDArray ndarray_set_tda(NDArray arr, size_t tda);

/**
 * Get the ownership flag.
 * Returns 1 if NDArray owns its data, 0 if data is borrowed.
 * 
 * @param arr NDArray
 * @return 1 if owner, 0 if borrowed
 */
int ndarray_get_owner(const NDArray arr);

/**
 * Set the ownership flag (advanced usage).
 * Set to 0 when wrapping external data that should not be freed.
 * 
 * @param arr NDArray to modify
 * @param owner 1 to own data, 0 if borrowed
 * @return Handler for chaining
 */
NDArray ndarray_set_owner(NDArray arr, int owner);

/* Prevent GSL from including its own CBLAS definitions which conflict with OpenBLAS.
 * We define the necessary types that GSL's gsl_cblas.h would provide, then skip its inclusion. */
#define __GSL_CBLAS_H__

/* Define types that GSL's gsl_cblas.h would provide */
#ifndef CBLAS_INDEX
#define CBLAS_INDEX size_t
#endif

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

/**
 * Create a gsl_vector view from a 1D slice of NDArray (read-write).
 * 
 * For 2D arrays where one dimension equals 1 (e.g., [n,1] or [1,n]), extracts
 * that single row/column as a vector. For higher-dimensional arrays, extracts
 * a slice perpendicular to the specified axis.
 * 
 * The returned view does NOT own the data and does NOT make a copy.
 * It points directly to NDArray's memory. Modifications through the vector
 * view directly modify the NDArray.
 * 
 * @param arr NDArray source
 * @param axis Axis perpendicular to extraction (0=rows, 1=cols for 2D)
 * @param index Index along the specified axis
 * @return A gsl_vector_view with data pointing to arr->data (no copy)
 * 
 * Example:
 * ```c
 * NDArray mat = ndarray_new(NDA_DIMS(4, 5));
 * // ... fill matrix ...
 * gsl_vector_view v = ndarray_to_gsl_vector(mat, 0, 2);  // Row 2
 * double norm = gsl_blas_dnrm2(&v.vector);
 * ```
 */
gsl_vector_view ndarray_to_gsl_vector(NDArray arr, int axis, size_t index);

/**
 * Create a const gsl_vector view from NDArray (read-only).
 * 
 * Same as ndarray_to_gsl_vector() but for const arrays and prevents modification.
 * 
 * @param arr Const NDArray source
 * @param axis Axis perpendicular to extraction
 * @param index Index along the specified axis
 * @return A gsl_vector_const_view (read-only)
 */
gsl_vector_const_view ndarray_to_gsl_vector_const(const NDArray arr, int axis, size_t index);

/**
 * Create a gsl_matrix view from a 2D NDArray (read-write).
 * 
 * Direct memory mapping: NDArray's row-major layout aligns with GSL's row-major format.
 * The returned view respects NDArray's stride and tda fields for proper element access.
 * 
 * The view does NOT own the data and does NOT make a copy. It points directly to
 * NDArray's memory.
 * 
 * @param arr NDArray (must be 2D, i.e., ndim == 2)
 * @return A gsl_matrix_view with:
 *   - size1 = arr->dims[0] (number of rows)
 *   - size2 = arr->dims[1] (number of columns)
 *   - tda = arr->tda (physical row width in memory, including padding)
 *   - data = arr->data (same memory as NDArray)
 * 
 * Example:
 * ```c
 * NDArray A = ndarray_new(NDA_DIMS(3, 3));
 * // ... fill A ...
 * gsl_matrix_view m = ndarray_to_gsl_matrix(A);
 * gsl_permutation* p = gsl_permutation_alloc(3);
 * int signum;
 * gsl_linalg_LU_decomp(&m.matrix, p, &signum);  // A is modified in-place
 * gsl_permutation_free(p);
 * ```
 */
gsl_matrix_view ndarray_to_gsl_matrix(NDArray arr);

/**
 * Create a const gsl_matrix view from NDArray (read-only).
 * 
 * Same as ndarray_to_gsl_matrix() but for const arrays and prevents modification.
 * 
 * @param arr Const NDArray (must be 2D)
 * @return A gsl_matrix_const_view (read-only)
 */
gsl_matrix_const_view ndarray_to_gsl_matrix_const(const NDArray arr);

/**
 * Create a gsl_vector view of a specific row from a 2D NDArray.
 * 
 * Extracts the row as a contiguous vector with stride=1 (natural row layout).
 * 
 * @param arr 2D NDArray
 * @param row_idx Row index (0 to dims[0]-1)
 * @return Vector view of the row (stride=1, no copy)
 * 
 * Example:
 * ```c
 * NDArray mat = ndarray_new(NDA_DIMS(4, 5));
 * gsl_vector_view row2 = ndarray_to_gsl_row(mat, 2);
 * gsl_vector_scale(&row2.vector, 2.0);  // Scale row 2 by 2
 * ```
 */
gsl_vector_view ndarray_to_gsl_row(NDArray arr, size_t row_idx);

/**
 * Create a const gsl_vector view of a specific row from a 2D NDArray (read-only).
 * 
 * @param arr Const 2D NDArray
 * @param row_idx Row index
 * @return Vector view of the row (read-only)
 */
gsl_vector_const_view ndarray_to_gsl_row_const(const NDArray arr, size_t row_idx);

/**
 * Create a gsl_vector view of a specific column from a 2D NDArray.
 * 
 * Extracts the column as a strided vector with stride=tda (column-major access).
 * 
 * @param arr 2D NDArray
 * @param col_idx Column index (0 to dims[1]-1)
 * @return Vector view of the column (stride=tda, no copy)
 * 
 * Example:
 * ```c
 * NDArray mat = ndarray_new(NDA_DIMS(4, 5));
 * gsl_vector_view col3 = ndarray_to_gsl_column(mat, 3);
 * double sum = gsl_vector_sum(&col3.vector);  // Sum column 3
 * ```
 */
gsl_vector_view ndarray_to_gsl_column(NDArray arr, size_t col_idx);

/**
 * Create a const gsl_vector view of a specific column from a 2D NDArray (read-only).
 * 
 * @param arr Const 2D NDArray
 * @param col_idx Column index
 * @return Vector view of the column (read-only)
 */
gsl_vector_const_view ndarray_to_gsl_column_const(const NDArray arr, size_t col_idx);

/**
 * Get the stride value for linear access (in elements, not bytes).
 * 
 * Returns the number of elements between consecutive accessed elements in the
 * underlying data array.
 * 
 * @param arr NDArray
 * @return Stride in elements (default: 1 for contiguous)
 */
size_t ndarray_get_stride(const NDArray arr);

/**
 * Set the stride value (advanced usage).
 * 
 * WARNING: Only use this if you understand GSL stride semantics. Improper stride
 * values can cause undefined behavior or memory access errors.
 * 
 * Stride affects how linear access is computed: element i is at data[i * stride].
 * 
 * @param arr NDArray to modify
 * @param stride New stride in elements (must be > 0)
 * @return Handler for chaining
 */
NDArray ndarray_set_stride(NDArray arr, size_t stride);

/**
 * Get the trailing dimension (tda) - physical row width in memory for 2D matrices.
 * 
 * For a 2D matrix, tda is the number of elements between the start of one row
 * and the start of the next row in physical memory. This can be larger than
 * dims[1] to accommodate padding.
 * 
 * @param arr NDArray
 * @return tda value (for 2D: row pitch; for 1D: 1)
 */
size_t ndarray_get_tda(const NDArray arr);

/**
 * Set the trailing dimension (advanced usage).
 * 
 * WARNING: Only use this for creating views of padded matrices or submatrices.
 * Setting incorrect tda values will cause undefined behavior.
 * 
 * For 2D matrices, tda must be >= dims[1]. For other dimensions, tda is typically 1.
 * 
 * @param arr NDArray to modify
 * @param tda New trailing dimension (must be > 0)
 * @return Handler for chaining
 */
NDArray ndarray_set_tda(NDArray arr, size_t tda);

/**
 * Get the ownership flag.
 * 
 * Returns 1 if the NDArray owns its data block (will be freed by ndarray_free),
 * or 0 if the data is borrowed from another source (will NOT be freed).
 * 
 * @param arr NDArray
 * @return 1 if owner, 0 if borrowed
 */
int ndarray_get_owner(const NDArray arr);

/**
 * Set the ownership flag (advanced usage).
 * 
 * WARNING: Only use this when creating NDArray structures from external data sources.
 * Setting owner=0 tells ndarray_free() not to deallocate the memory.
 * 
 * @param arr NDArray to modify
 * @param owner 1 to own data, 0 if borrowed
 * @return Handler for chaining
 */
NDArray ndarray_set_owner(NDArray arr, int owner);

#endif // _NDARRAY_H
