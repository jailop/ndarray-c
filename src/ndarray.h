#ifndef _NDARRAY_H
#define _NDARRAY_H

/**
 * All ndarrays in this library must have ndim >= 2.
 * 1D arrays are not supported - use 2D arrays with one dimension set to 1.
 */

/**
 * A ndarray structure to represent and operate over
 * multi-dimensional arrays of doubles.
 */
typedef struct {
    double *data;
    size_t *dims;
    size_t ndim;
} _NDArray;

/**
 * A handle to a ndarray structure.
 * It is expected that users will interact with ndarrays
 * through this handle rather than directly manipulating
 * the underlying structure.
 */
typedef _NDArray* NDArray;

/**
 * Macro to create a dimensions array for ndarray creation.
 * Usage: NDA_DIMS(3, 4, 5) creates an array {3, 4, 5, 0}.
 * The last element (0) is used as a sentinel to indicate
 * the end of the dimensions.
 */
#define NDA_DIMS(...) ((size_t[]){__VA_ARGS__, 0})

/**
 * Macro to create a position array for ndarray access.
 * Usage: TPOS(i, j, k) creates an array {i, j, k}.
 */
#define NDA_POS(...) ((size_t[]){__VA_ARGS__})

/**
 * Macro to create an axis array for tensor operations.
 * Usage: NDA_AXES(1, 2) creates an array {1, 2, -1}.
 * For no contraction (outer product), use NDA_NO_AXES.
 */
#define NDA_AXES(...) ((int[]){__VA_ARGS__, -1})

/**
 * Macro to indicate no axes for tensor operations.
 * Used for outer products in tensor contractions.
 */
#define NDA_NO_AXES ((int[]){-1})

/**
 * Macro to create a list of ndarrays for functions
 * that accept multiple arrays.
 * Usage: NDA_LIST(A, B, C) creates an array {A, B, C, NULL}.
 */
#define NDA_LIST(...) ((NDArray[]){__VA_ARGS__, NULL})

/**
 * Constant to indicate operations on all axes.
 * Used with functions like ndarray_new_axis_aggr.
 * Example: ndarray_new_axis_aggr(A, NDA_ALL_AXES, NDARRAY_AGGR_SUM)
 */
#define NDA_ALL_AXES (-1)

/**
 * Creates a new ndarray with the specified dimensions.
 * The dimensions are provided as a variable number of arguments,
 * using the last one as a sentinel (0).
 * CONSTRAINT: ndim must be >= 2 (at least two dimensions required).
 * @param dims Array of dimensions, ending with 0.
 * @return A handle to the newly created ndarray, or NULL if ndim < 2.
 */
NDArray ndarray_new(size_t *dims);

/**
 * Frees the memory allocated for the ndarray.
 * @param t The ndarray to free.
 */
void ndarray_free(NDArray t);

/**
 * Frees the memory allocated for multiple ndarrays.
 * The ndarrays are provided as a variable number of arguments,
 * using the last one as a sentinel (NULL).
 * @param arr_list List of arrays to free
 */
void ndarray_free_all(NDArray* arr_list);

/**
 * Computes the offset in the data array for the given position.
 * @param t The ndarray.
 * @param pos An array of size_t representing the position in each
 * dimension.
 * @return The computed offset in the data array.
 */
size_t ndarray_offset(NDArray t, size_t *pos);

/**
 * Sets the value at the specified position in the ndarray.
 * @param t The ndarray.
 * @param pos An array of size_t representing the position in each dimension.
 * @param value The value to set.
 */
void ndarray_set(NDArray t, size_t* pos, double value);

/**
 * Gets the value at the specified position in the ndarray.
 * @param t The ndarray.
 * @param pos An array of size_t representing the position in each dimension.
 * @return The value at the specified position.
 */
double ndarray_get(NDArray t, size_t* pos);

/**
 * Pretty-prints an ndarray to stdout.
 * Automatically formats output based on dimensionality:
 * - 2D: matrix format with aligned columns
 * - 3D+: nested bracket notation with proper indentation
 * 
 * @param arr The ndarray to print
 * @param name Optional name to display (can be NULL)
 * @param precision Number of decimal places (default 4 if < 0)
 * 
 * Example output for 2D [3, 4]:
 *   Array 'A' [3, 4]:
 *   [[  1.0000   2.0000   3.0000   4.0000]
 *    [  5.0000   6.0000   7.0000   8.0000]
 *    [  9.0000  10.0000  11.0000  12.0000]]
 */
void ndarray_print(NDArray arr, const char *name, int precision);

/**
 * Creates a copy of the given ndarray.
 * @param t The ndarray to copy.
 * @return A handle to the newly created copy of the ndarray.
 */
NDArray ndarray_new_copy(NDArray t);

/**
 * Creates a new ndarray filled with zeros.
 * @param dims An array of size_t representing the dimensions of the ndarray.
 * @return A handle to the newly created ndarray filled with zeros.
 */
NDArray ndarray_new_zeros(size_t *dims);

/**
 * Creates a new ndarray from existing data.
 * The data is copied into the new ndarray.
 * Supports any number of dimensions (ndim >= 2).
 * The size is automatically calculated from the dimensions.
 * @param dims Array of dimensions, ending with 0.
 * @param data Pointer to the data array to copy.
 * @return A handle to the newly created ndarray.
 * 
 * Example usage:
 *   // 2D array: 2x3
 *   double data2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
 *   NDArray a = ndarray_new_from_data(NDA_DIMS(2, 3), (double*)data2d);
 * 
 *   // 3D array: 2x3x4
 *   double data3d[2][3][4] = {...};
 *   NDArray b = ndarray_new_from_data(NDA_DIMS(2, 3, 4), (double*)data3d);
 * 
 *   // 4D array: 2x2x3x2
 *   double data4d[2][2][3][2] = {...};
 *   NDArray c = ndarray_new_from_data(NDA_DIMS(2, 2, 3, 2), (double*)data4d);
 */
NDArray ndarray_new_from_data(size_t *dims, double *data);

/**
 * Creates a new ndarray filled with ones.
 * @param dims An array of size_t representing the dimensions of the ndarray.
 * @return A handle to the newly created ndarray filled with ones.
 */
NDArray ndarray_new_ones(size_t *dims);

/**
 * Creates a new ndarray filled with the specified value.
 * @param dims An array of size_t representing the dimensions of the ndarray.
 * @param value The value to fill the ndarray with.
 * @return A handle to the newly created ndarray filled with the specified value.
 */
NDArray ndarray_new_full(size_t *dims, double value);

/**
 * Creates a new ndarray with values in the range
 * [start, stop) with the given step.
 * @param dims An array of size_t representing the dimensions of the ndarray.
 * @param start The starting value of the sequence.
 * @param stop The ending value of the sequence (exclusive).
 * @param step The step size between values.
 * @return A handle to the newly created ndarray with the specified range of values.
 */
NDArray ndarray_new_arange(size_t *dims, double start, double stop, double step);

/**
 * Creates a new ndarray with linearly spaced values
 * between start and stop.
 * @param dims An array of size_t representing the dimensions of the ndarray.
 * @param start The starting value of the sequence.
 * @param stop The ending value of the sequence.
 * @param num The number of values to generate.
 * @return A handle to the newly created ndarray with linearly spaced values.
 */
NDArray ndarray_new_linspace(size_t *dims, double start, double stop, size_t num);

/**
 * Creates a new ndarray with random values normally
 * distributed with the given mean and standard deviation.
 * @param dims An array of size_t representing the dimensions of the ndarray.
 * @param mean The mean of the normal distribution.
 * @param stddev The standard deviation of the normal distribution.
 * @return A handle to the newly created ndarray with random values.
 */
NDArray ndarray_new_randnorm(size_t *dims, double mean, double stddev);

/**
 * Creates a new ndarray with random values uniformly
 * distributed between low and high.
 * @param dims An array of size_t representing the dimensions of the ndarray.
 * @param low The lower bound of the uniform distribution.
 * @param high The upper bound of the uniform distribution.
 * @return A handle to the newly created ndarray with random values.
 */
NDArray ndarray_new_randunif(size_t *dims, double low, double high);

/**
 * Adds two ndarrays element-wise. The result is stored in
 * the first ndarray (A).
 * @param A The first input ndarray.
 * @param B The second input ndarray.
 * @return A handle to the same ndarray A with the result of
 * the addition.
 * This function modifies the input ndarray A in place.
 */
NDArray ndarray_add(NDArray A, NDArray B);

/**
 * Multiplies two ndarrays element-wise. The result is stored in
 * the first ndarray (A).
 * @param A The first input ndarray.
 * @param B The second input ndarray.
 * @return A handle to the same ndarray A with the result of
 * the multiplication.
 * This function modifies the input ndarray A in place.
 */
NDArray ndarray_mul(NDArray A, NDArray B);

/**
 * Adds a scalar value to each element of the input ndarray.
 * @param A The input ndarray.
 * @param scalar The scalar value to add to each element.
 * @return A handle to the same ndarray with each element increased
 * by the scalar value.
 * This function modifies the input ndarray in place.
 */
NDArray ndarray_add_scalar(NDArray A, double scalar);

/**
 * Multiplies each element of the input ndarray by a scalar value.
 * @param A The input ndarray.
 * @param scalar The scalar value to multiply with each element.
 * @return A handle to the same ndarray with each element multiplied
 * by the scalar value.
 * This function modifies the input ndarray in place.
 */
NDArray ndarray_mul_scalar(NDArray A, double scalar);

/**
 * Applies a function element-wise
 * over the input ndarray.
 * @param A The input ndarray.
 * @param func A pointer to the function to apply to each element.
 * @return A handle to the same ndarray with the function applied to
 * each element.
 * This function modifies the input ndarray in place.
 */
NDArray ndarray_mapfnc(NDArray A, double (*func)(double));

/**
 * Tensor contraction (generalized tensor product).
 * Contracts specified axes between two tensors using Einstein summation convention.
 * Example: A[i,j,k] ⊗ B[k,l,m] contracts on k → C[i,j,l,m]
 * 
 * This is the most general tensor operation. For standard matrix multiplication,
 * use ndarray_new_matmul which is optimized for that case.
 * 
 * @param A First tensor (must have ndim >= 2)
 * @param B Second tensor (must have ndim >= 2)
 * @param axes_a Array of axes from A to contract, terminated by -1
 * @param axes_b Array of axes from B to contract, terminated by -1 (must match axes_a)
 * @return New tensor with contracted dimensions (ndim >= 2)
 * 
 * Examples:
 *   Single axis:  ndarray_new_tensordot(A, B, NDA_AXES(1), NDA_AXES(0))
 *   Multi-axis:   ndarray_new_tensordot(A, B, NDA_AXES(2, 3), NDA_AXES(0, 1))
 *   Outer product: ndarray_new_tensordot(A, B, NDA_NO_AXES, NDA_NO_AXES)
 */
NDArray ndarray_new_tensordot(NDArray A, NDArray B, 
                               int *axes_a, int *axes_b);

/**
 * Batched matrix multiplication with broadcasting.
 * Operates on the last two dimensions and broadcasts over leading dimensions.
 * For A with shape [..., m, n] and B with shape [..., n, p], result is [..., m, p].
 * 
 * This is a special case of tensor contraction optimized for matrix multiplication.
 * Uses cache-optimized blocked algorithm for better performance.
 * For general tensor contractions, use ndarray_new_tensordot.
 * 
 * @param A The first input ndarray (must have ndim >= 2).
 * @param B The second input ndarray (must have ndim >= 2).
 * @return A handle to the newly created ndarray resulting from the
 * matrix multiplication, or NULL if dimensions are incompatible.
 */
NDArray ndarray_new_matmul(NDArray A, NDArray B);

/**
 * Stacks ndarrays along a new axis (all shapes must be identical).
 * Creates a new dimension at the specified position.
 * 
 * @param axis Position for the new dimension (0 to ndim)
 * @param arr_list NULL-terminated array of ndarrays (same shape required)
 * @return New ndarray with ndim+1 dimensions
 * 
 * Example:
 *   A=[2,3], B=[2,3], C=[2,3] -> stack(axis=0) -> [3, 2, 3]
 *   A=[2,3], B=[2,3] -> stack(axis=2) -> [2, 3, 2]
 */
NDArray ndarray_new_stack(int axis, NDArray* arr_list);

/**
 * Concatenates ndarrays along an existing axis.
 * All dimensions except the concatenation axis must match.
 * 
 * @param axis Axis along which to concatenate (0 to ndim-1)
 * @param arr_list List of ndarrays to concatenate
 * @return New ndarray with extended dimension along axis
 * 
 * Examples:
 *   A=[2,3,4], B=[2,5,4] -> concat(axis=1) -> [2, 8, 4]
 *   A=[3,3,4], B=[5,3,4] -> concat(axis=0) -> [8, 3, 4]
 */
NDArray ndarray_new_concat(int axis, NDArray* arr_list);

/**
 * Extract a subregion from an ndarray.
 * Creates a new ndarray containing elements from start to end indices
 * (exclusive).
 * 
 * @param arr Source ndarray
 * @param axis Axis along which to take the subregion
 * @param start Starting index (inclusive)
 * @param end Ending index (exclusive)
 * @return New ndarray with copied subregion
 * 
 * Examples:
 *   A=[4,5], axis=0, start=1, end=3 -> [2,5] (rows 1 and 2)
 *   A=[3,6], axis=1, start=2, end=5 -> [3,3] (columns 2,3,4)
 */
NDArray ndarray_new_take(NDArray arr, int axis, size_t start, size_t end);

/**
 * Creates a new ndarray that is the transpose of the given ndarray.
 * For N-dimensional arrays, reverses all axes (e.g., shape [2,3,4]
 * becomes [4,3,2]).
 *
 * @param A The input ndarray.
 * @return A handle to the newly created transposed ndarray.
 */
NDArray ndarray_new_transpose(NDArray A);

/**
 * Aggregation types for ndarray_new_axis_aggr function.
 */
enum {
    NDARRAY_AGGR_SUM = 0,
    NDARRAY_AGGR_MEAN,
    NDARRAY_AGGR_STD,
    NDARRAY_AGGR_MAX,
    NDARRAY_AGGR_MIN
};

/**
 * Creates a new ndarray by aggregating over a specified axis.
 *
 * Result maintains ndim >= 2 constraint:
 * - axis == NDA_ALL_AXES: returns shape [1, 1] with scalar result
 * - axis in [0, ndim-1]: if result would be 1D, adds dimension of 1
 *
 * @param A The input ndarray
 * @param axis The axis to aggregate over (0 to ndim-1), or NDA_ALL_AXES for all axes
 * @param aggr_type The type of aggregation to perform.
 * @return A handle to the result ndarray (ndim >= 2)
 * 
 * Example:
 *   ndarray_new_axis_aggr(A, 0, NDARRAY_AGGR_SUM)         // Sum along axis 0
 *   ndarray_new_axis_aggr(A, NDA_ALL_AXES, NDARRAY_AGGR_MEAN)  // Mean of all elements
 */
NDArray ndarray_new_axis_aggr(NDArray A, int axis, int aggr_type);

/**
 * Saves an ndarray to a binary file.
 * File format:
 *   - Magic number (uint32_t): 0x4E444152 ("NDAR" in ASCII)
 *   - Version (uint32_t): 1
 *   - Number of dimensions (uint64_t)
 *   - Dimension sizes (uint64_t array)
 *   - Data (double array, row-major order)
 *
 * @param arr The ndarray to save
 * @param filename Path to the output file (use .bin extension)
 * @return 0 on success, -1 on error
 *
 * Example:
 *   ndarray_save(arr, "mydata.bin");
 */
int ndarray_save(NDArray arr, const char *filename);

/**
 * Loads an ndarray from a binary file.
 * Reads the file format created by ndarray_save().
 *
 * @param filename Path to the input file (.bin extension)
 * @return A newly allocated ndarray, or NULL on error
 *
 * Example:
 *   NDArray arr = ndarray_load("mydata.bin");
 *   if (arr == NULL) {
 *       fprintf(stderr, "Failed to load array\n");
 *       return 1;
 *   }
 */
NDArray ndarray_load(const char *filename);

#endif // _NDARRAY_H

