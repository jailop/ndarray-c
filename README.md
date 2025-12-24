# ndarray for C

A numpy-like ndarray library for C.

**Disclaimer**: This is a project for learning. The API can change at any
moment. It is not intended for production use. Feedback is welcome.

## Design Considerations

- The priority is a clean and simple API over performance
- Provides helper macros for list manipulation to express dimensions,
  positions, axes, and many ndarrays.
- Uses assertions for runtime checks; can be disabled with `NDEBUG`
- All arrays must have at least 2 dimensions (ndim >= 2)
- For 1D arrays, use shape `[1, n]` or `[n, 1]`
- For most of the operations, it is prefered to overwrite existing
  arrays instead of creating new ones (in place operations)
- Operations that create new arrays have `_new_` in their name (e.g.,
  `ndarray_new_matmul`). In that way, it is easy to identify which
  objects need to be freed later.
- Always free arrays with `ndarray_free()` or `ndarray_free_all()` to
  avoid memory leaks
- Uses C99 standard; requires C99-compatible compiler
- Uses `double` as the default data type for array elements
- Uses row-major order for array storage
- Supports OpenMP for parallel operations (optional)

## Usage

Helper Macros:

The library uses macros that simplify array operations and make the code
more readable. These macros use C99's compound literal feature to create
temporary arrays.

`NDA_DIMS(...)`: To define a list of dimensions for array creation.

```c
// Create a 3x4 array
NDArray arr = ndarray_new(NDA_DIMS(3, 4));
```

`NDA_POS(...)`: A position list for accessing array elements.

```c
// Set value at position (1, 2)
ndarray_set(arr, NDA_POS(1, 2), 42.0);
```

`NDA_AXES(...)`: A list of axes for tensor operations.

```c
// Contract on axes 2 of A and axis 0 of B
NDArray result = ndarray_new_tensordot(A, B, NDA_AXES(2), NDA_AXES(0));

// Contract on multiple axes
NDArray result = ndarray_new_tensordot(A, B, NDA_AXES(1, 2), NDA_AXES(0, 1));
```

`NDA_NOAXES`: Special constant for tensor operations with no axis contraction.

```c
// Compute outer product
NDArray result = ndarray_new_tensordot(A, B, NDA_NOAXES, NDA_NOAXES);
```

`NDA_LIST(...)`: Creates list of NDArray pointers.

```c
// Free multiple arrays at once
ndarray_free_all(NDA_LIST(A, B, C));
```

`NDA_AXES_ALL`: Constant for operations on all axes (value: -1).

```c
// Aggregate over all axes
NDArray total = ndarray_new_axis_aggr(A, NDA_AXES_ALL, NDARRAY_AGGR_SUM);
NDArray max_val = ndarray_new_axis_aggr(A, NDA_AXES_ALL, NDARRAY_AGGR_MAX);

// More readable than using -1 directly
NDArray mean = ndarray_new_axis_aggr(A, NDA_AXES_ALL, NDARRAY_AGGR_MEAN);
```

Basic Array Creation:

```c
#include "ndarray.h"

// Create a 3x4 array
NDArray arr = ndarray_new(NDA_DIMS(3, 4));

// Create arrays filled with values
NDArray zeros = ndarray_new_zeros(NDA_DIMS(2, 3));
NDArray ones = ndarray_new_ones(NDA_DIMS(2, 3));
NDArray filled = ndarray_new_full(NDA_DIMS(2, 3), 5.0);

// Create arrays with sequences
NDArray range = ndarray_new_arange(NDA_DIMS(1, 10), 0.0, 10.0, 1.0);
NDArray linsp = ndarray_new_linspace(NDA_DIMS(1, 100), 0.0, 1.0, 100);

// Create arrays with random values
NDArray norm = ndarray_new_randnorm(NDA_DIMS(3, 3), 0.0, 1.0);  // mean=0, std=1
NDArray unif = ndarray_new_randunif(NDA_DIMS(3, 3), 0.0, 1.0);  // min=0, max=1

// Always free arrays when done
ndarray_free(arr);
ndarray_free_all(NDA_LIST(zeros, ones, filled, range, linsp, norm, unif));
```

Getting and Setting Values:

```c
NDArray arr = ndarray_new_zeros(NDA_DIMS(3, 4));

// Set value at position (1, 2)
ndarray_set(arr, NDA_POS(1, 2), 42.0);

// Get value at position (1, 2)
double val = ndarray_get(arr, NDA_POS(1, 2));

// Print the array
ndarray_print(arr, "My Array", 4);  // precision = 4 decimal places

ndarray_free(arr);
```

Element-wise Operations:

```c
NDArray A = ndarray_new_ones(NDA_DIMS(3, 3));
NDArray B = ndarray_new_full(NDA_DIMS(3, 3), 2.0);

// Element-wise addition (modifies A in place)
ndarray_add(A, B);

// Element-wise multiplication (modifies A in place)
ndarray_mul(A, B);

// Scalar operations
ndarray_add_scalar(A, 10.0);
ndarray_mul_scalar(A, 2.0);

// Apply custom function element-wise
double square(double x) { return x * x; }
ndarray_mapfnc(A, square);

ndarray_free_all(NDA_LIST(A, B));
```

Matrix Operations:

```c
// Matrix multiplication
NDArray A = ndarray_new_ones(NDA_DIMS(3, 4));
NDArray B = ndarray_new_ones(NDA_DIMS(4, 5));
NDArray C = ndarray_new_matmul(A, B);  // Result: 3x5

// Transpose
NDArray At = ndarray_new_transpose(A);

// Tensor contraction
NDArray X = ndarray_new(NDA_DIMS(2, 3, 4));
NDArray Y = ndarray_new(NDA_DIMS(4, 5));
// Contract on axis 2 of X and axis 0 of Y
NDArray Z = ndarray_new_tensordot(X, Y, NDA_AXES(2), NDA_AXES(0));  

ndarray_free_all(NDA_LIST(A, B, C, At, X, Y, Z));
```

Array Manipulation:

```c
// Stack arrays along a new axis
NDArray A = ndarray_new_ones(NDA_DIMS(2, 3));
NDArray B = ndarray_new_zeros(NDA_DIMS(2, 3));
NDArray stacked = ndarray_new_stack(0, NDA_LIST(A, B));  // Result: 2x2x3

// Concatenate arrays along an existing axis
NDArray concat = ndarray_new_concat(1, NDA_LIST(A, B));  // Result: 2x6

// Extract subregion
NDArray sub = ndarray_new_take(A, 1, 0, 2);  // Take columns 0 and 1

ndarray_free_all(NDA_LIST(A, B, stacked, concat, sub));
```

Aggregations:

```c
NDArray A = ndarray_new_randunif(NDA_DIMS(3, 4), 0.0, 10.0);

// Aggregate along specific axis
NDArray sum_axis0 = ndarray_new_axis_aggr(A, 0, NDARRAY_AGGR_SUM);
NDArray mean_axis1 = ndarray_new_axis_aggr(A, 1, NDARRAY_AGGR_MEAN);
NDArray std_axis0 = ndarray_new_axis_aggr(A, 0, NDARRAY_AGGR_STD);

// Aggregate over all axes using NDA_AXES_ALL constant
NDArray max_all = ndarray_new_axis_aggr(A, NDA_AXES_ALL, NDARRAY_AGGR_MAX);
NDArray min_all = ndarray_new_axis_aggr(A, NDA_AXES_ALL, NDARRAY_AGGR_MIN);

ndarray_free_all(NDA_LIST(A, sum_axis0, mean_axis1, std_axis0, max_all, min_all));
```

Complete Example:

```c
#include <stdio.h>
#include <math.h>
#include "ndarray.h"

int main() {
    // Create a 3x3 matrix
    NDArray A = ndarray_new_arange(NDA_DIMS(3, 3), 1.0, 10.0, 1.0);
    ndarray_print(A, "Matrix A", 2);
    
    // Create identity-like matrix
    NDArray B = ndarray_new_ones(NDA_DIMS(3, 3));
    
    // Multiply and add
    ndarray_mul(A, B);
    ndarray_add_scalar(A, 5.0);
    ndarray_print(A, "Result", 2);
    
    // Matrix multiplication
    NDArray C = ndarray_new_matmul(A, B);
    ndarray_print(C, "Matrix Product", 2);
    
    // Cleanup
    ndarray_free_all(NDA_LIST(A, B, C));
    
    return 0;
}
```

## Building

Build the example program:

```bash
cd src
make
```

Build the library:

```bash
cd src
make lib                    # Build both static and shared libraries
make static                 # Build static library only
make shared                 # Build shared library only
```

Install the library:

```bash
cd src
sudo make install          # Installs to /usr/local by default
# Or specify custom prefix:
sudo make install PREFIX=/usr
```

Run tests:

```bash
cd src
make test
```

Generate API documentation:

```bash
cd src
make docs
# Open ../docs/html/index.html in a browser
```

The project uses Doxygen to generate HTML documentation from source code comments. The documentation includes:
- API reference for all functions
- Struct and type definitions
- Macro documentation
- Usage examples from the README

**Note:** Requires Doxygen to be installed (`apt install doxygen` on Debian/Ubuntu).

OpenMP Support:

Enable OpenMP for parallel operations in aggregations, transpose, and
some element-wise operations:

```bash
cd src
# Build with OpenMP support
make USE_OPENMP=1

# Build library with OpenMP
make lib USE_OPENMP=1

# Build tests with OpenMP
make test USE_OPENMP=1
```

When compiled with OpenMP:

- Aggregation operations (sum, mean, std, max, min) are parallelized
- Transpose operations are parallelized
- Large matrix operations benefit from multi-threading
- Performance improvements are most noticeable with arrays larger than 1000x1000

Requires OpenMP-compatible compiler (gcc, clang with libomp).

Debug vs Release Builds:

The library uses assertions for runtime checks. Control them with `NDEBUG`.

```bash
cd src
# Debug build (default - assertions enabled)
make

# Release build (assertions disabled for performance)
make CFLAGS="-O3 -DNDEBUG -std=c99 -march=native"

# Release build with OpenMP
make USE_OPENMP=1 CFLAGS="-O3 -DNDEBUG -fopenmp -DUSE_OPENMP -std=c99 -march=native"
```

Debug builds (without `-DNDEBUG`):

- Enable runtime assertions for parameter validation
- Check array dimensions, axis bounds, NULL pointers
- Slower but safer for development

Release builds (with `-DNDEBUG`):

- Disable assertions for maximum performance
- About 5-10% faster for small operations
- Use only with well-tested code

Custom Compiler Flags:

You can override the default flags:

```bash
cd src
# Custom optimization level
make CFLAGS="-O2 -Wall -std=c99"

# Enable sanitizers for debugging
make CFLAGS="-O0 -g -fsanitize=address -std=c99"

# Profile-guided optimization
make CFLAGS="-O3 -fprofile-generate -std=c99"
```

## Building Your Program

Compile your program with the ndarray library:

```bash
# Using static library
gcc -static -o myprogram myprogram.c -lndarray -lm

# Using shared library
gcc -o myprogram myprogram.c -lndarray -lm

# Or link directly with object file
gcc -o myprogram myprogram.c ndarray.c -lm
```
