# ndarray for C and Zig

A numpy-like ndarray library for C, with Zig bindings.

- [API Documentation](https://jailop.github.io/ndarray-c/)
- [Source Code](https://github.com/jailop/ndarray-c)
- [Zig](README-zig.md)

**Disclaimers**:

- This is a project for learning.
- The API can change at any moment.
- It is not intended for production use.
- Feedback is welcome

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
- Integrated with OpenMP for parallel operations (required)
- Uses CBLAS (OpenBLAS) for optimized linear algebra operations (required)

## Usage

**Helper Macros:**

The library uses macros that simplify array operations and make the code
more readable. These macros use C99's compound literal feature to create
temporary arrays.

`NDA_DIMS(...)`: To define a list of dimensions for array creation.

```c
// Create a 3x4 array
NDArray arr = ndarray_new(NDA_DIMS(3, 4));
```

`NDA_POS(...)`: A position index for accessing array elements.

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

`NDA_NO_AXES`: Special constant for tensor operations with no axis contraction.

```c
// Compute outer product
NDArray result = ndarray_new_tensordot(A, B, NDA_NO_AXES, NDA_NO_AXES);
```

`NDA_ALL_AXES`: Constant for operations on all axes (value: -1).

```c
// Aggregate over all axes
NDArray total = ndarray_new_aggr(A, NDA_ALL_AXES, NDA_AGGR_SUM);
NDArray max_val = ndarray_new_aggr(A, NDA_ALL_AXES, NDA_AGGR_MAX);
```

`NDA_LIST(...)`: Creates list of NDArray pointers.

```c
// Free multiple arrays at once
ndarray_free_all(NDA_LIST(A, B, C));
```

**Basic Creation:**

```c
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

**Creating Arrays from Literals:**

```c
// 2D array (matrix)
double data_2d[3][4] = {
    {1.0, 2.0, 3.0, 4.0},
    {5.0, 6.0, 7.0, 8.0},
    {9.0, 10.0, 11.0, 12.0}
};
NDArray arr_2d = ndarray_new_from_data(NDA_DIMS(3, 4), (double*)data_2d);

// 3D array (tensor)
double data_3d[2][3][4] = {
    {
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0}
    },
    {
        {13.0, 14.0, 15.0, 16.0},
        {17.0, 18.0, 19.0, 20.0},
        {21.0, 22.0, 23.0, 24.0}
    }
};
NDArray arr_3d = ndarray_new_from_data(NDA_DIMS(2, 3, 4), (double*)data_3d);

// 4D and higher dimensions are also supported
double data_4d[2][2][3][2] = { /* ... */ };
NDArray arr_4d = ndarray_new_from_data(NDA_DIMS(2, 2, 3, 2), (double*)data_4d);

ndarray_free_all(NDA_LIST(arr_2d, arr_3d, arr_4d));
```

**Saving and Loading Arrays:**

```c
// Save an array to file
NDArray arr = ndarray_new_randunif(NDA_DIMS(3, 4), 0.0, 1.0);
if (ndarray_save(arr, "mydata.bin") != 0) {
    fprintf(stderr, "Failed to save array\n");
}

// Load an array from file
NDArray loaded = ndarray_load("mydata.bin");
if (loaded == NULL) {
    fprintf(stderr, "Failed to load array\n");
    return 1;
}

ndarray_print(loaded, "Loaded Array", 4);
ndarray_free_all(NDA_LIST(arr, loaded));
```

Binary File Format:

- Magic number: `0x4E444152` ("NDAR" in ASCII)
- Version: `uint32_t` (currently 1)
- Number of dimensions: `uint64_t`
- Dimension sizes: array of `uint64_t`
- Data: array of `double` (row-major order)

**Getting and Setting Values:**

```c
NDArray arr = ndarray_new_zeros(NDA_DIMS(3, 4));

// Set value at position (1, 2)
ndarray_set(arr, NDA_POS(1, 2), 42.0);

// Get value at position (1, 2)
double val = ndarray_get(arr, NDA_POS(1, 2));

// Set an entire row (axis 0) from an array of values
double row_data[] = {1.0, 2.0, 3.0, 4.0};
ndarray_set_slice(arr, 0, 1, row_data);  // Set row 1

// Set an entire column (axis 1) from an array of values
double col_data[] = {10.0, 20.0, 30.0};
ndarray_set_slice(arr, 1, 2, col_data);  // Set column 2

// Fill an entire row with a scalar value
ndarray_fill_slice(arr, 0, 0, 0.0);  // Fill row 0 with zeros

// Fill an entire column with a scalar value
ndarray_fill_slice(arr, 1, 3, 99.0);  // Fill column 3 with 99.0

// Works for any dimension - set a hyperplane in 3D array
NDArray arr3d = ndarray_new_zeros(NDA_DIMS(2, 3, 4));
double plane_data[8];  // 2*4 = 8 values for plane perpendicular to axis 1
for (int i = 0; i < 8; i++) plane_data[i] = i * 5.0;
ndarray_set_slice(arr3d, 1, 1, plane_data);  // Set middle plane

ndarray_free_all(NDA_LIST(arr, arr3d));
```

// Print the array
ndarray_print(arr, "My Array", 4);  // precision = 4 decimal places

ndarray_free(arr);
```

**Element-wise Operations:**

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

// Linear combination: A = alpha*A + beta*B
NDArray C = ndarray_new_full(NDA_DIMS(3, 3), 5.0);
NDArray D = ndarray_new_full(NDA_DIMS(3, 3), 3.0);
ndarray_axpby(C, 2.0, D, 3.0);  // C = 2*C + 3*D = 2*5 + 3*3 = 19

// Fused operations for better performance
ndarray_scale_shift(C, 0.5, 10.0);  // C = 0.5*C + 10 (scale and shift)
ndarray_mul_scaled(C, D, 2.0);       // C = C * D * 2 (multiply then scale)

// Map function then multiply (common in numerical computing)
double sqrt_fn(double x) { return sqrt(x); }
ndarray_map_mul(C, sqrt_fn, D, 0.5);  // C = sqrt(C) * D * 0.5

// Fused multiply-add
NDArray E = ndarray_new_ones(NDA_DIMS(3, 3));
ndarray_mul_add(C, D, E, 2.0, 1.0);  // E = 2*(C*D) + E

// Apply custom function element-wise
double square(double x) { return x * x; }
ndarray_mapfnc(A, square);

ndarray_free_all(NDA_LIST(A, B, C, D, E));
```

**Matrix Operations:**

```c
// Matrix multiplication
NDArray A = ndarray_new_ones(NDA_DIMS(3, 4));
NDArray B = ndarray_new_ones(NDA_DIMS(4, 5));
NDArray C = ndarray_new_matmul(A, B);  // Result: 3x5

// Transpose
NDArray At = ndarray_new_transpose(A);

// Matrix-vector multiply: y = alpha * A * x + beta * y
NDArray x = ndarray_new_full(NDA_DIMS(4, 1), 2.0);  // Column vector
NDArray y = ndarray_new_zeros(NDA_DIMS(3, 1));
ndarray_gemv(A, x, 1.0, 0.0, y);  // y = A * x

// Tensor contraction
NDArray X = ndarray_new(NDA_DIMS(2, 3, 4));
NDArray Y = ndarray_new(NDA_DIMS(4, 5));
// Contract on axis 2 of X and axis 0 of Y
NDArray Z = ndarray_new_tensordot(X, Y, NDA_AXES(2), NDA_AXES(0));  

ndarray_free_all(NDA_LIST(A, B, C, At, x, y, X, Y, Z));
```

**Conditional Operations:**

```c
// Non-negativity constraints (common in physics, finance, ML)
NDArray values = ndarray_new_arange(NDA_DIMS(3, 3), -4.0, 5.0, 1.0);
ndarray_clip_min(values, 0.0);  // ReLU activation, non-negative constraints

// Saturation / capping
ndarray_clip_max(values, 100.0);  // Prevent overflow

// Normalize to range [0, 1]
ndarray_clip(values, 0.0, 1.0);  // Probabilities, normalized ranges

// Absolute value and sign
NDArray errors = ndarray_new_arange(NDA_DIMS(2, 3), -2.0, 4.0, 1.0);
ndarray_abs(errors);   // Distance calculations, L1 norm
ndarray_sign(errors);  // Direction indicators

ndarray_free_all(NDA_LIST(values, errors));
```

**Comparison and Logical Operations:**

```c
// Element-wise comparisons (returns 1.0 for true, 0.0 for false)
NDArray A = ndarray_new_arange(NDA_DIMS(2, 3), 0.0, 6.0, 1.0);
NDArray B = ndarray_new_full(NDA_DIMS(2, 3), 3.0);

NDArray eq = ndarray_new_equal(A, B);           // A == B
NDArray lt = ndarray_new_less(A, B);            // A < B
NDArray gt = ndarray_new_greater(A, B);         // A > B

// Scalar comparisons
NDArray positive = ndarray_new_greater_scalar(A, 0.0);  // Find positive values
NDArray zeros = ndarray_new_equal_scalar(A, 0.0);       // Find zeros

// Logical operations
NDArray mask1 = ndarray_new_greater_scalar(A, 2.0);
NDArray mask2 = ndarray_new_less_scalar(A, 5.0);
NDArray combined = ndarray_logical_and(mask1, mask2);   // 2 < A < 5

// NumPy-style where (ternary operator)
NDArray x = ndarray_new_full(NDA_DIMS(2, 3), 10.0);
NDArray y = ndarray_new_full(NDA_DIMS(2, 3), -10.0);
NDArray result = ndarray_where(positive, x, y);  // positive ? x : y

ndarray_free_all(NDA_LIST(A, B, eq, lt, gt, positive, zeros, 
                          mask1, mask2, combined, x, y, result));
```

**Slice Access (Advanced):**

```c
// Direct pointer access for advanced users
NDArray matrix = ndarray_new_arange(NDA_DIMS(4, 5), 0.0, 20.0, 1.0);

// Get pointer to row 2 (5 elements)
double* row2 = ndarray_get_slice_ptr(matrix, 0, 2);
printf("Row 2: [%.1f, %.1f, %.1f, %.1f, %.1f]\n", 
       row2[0], row2[1], row2[2], row2[3], row2[4]);

// Copy slices between arrays
NDArray dest = ndarray_new_zeros(NDA_DIMS(4, 5));
ndarray_copy_slice(matrix, 0, 1, dest, 0, 3);  // Copy row 1 to row 3

// Get slice size for allocation
size_t row_size = ndarray_get_slice_size(matrix, 0);  // 5 elements per row
size_t col_size = ndarray_get_slice_size(matrix, 1);  // 4 elements per column

ndarray_free_all(NDA_LIST(matrix, dest));
```

**Array Manipulation:**

```c
// Stack arrays along a new axis
NDArray A = ndarray_new_ones(NDA_DIMS(2, 3));
NDArray B = ndarray_new_zeros(NDA_DIMS(2, 3));
NDArray stacked = ndarray_new_stack(0, NDA_LIST(A, B));  // Result: 2x2x3

// Concatenate arrays along an existing axis
NDArray concat = ndarray_new_concat(1, NDA_LIST(A, B));  // Result: 2x6

// Extract subregion
NDArray sub = ndarray_new_take(A, 1, 0, 2);  // Take columns 0 and 1

// Reshape array in-place (data preserved in row-major order)
NDArray C = ndarray_new_arange(NDA_DIMS(2, 6), 0, 12, 1);  // [2,6]
ndarray_reshape(C, NDA_DIMS(3, 4));  // Now [3,4]
ndarray_reshape(C, NDA_DIMS(2, 2, 3));  // Now [2,2,3]
ndarray_reshape(C, NDA_DIMS(4, -1));  // Now [4,3] (inferred dimension)

ndarray_free_all(NDA_LIST(A, B, stacked, concat, sub, C));
```

**Aggregations:**

```c
NDArray A = ndarray_new_randunif(NDA_DIMS(3, 4), 0.0, 10.0);

// Aggregate along specific axis
NDArray sum_axis0 = ndarray_new_aggr(A, 0, NDA_AGGR_SUM);
NDArray mean_axis1 = ndarray_new_aggr(A, 1, NDA_AGGR_MEAN);
NDArray std_axis0 = ndarray_new_aggr(A, 0, NDA_AGGR_STD);

// Aggregate over all axes using NDA_ALL_AXES constant
NDArray max_all = ndarray_new_aggr(A, NDA_ALL_AXES, NDA_AGGR_MAX);
NDArray min_all = ndarray_new_aggr(A, NDA_ALL_AXES, NDA_AGGR_MIN);

ndarray_free_all(NDA_LIST(A, sum_axis0, mean_axis1, std_axis0, max_all, min_all));
```

**Example:**

```c
#include <stdio.h>
#include <math.h>
#include "ndarray.h"

int main() {
    // Create matrices from literals
    double m1_data[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    double m2_data[3][2] = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
    
    NDArray A = ndarray_new_from_data(NDA_DIMS(2, 3), (double*)m1_data);
    NDArray B = ndarray_new_from_data(NDA_DIMS(3, 2), (double*)m2_data);
    
    ndarray_print(A, "Matrix A", 1);
    ndarray_print(B, "Matrix B", 1);
    
    // Matrix multiplication
    NDArray C = ndarray_new_matmul(A, B);
    ndarray_print(C, "A @ B", 1);
    
    // Save result
    if (ndarray_save(C, "result.bin") == 0) {
        printf("Result saved to result.bin\n");
    }
    
    // Load and verify
    NDArray loaded = ndarray_load("result.bin");
    if (loaded != NULL) {
        ndarray_print(loaded, "Loaded Result", 1);
        ndarray_free(loaded);
    }
    
    // Clean up
    ndarray_free_all(NDA_LIST(A, B, C));
    return 0;
}
```

Output:

```
Array 'Matrix A' [2, 3]:
[[    1.0     2.0     3.0]
 [    4.0     5.0     6.0]]
Array 'Matrix B' [3, 2]:
[[    7.0     8.0]
 [    9.0    10.0]
 [   11.0    12.0]]
Array 'A @ B' [2, 2]:
[[   58.0    64.0]
 [  139.0   154.0]]
Result saved to result.bin
Array 'Loaded Result' [2, 2]:
[[   58.0    64.0]
 [  139.0   154.0]]
```

## Building

Requirements:

- OpenMP: For parallel operations (required)
- OpenBLAS: For optimized BLAS operations (required)
- CUnit: For running tests (optional, only needed for tests)

### Build System Options

The library supports three build systems for maximum cross-platform
compatibility:

#### 1. CMake (Recommended for cross-platform)

```bash
# Configure (auto-detects GCC and Homebrew libraries on macOS)
mkdir build && cd build
cmake ..

# Build everything
cmake --build .

# Build specific targets
cmake --build . --target ndarray_static    # Static library
cmake --build . --target ndarray_shared    # Shared library
cmake --build . --target example           # Example program
cmake --build . --target ndarray_test      # Test suite
cmake --build . --target benchmark_seq     # Sequential benchmark
cmake --build . --target benchmark_omp     # OpenMP benchmark

# Install
sudo cmake --build . --target install

# Run tests
ctest
```

CMake options:
```bash
cmake -DBUILD_SHARED_LIBS=ON      # Build shared library (default: ON)
cmake -DBUILD_STATIC_LIBS=ON      # Build static library (default: ON)
cmake -DBUILD_EXAMPLES=ON         # Build examples (default: ON)
cmake -DBUILD_TESTS=OFF           # Build tests (default: ON)
cmake -DBUILD_BENCHMARKS=ON       # Build benchmarks (default: ON)
cmake -DCMAKE_INSTALL_PREFIX=/usr # Install location
```

#### 2. Zig Build System

```bash
# Build examples
zig build                          # Build examples
zig build run                      # Run basic example
zig build run-extended             # Run extended example

# Build libraries
zig build lib                      # Build both static and shared
zig build static                   # Build static library only
zig build shared                   # Build shared library only

# Tests
zig build test                     # Run Zig tests
zig build test-c                   # Run C tests (requires CUnit)
zig build test-all                 # Run all tests

# Benchmarks
zig build bench                    # Build benchmark executables
zig build run-bench-seq            # Run sequential benchmark
zig build run-bench-omp            # Run OpenMP benchmark
```

Libraries and headers are installed to `zig-out/lib/` and `zig-out/include/`.

#### 3. Traditional Makefile

```bash
# Build example
make

# Build libraries
make lib                           # Build both static and shared
make static                        # Build static library only
make shared                        # Build shared library only

# Install
sudo make install                  # Installs to /usr/local by default
sudo make install PREFIX=/usr      # Custom install location

# Tests and benchmarks
make test                          # Run test suite
make benchmark                     # Run benchmarks
```

### Running Benchmarks

The benchmark script automatically detects and uses the available build system:

```bash
cd benchmarks
./run_benchmark.sh
```

This will:
- Build sequential and OpenMP versions
- Run both benchmarks
- Generate a performance comparison report
- Show speedup for each operation

Run tests:

### Installing Dependencies

**macOS (Homebrew):**
```bash
brew install gcc libomp openblas cunit
```

**Ubuntu/Debian:**
```bash
sudo apt-get install gcc libomp-dev libopenblas-dev libcunit1-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install gcc libomp-devel openblas-devel CUnit-devel
```

## Documentation

Generate API documentation:

```bash
make docs
# Open docs/html/index.html in a browser
```

Requires Doxygen to be installed for local generation.

OpenMP Support:

OpenMP is required by default. All builds include OpenMP support:

```bash
# Build with OpenMP support (default)
make

# Build library with OpenMP (default)
make lib

# Build tests with OpenMP (default)
make test
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
# Debug build (default - assertions enabled)
make

# Release build (assertions disabled for performance)
make CFLAGS="-O3 -DNDEBUG -std=c99 -march=native -fopenmp"
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
# Custom optimization level
make CFLAGS="-O2 -Wall -std=c99 -fopenmp"

# Enable sanitizers for debugging
make CFLAGS="-O0 -g -fsanitize=address -std=c99 -fopenmp"

# Profile-guided optimization
make CFLAGS="-O3 -fprofile-generate -std=c99 -fopenmp"
```

## Building Your Program

After installing the library, compile your program:

```bash
# Using installed library (CMake install or Makefile install)
gcc -fopenmp -o myprogram myprogram.c -I/usr/local/include -L/usr/local/lib -lndarray -lopenblas -lm

# On macOS with Homebrew
gcc-15 -fopenmp -o myprogram myprogram.c \
    -I/usr/local/include \
    -I/opt/homebrew/opt/libomp/include \
    -I/opt/homebrew/opt/openblas/include \
    -L/usr/local/lib \
    -L/opt/homebrew/opt/libomp/lib \
    -L/opt/homebrew/opt/openblas/lib \
    -lndarray -lopenblas -lgomp -lm

# Or link directly with source files (no installation needed)
gcc -fopenmp -o myprogram myprogram.c -Isrc src/ndarray_*.c -lopenblas -lm
```

### Using with CMake

Create a `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject C)

find_package(ndarray REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(myprogram myprogram.c)
target_link_libraries(myprogram PRIVATE ndarray::ndarray OpenMP::OpenMP_C)
```

Then build:
```bash
mkdir build && cd build
cmake ..
cmake --build .
```
