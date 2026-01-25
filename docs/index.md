# ndarray-c Documentation

A numpy-like ndarray library for C with multi-dimensional arrays, OpenMP parallelization, and BLAS-optimized operations. Also includes native Zig bindings.

## Overview

ndarray-c provides high-performance N-dimensional array operations with a clean API for both C and Zig. The library is designed for numerical computing with support for parallel operations and optimized linear algebra.

## Features

- **Multi-dimensional arrays** (ndim >= 2)
- **OpenMP parallelization** for performance
- **BLAS-optimized operations** for linear algebra
- **Native C API** with simple macros
- **Idiomatic Zig bindings** with error handling and memory safety
- **Extensive operations** - arithmetic, linear algebra, aggregation, comparison
- **I/O support** - save/load binary format

## Quick Start

### C Example

```c
#include "ndarray.h"

int main() {
    // Create a 2x3 array of ones
    NDArray arr = ndarray_new_ones(NDA_DIMS(2, 3));
    
    // Set value at position (1, 2)
    ndarray_set(arr, NDA_POS(1, 2), 42.0);
    
    // Print the array
    ndarray_print(arr, "My Array", 2);
    
    // Clean up
    ndarray_free(arr);
    return 0;
}
```

### Zig Example

```zig
const ndarray = @import("ndarray");
const NDArray = ndarray.NDArray;

pub fn main() !void {
    // Create arrays
    const a = try NDArray.ones(&.{2, 3});
    defer a.deinit();
    
    const b = try NDArray.full(&.{2, 3}, 2.0);
    defer b.deinit();
    
    // Operations
    _ = a.add(b);  // a = a + b
    a.print("Result", 2);
}
```

## API Documentation

### [C API Reference](c-api/index.html)

Complete C API documentation generated with Doxygen, including:
- Data structures and types
- Array creation functions
- Arithmetic operations
- Linear algebra operations
- Array manipulation
- Aggregation functions
- I/O operations

### [Zig API Reference](zig-api/index.html)

Complete Zig API documentation generated with `zig build-lib`, including:
- `NDArray` struct and methods
- Type-safe wrappers
- Error handling patterns
- Memory management
- Examples and usage patterns

## User Guides

### For C Users

- [Usage Guide](../guide/usage.md) - Getting started with C API
- [Building](../guide/building.md) - Compilation and installation
- [Advanced Topics](../guide/advanced.md) - Performance tuning and best practices

### For Zig Users

- [Zig Usage Guide](../guide/zig-usage.md) - Getting started with Zig bindings
- [Zig Building](../guide/zig-building.md) - Building with Zig build system

## Installation

### Building from Source (C)

```bash
git clone https://github.com/jailop/ndarray-c.git
cd ndarray-c
mkdir build && cd build
cmake ..
make
sudo make install
```

### Using with Zig

Add to your `build.zig.zon`:

```zig
.dependencies = .{
    .ndarray = .{
        .url = "https://github.com/jailop/ndarray-c/archive/refs/tags/v0.2.0.tar.gz",
    },
}
```

## Key Concepts

### Memory Management

**C**: Manual memory management with explicit `ndarray_free()`:

```c
NDArray arr = ndarray_new_ones(NDA_DIMS(2, 3));
// Use arr...
ndarray_free(arr);  // Don't forget!
```

**Zig**: Automatic with `defer` or manual with `deinit()`:

```zig
const arr = try NDArray.ones(&.{2, 3});
defer arr.deinit();  // Cleanup when scope exits
```

### Operation Types

1. **In-place operations** - Modify the first argument and return it:
   ```c
   ndarray_add(a, b);           // a = a + b
   ndarray_mul_scalar(a, 2.0);  // a = a * 2.0
   ```

2. **New array operations** - Return newly allocated arrays:
   ```c
   NDArray c = ndarray_new_matmul(a, b);  // c = a @ b
   NDArray d = ndarray_new_transpose(a);  // d = transpose(a)
   ```

3. **Scalar operations** - Return scalar values:
   ```c
   double sum = ndarray_scalar_aggregate(a, NDA_AGGR_SUM);
   double mean = ndarray_scalar_aggregate(a, NDA_AGGR_MEAN);
   ```

### Thread Safety

Operations use OpenMP for parallelization. Thread safety depends on:
- Not modifying the same array from multiple threads simultaneously
- Understanding shared vs. private data in OpenMP regions

## Main API Categories

### Array Creation
- `ndarray_new_zeros`, `ndarray_new_ones`, `ndarray_new_full`
- `ndarray_new_from_data`
- `ndarray_new_random_uniform`, `ndarray_new_random_normal`
- `ndarray_new_arange`, `ndarray_new_linspace`

### Element Access
- `ndarray_get(arr, pos)`, `ndarray_set(arr, pos, value)`
- `ndarray_set_slice`, `ndarray_fill_slice`
- `ndarray_get_slice_ptr`, `ndarray_copy_slice`

### Arithmetic Operations (In-place)
- `ndarray_add`, `ndarray_mul` - element-wise operations
- `ndarray_add_scalar`, `ndarray_mul_scalar` - scalar operations
- `ndarray_axpby` - linear combination: a*x + b*y
- `ndarray_scale_shift` - a*x + b
- `ndarray_clip_min`, `ndarray_clip_max`, `ndarray_clip`
- `ndarray_abs`, `ndarray_sign`

### Comparison and Logical (Returns new array)
- `ndarray_new_equal`, `ndarray_new_less`, `ndarray_new_greater`
- `ndarray_new_equal_scalar`, `ndarray_new_less_scalar`, `ndarray_new_greater_scalar`
- `ndarray_new_logical_and`, `ndarray_new_logical_or`, `ndarray_new_logical_not`
- `ndarray_new_where` - conditional selection

### Linear Algebra (Returns new array)
- `ndarray_new_matmul` - matrix multiplication
- `ndarray_new_tensordot` - tensor contraction
- `ndarray_new_transpose` - transpose

### Array Manipulation
- `ndarray_reshape` - reshape in-place
- `ndarray_new_take` - extract slice
- `ndarray_new_stack` - stack arrays along new axis
- `ndarray_new_concat` - concatenate arrays

### Aggregation
- `ndarray_new_aggregate(arr, axis, type)` - aggregate along axis
- `ndarray_scalar_aggregate(arr, type)` - aggregate all elements
- Types: `NDA_AGGR_SUM`, `NDA_AGGR_MEAN`, `NDA_AGGR_STD`, `NDA_AGGR_MAX`, `NDA_AGGR_MIN`

### I/O Operations
- `ndarray_save(arr, filename)` - save to binary file
- `ndarray_new_load(filename)` - load from binary file

### Properties
- `ndarray_ndim(arr)` - number of dimensions
- `ndarray_shape(arr)` - dimension sizes

## Examples

### Matrix Multiplication

```c
#include "ndarray.h"

int main() {
    NDArray a = ndarray_new_ones(NDA_DIMS(2, 3));
    NDArray b = ndarray_new_ones(NDA_DIMS(3, 4));
    NDArray c = ndarray_new_matmul(a, b);
    
    ndarray_print(c, "Result", 2);
    // Result [2, 4]:
    // [[    3.00     3.00     3.00     3.00]
    //  [    3.00     3.00     3.00     3.00]]
    
    ndarray_free(a);
    ndarray_free(b);
    ndarray_free(c);
    return 0;
}
```

### Aggregation

```c
#include "ndarray.h"

int main() {
    NDArray arr = ndarray_new_arange(NDA_DIMS(3, 4), 0.0, 12.0, 1.0);
    
    // Sum along axis 0
    NDArray sum_axis0 = ndarray_new_aggregate(arr, 0, NDA_AGGR_SUM);
    ndarray_print(sum_axis0, "Sum axis 0", 2);
    
    // Mean of all elements
    double mean = ndarray_scalar_aggregate(arr, NDA_AGGR_MEAN);
    printf("Mean: %.2f\n", mean);  // 5.50
    
    ndarray_free(arr);
    ndarray_free(sum_axis0);
    return 0;
}
```

### Conditional Operations

```c
#include "ndarray.h"

int main() {
    NDArray data = ndarray_new_arange(NDA_DIMS(2, 3), 0.0, 6.0, 1.0);
    NDArray mask = ndarray_new_greater_scalar(data, 2.5);
    NDArray zeros = ndarray_new_zeros(NDA_DIMS(2, 3));
    NDArray filtered = ndarray_new_where(mask, data, zeros);
    
    ndarray_print(filtered, "Filtered", 2);
    // Elements > 2.5 kept, others set to 0
    
    ndarray_free(data);
    ndarray_free(mask);
    ndarray_free(zeros);
    ndarray_free(filtered);
    return 0;
}
```

## Performance Tips

1. **Use in-place operations** when you don't need to preserve the original array
2. **Compile with optimizations**: `-O3 -march=native -fopenmp`
3. **Link with optimized BLAS**: OpenBLAS or Intel MKL for best performance
4. **Avoid unnecessary copies** - reuse arrays when possible
5. **Use appropriate aggregation axis** - operations along memory-contiguous dimensions are faster

## Building and Testing

### C Build

```bash
# CMake build
mkdir build && cd build
cmake ..
make
make test

# Or with Make
make lib
make test
make docs
```

### Zig Build

```bash
# Build library
zig build lib

# Run tests
zig build test        # Zig tests
zig build test-c      # C tests (requires CUnit)
zig build test-all    # All tests

# Run examples
zig build run
zig build run-extended

# Generate documentation
zig build-lib --docs src/ndarray.zig
```

## Generating Documentation

### C Documentation (Doxygen)

```bash
doxygen Doxyfile
# Output in docs/html/
```

### Zig Documentation

```bash
zig build-lib -femit-docs -fno-emit-bin src/ndarray.zig
# Output in docs/zig-api/
```

## Important Notes

- **Minimum dimension**: All arrays must have `ndim >= 2`
- **Type**: The library works with `double` precision floats
- **Memory**: Arrays use C heap allocation
- **Error handling**: C uses assertions; Zig uses error returns

## Troubleshooting

### Library not found

If you get linking errors:

```bash
# For CMake
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..

# For manual compilation
gcc -o myapp myapp.c -lndarray -L/usr/local/lib -I/usr/local/include

# Or add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### Shape mismatches

Check array shapes before operations:

```c
printf("Shape: [%zu, %zu]\n", arr->shape[0], arr->shape[1]);
printf("Dimensions: %d\n", arr->ndim);
```

## Links

- [GitHub Repository](https://github.com/jailop/ndarray-c)
- [C API Reference](c-api/index.html)
- [Zig API Reference](zig-api/index.html)
- [Nim Bindings](https://github.com/jailop/ndarray-c-nim)
- [Design Considerations](../guide/design.md)
- [GitHub Issues](https://github.com/jailop/ndarray-c/issues)

## Disclaimers

- This is a project for learning
- The API can change at any moment
- It is not intended for production use
- Feedback is welcomed

## License

BSD 3-Clause License. See [LICENSE](../LICENSE) file for details.
