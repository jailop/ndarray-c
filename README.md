# ndarray for C and Zig

A numpy-like ndarray library for C, with Zig bindings.

## Features

- Multi-dimensional arrays (ndim >= 2)
- NumPy-like API with convenient macros
- Element-wise operations with function chaining
- Matrix multiplication
- Tensor operations (stack, concat, tensordot)
- Aggregations (sum, mean, max, min, std)
- Comparison and logical operations
- Array manipulation (reshape, transpose, take)
- Random number generation (uniform, normal)
- Save/load arrays to binary files
- OpenMP parallelization
- BLAS-optimized operations

## Disclaimers

- This is a project for learning.
- The API can change at any moment.
- It is not intended for production use.
- Feedback is welcomed

**Pending decisions**:

- It has not decided the error management approach. At this moment, only asserts are applied.

## For C Users

- [Design Considerations](guide/design.md)
- [Usage Guide](guide/usage.md)
- [Building](guide/building.md)
- [Advanced Topics](guide/advanced.md)

## For Zig Users

- [Zig Bindings](guide/zig-intro.md)
- [Zig Usage Guide](guide/zig-usage.md)
- [Zig Building](guide/zig-building.md)

## Examples

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

```zig
const ndarray = @import("ndarray");
const NDArray = ndarray.NDArray;

pub fn main() !void {
    // Create arrays
    const a = try NDArray.ones(&[_]usize{2, 3});
    defer a.deinit();
    const b = try NDArray.full(&[_]usize{2, 3}, 2.0);
    defer b.deinit();
    // Operations
    _ = a.add(b);  // a = a + b
    a.print("Result", 2);
}
```

## Installation

### Requirements

- C99-compatible compiler (GCC, Clang)
- OpenMP (for parallel operations)
- OpenBLAS (for optimized linear algebra)
- CUnit (optional, for tests)

### Build with CMake

```bash
mkdir build && cd build
cmake ..
cmake --build .
sudo cmake --build . --target install
```

### Build with Zig

```bash
zig build
```

See [Building](guide/building.md) for more options.


## License

BSD 3-Clause License. See [LICENSE](LICENSE) file for details.
