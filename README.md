# ndarray for C and Zig

A numpy-like ndarray library for C, with Zig bindings.

- Multi-dimensional arrays (ndim >= 2)
- OpenMP parallelization
- BLAS-optimized operations
 
[Design Considerations](guide/design.md)  
[API Reference](https://jailop.github.io/ndarray-c/)  
[Nim Bindings](https://github.com/jailop/ndarray-nim)

## Disclaimers

- This is a project for learning.
- The API can change at any moment.
- It is not intended for production use.
- Feedback is welcomed

**Pending decisions**:

- It has not being decided the error management approach. At this
  moment, only asserts are applied.
- The intention is that the name of the function indicates if the result
  is a new allocated array or it is only an scalar. Functions that
  doesn't have an indication about its return value perform inplace
  operations over the first argument. This still needs to be refined.

## For C Users

- [Usage Guide](guide/usage.md)
- [Building](guide/building.md)
- [Advanced Topics](guide/advanced.md)

## For Zig Users

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
    ndarray_print(arr, "My Array", 2);  // 2 is the output precision
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
    const a = try NDArray.ones(&.{2, 3});
    defer a.deinit();
    const b = try NDArray.full(&.{2, 3}, 2.0);
    defer b.deinit();
    // Operations
    _ = a.add(b);  // a = a + b
    a.print("Result", 2);  // 2 is the output precision
}
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE) file for details.
