## Usage

**Array Creation:**

```zig
const ndarray = @import("ndarray");
const NDArray = ndarray.NDArray;

// Create arrays with different initializations
const zeros = try NDArray.zeros(&[_]usize{3, 4});
defer zeros.deinit();

const ones = try NDArray.ones(&[_]usize{3, 4});
defer ones.deinit();

const filled = try NDArray.full(&[_]usize{3, 4}, 42.0);
defer filled.deinit();

// Create from existing data
const data = [_]f64{ 1, 2, 3, 4, 5, 6 };
const arr = try NDArray.fromData(&[_]usize{2, 3}, &data);
defer arr.deinit();
```

**Array Creation with Ranges:**

```zig
// Create array with evenly spaced values
const range = try NDArray.arange(&[_]usize{1, 10}, 0, 10, 1);
defer range.deinit();

// Create array with linearly spaced values
const linear = try NDArray.linspace(&[_]usize{1, 100}, 0, 1, 100);
defer linear.deinit();
```

**Random Arrays:**

```zig
// Uniform distribution [0, 1)
const rand_unif = try NDArray.randomUniform(&[_]usize{3, 4}, 0.0, 1.0);
defer rand_unif.deinit();

// Normal distribution (mean=0, std=1)
const rand_norm = try NDArray.randomNormal(&[_]usize{3, 4}, 0.0, 1.0);
defer rand_norm.deinit();
```

**Getting and Setting Values:**

```zig
const arr = try NDArray.zeros(&[_]usize{3, 4});
defer arr.deinit();

// Set value at position (1, 2)
arr.set(&[_]usize{1, 2}, 42.0);

// Get value at position (1, 2)
const val = arr.get(&[_]usize{1, 2});

// Set an entire row from an array of values
const row_data = [_]f64{1.0, 2.0, 3.0, 4.0};
arr.setSlice(0, 1, &row_data);  // Set row 1

// Set an entire column from an array of values
const col_data = [_]f64{10.0, 20.0, 30.0};
arr.setSlice(1, 2, &col_data);  // Set column 2

// Fill an entire row with a scalar value
arr.fillSlice(0, 0, 0.0);  // Fill row 0 with zeros

// Fill an entire column with a scalar value
arr.fillSlice(1, 3, 99.0);  // Fill column 3 with 99.0

// Works for any dimension - set a hyperplane in 3D array
const arr3d = try NDArray.zeros(&[_]usize{2, 3, 4});
defer arr3d.deinit();
var plane_data: [8]f64 = undefined;  // 2*4 = 8 values
for (0..8) |i| plane_data[i] = @as(f64, @floatFromInt(i)) * 5.0;
arr3d.setSlice(1, 1, &plane_data);  // Set middle plane

// Print the array
arr.print("My Array", 4); // precision = 4 decimal places
```

**Method Chaining (Fluent Interface):**

Many operations support method chaining for cleaner code:

```zig
const arr = try NDArray.zeros(&[_]usize{3, 4});
defer arr.deinit();

// Chain multiple operations
_ = arr.addScalar(5.0)
    .mulScalar(2.0)
    .clipMin(0.0);

// More complex chains
const a = try NDArray.randomNormal(&[_]usize{100, 100}, 0.0, 1.0);
defer a.deinit();

_ = a.mapFn(sqrt)           // Apply sqrt to each element
    .mul(b)                  // Multiply by b
    .addScalar(10.0)        // Add 10
    .clipMin(0.0);          // Ensure non-negative

// Variance calculation with chaining
const variance = try NDArray.zeros(&[_]usize{3, 4});
defer variance.deinit();

_ = variance.scaleShift(1.0 - kappa * dt, kappa * theta * dt)
    .add(stochastic_term)
    .clipMin(0.0);
```

**Using Math Functions with mapFn:**

The `mapFn()` method applies a function to each element. You can use
Zig's built-in math functions by wrapping them with C calling
convention:

```zig
const std = @import("std");
const ndarray = @import("ndarray");
const NDArray = ndarray.NDArray;

// Wrapper functions for Zig math with C calling convention
fn sqrt(x: f64) callconv(.c) f64 {
    return @sqrt(x);  // Zig built-in sqrt
}

fn exp(x: f64) callconv(.c) f64 {
    return @exp(x);   // Zig built-in exp
}

fn log(x: f64) callconv(.c) f64 {
    return @log(x);   // Zig built-in log
}

fn sin(x: f64) callconv(.c) f64 {
    return @sin(x);   // Zig built-in sin
}

// Custom function
fn square(x: f64) callconv(.c) f64 {
    return x * x;
}

pub fn main() !void {
    const arr = try NDArray.arange(&[_]usize{1, 10}, 1.0, 11.0, 1.0);
    defer arr.deinit();
    // Apply sqrt to all elements
    _ = arr.mapFn(sqrt);
    // Apply exponential
    _ = arr.mapFn(exp);
    // Apply custom function
    _ = arr.mapFn(square);
    arr.print("Result", 4);
}
```

Alternative - Using std.math:

You can also use `std.math` functions:

```zig
fn sqrt(x: f64) callconv(.c) f64 {
    return std.math.sqrt(x);  // Runtime function
}

fn exp(x: f64) callconv(.c) f64 {
    return std.math.exp(x);   // Runtime function
}
```

Both `@sqrt` and `std.math.sqrt` produce identical results, but `@sqrt`
may be optimized better by the compiler.

**Element-wise Operations:**

```zig
const a = try NDArray.ones(&[_]usize{3, 3});
defer a.deinit();

const b = try NDArray.full(&[_]usize{3, 3}, 2.0);
defer b.deinit();

// Element-wise addition (modifies a in place, returns a for chaining)
_ = a.add(b);

// Element-wise multiplication (modifies a in place, returns a for chaining)
_ = a.mul(b);

// Scalar operations
_ = a.addScalar(5.0);
_ = a.mulScalar(2.0);

// Linear combination: a = alpha*a + beta*b
const c = try NDArray.full(&[_]usize{3, 3}, 5.0);
defer c.deinit();
const d = try NDArray.full(&[_]usize{3, 3}, 3.0);
defer d.deinit();
_ = c.axpby(2.0, d, 3.0);  // c = 2*c + 3*d = 2*5 + 3*3 = 19

// Fused operations
_ = c.scaleShift(0.5, 10.0);     // c = 0.5*c + 10
_ = c.mulScaled(d, 2.0);          // c = c * d * 2

// Apply custom function to each element
extern fn sqrt(f64) f64;
_ = a.mapFn(sqrt);  // Apply sqrt to all elements
```

**Conditional Operations:**

```zig
const values = try NDArray.arange(&[_]usize{3, 3}, -4.0, 5.0, 1.0);
defer values.deinit();

// Clipping operations (support chaining)
_ = values.clipMin(0.0)           // Non-negativity constraint
    .clipMax(100.0)                // Cap maximum value
    .clip(0.0, 1.0);               // Normalize to [0, 1]

// Absolute value and sign
const errors = try NDArray.arange(&[_]usize{2, 3}, -2.0, 4.0, 1.0);
defer errors.deinit();
_ = errors.abs();                  // Distance calculations
_ = errors.sign();                 // Direction indicators (-1, 0, 1)
```

**Comparison and Logical Operations:**

```zig
const a = try NDArray.arange(&[_]usize{2, 3}, 0.0, 6.0, 1.0);
defer a.deinit();

const b = try NDArray.full(&[_]usize{2, 3}, 3.0);
defer b.deinit();

// Element-wise comparisons (returns 1.0/0.0)
const eq = try a.equal(b);           // a == b
defer eq.deinit();

const lt = try a.less(b);            // a < b
defer lt.deinit();

const gt = try a.greater(b);         // a > b
defer gt.deinit();

// Scalar comparisons
const positive = try a.greaterScalar(0.0);  // Find positive values
defer positive.deinit();

const zeros = try a.equalScalar(0.0);       // Find zeros
defer zeros.deinit();

// Logical operations
const mask1 = try a.greaterScalar(2.0);
defer mask1.deinit();

const mask2 = try a.lessScalar(5.0);
defer mask2.deinit();

const combined = try mask1.logicalAnd(mask2);  // 2 < a < 5
defer combined.deinit();

// NumPy-style where
const x = try NDArray.full(&[_]usize{2, 3}, 10.0);
defer x.deinit();

const y = try NDArray.full(&[_]usize{2, 3}, -10.0);
defer y.deinit();

const result = try NDArray.where(positive, x, y);  // positive ? x : y
defer result.deinit();
```

**Slice Access:**

```zig
const matrix = try NDArray.arange(&[_]usize{4, 5}, 0.0, 20.0, 1.0);
defer matrix.deinit();

// Get pointer to slice (advanced users)
const row2_ptr = matrix.getSlicePtr(0, 2);  // Pointer to row 2

// Copy slices between arrays
const dest = try NDArray.zeros(&[_]usize{4, 5});
defer dest.deinit();
NDArray.copySlice(matrix, 0, 1, dest, 0, 3);  // Copy row 1 to row 3

// Get slice size
const row_size = matrix.getSliceSize(0);  // Elements per row
const col_size = matrix.getSliceSize(1);  // Elements per column
```

**Matrix Operations:**

```zig
const a = try NDArray.ones(&[_]usize{3, 4});
defer a.deinit();

const b = try NDArray.ones(&[_]usize{4, 2});
defer b.deinit();

// Matrix multiplication (creates new array)
const c = try a.matmul(b);
defer c.deinit();

c.print("Result", 2);
```

**Tensor Operations:**

```zig
// Create 3D tensors
const t1 = try NDArray.ones(&[_]usize{2, 3, 4});
defer t1.deinit();

const t2 = try NDArray.full(&[_]usize{4, 5}, 2.0);
defer t2.deinit();

// Tensor contraction (dot product along specified axes)
const result = try t1.tensordot(t2, &[_]i32{2}, &[_]i32{0});
defer result.deinit();
```

**Array Manipulation:**

```zig
const mat = try NDArray.arange(&[_]usize{3, 4}, 0, 12, 1);
defer mat.deinit();

// Transpose
const transposed = try mat.transpose();
defer transposed.deinit();

// Slicing (take rows 1-2 along axis 0)
const sliced = try mat.take(0, 1, 3);
defer sliced.deinit();

// Copy array
const copied = try mat.copy();
defer copied.deinit();
```

**Stacking and Concatenating:**

```zig
const a = try NDArray.ones(&[_]usize{2, 3});
defer a.deinit();

const b = try NDArray.full(&[_]usize{2, 3}, 2.0);
defer b.deinit();

const c = try NDArray.full(&[_]usize{2, 3}, 3.0);
defer c.deinit();

// Stack arrays along axis 0 (creates new dimension)
const arrays = [_]NDArray{a, b, c};
const stacked = try ndarray.stack(0, &arrays);
defer stacked.deinit();

// Concatenate arrays along axis 1 (along columns)
const concatenated = try ndarray.concat(1, &arrays);
defer concatenated.deinit();

// Reshape array in-place
const arr = try NDArray.arange(&[_]usize{2, 6}, 0, 12, 1);
defer arr.deinit();
try arr.reshape(&[_]isize{3, 4});  // Now [3,4]
try arr.reshape(&[_]isize{2, 2, 3});  // Now [2,2,3]
try arr.reshape(&[_]isize{4, -1});  // Now [4,3] (inferred dimension)
```

**Aggregations:**

```zig
const arr = try NDArray.arange(&[_]usize{3, 4}, 1, 13, 1);
defer arr.deinit();

// Sum all elements
const total = try ndarray.aggregate(arr, -1, .sum);
defer total.deinit();

// Mean along axis 0
const col_means = try ndarray.aggregate(arr, 0, .mean);
defer col_means.deinit();

// Max along axis 1
const row_maxs = try ndarray.aggregate(arr, 1, .max);
defer row_maxs.deinit();

// Available aggregation types: .sum, .mean, .max, .min, .std
```

**Saving and Loading Arrays:**

```zig
// Save array to file
const arr = try NDArray.randomUniform(&[_]usize{3, 4}, 0.0, 1.0);
defer arr.deinit();

try arr.save("mydata.bin");

// Load array from file
const loaded = try NDArray.load("mydata.bin");
defer loaded.deinit();

loaded.print("Loaded Array", 4);
```

**Array Metadata:**

```zig
const arr = try NDArray.ones(&[_]usize{3, 4, 5});
defer arr.deinit();

const dims = arr.ndim();        // Returns: 3
const total = arr.size();       // Returns: 60
const shape = arr.shape();      // Returns: &[_]usize{3, 4, 5}
```


---
[← Intro](zig-intro.md) | [Back to Main](../README.md) | [Zig Building →](zig-building.md)
