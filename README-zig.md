# ndarray for Zig

Zig bindings for the ndarray-c library.

- [Main Project](README.md)

## Usage

**Array Creation:**

```zig
const ndarray = @import("ndarray");

// Create arrays with different initializations
const zeros = try ndarray.NDArray.zeros(&[_]usize{3, 4});
defer zeros.deinit();

const ones = try ndarray.NDArray.ones(&[_]usize{3, 4});
defer ones.deinit();

const filled = try ndarray.NDArray.full(&[_]usize{3, 4}, 42.0);
defer filled.deinit();

// Create from existing data
const data = [_]f64{ 1, 2, 3, 4, 5, 6 };
const arr = try ndarray.NDArray.fromData(&[_]usize{2, 3}, &data);
defer arr.deinit();
```

**Array Creation with Ranges:**

```zig
// Create array with evenly spaced values
const range = try ndarray.NDArray.arange(&[_]usize{1, 10}, 0, 10, 1);
defer range.deinit();

// Create array with linearly spaced values
const linear = try ndarray.NDArray.linspace(&[_]usize{1, 100}, 0, 1, 100);
defer linear.deinit();
```

**Random Arrays:**

```zig
// Uniform distribution [0, 1)
const rand_unif = try ndarray.NDArray.randomUniform(&[_]usize{3, 4}, 0.0, 1.0);
defer rand_unif.deinit();

// Normal distribution (mean=0, std=1)
const rand_norm = try ndarray.NDArray.randomNormal(&[_]usize{3, 4}, 0.0, 1.0);
defer rand_norm.deinit();
```

**Getting and Setting Values:**

```zig
const arr = try ndarray.NDArray.zeros(&[_]usize{3, 4});
defer arr.deinit();

// Set value at position (1, 2)
arr.set(&[_]usize{1, 2}, 42.0);

// Get value at position (1, 2)
const val = arr.get(&[_]usize{1, 2});

// Print the array
arr.print("My Array", 4); // precision = 4 decimal places
```

**Element-wise Operations:**

```zig
const a = try ndarray.NDArray.ones(&[_]usize{3, 3});
defer a.deinit();

const b = try ndarray.NDArray.full(&[_]usize{3, 3}, 2.0);
defer b.deinit();

// Element-wise addition (modifies a in place)
a.add(b);

// Element-wise multiplication (modifies a in place)
a.mul(b);

// Scalar operations
a.addScalar(5.0);
a.mulScalar(2.0);
```

**Matrix Operations:**

```zig
const a = try ndarray.NDArray.ones(&[_]usize{3, 4});
defer a.deinit();

const b = try ndarray.NDArray.ones(&[_]usize{4, 2});
defer b.deinit();

// Matrix multiplication (creates new array)
const c = try a.matmul(b);
defer c.deinit();

c.print("Result", 2);
```

**Tensor Operations:**

```zig
// Create 3D tensors
const t1 = try ndarray.NDArray.ones(&[_]usize{2, 3, 4});
defer t1.deinit();

const t2 = try ndarray.NDArray.full(&[_]usize{4, 5}, 2.0);
defer t2.deinit();

// Tensor contraction (dot product along specified axes)
const result = try t1.tensordot(t2, &[_]i32{2}, &[_]i32{0});
defer result.deinit();
```

**Array Manipulation:**

```zig
const mat = try ndarray.NDArray.arange(&[_]usize{3, 4}, 0, 12, 1);
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
const a = try ndarray.NDArray.ones(&[_]usize{2, 3});
defer a.deinit();

const b = try ndarray.NDArray.full(&[_]usize{2, 3}, 2.0);
defer b.deinit();

const c = try ndarray.NDArray.full(&[_]usize{2, 3}, 3.0);
defer c.deinit();

// Stack arrays along axis 0 (creates new dimension)
const arrays = [_]ndarray.NDArray{a, b, c};
const stacked = try ndarray.stack(0, &arrays);
defer stacked.deinit();

// Concatenate arrays along axis 1 (along columns)
const concatenated = try ndarray.concat(1, &arrays);
defer concatenated.deinit();
```

**Aggregations:**

```zig
const arr = try ndarray.NDArray.arange(&[_]usize{3, 4}, 1, 13, 1);
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
const arr = try ndarray.NDArray.randomUniform(&[_]usize{3, 4}, 0.0, 1.0);
defer arr.deinit();

try arr.save("mydata.bin");

// Load array from file
const loaded = try ndarray.NDArray.load("mydata.bin");
defer loaded.deinit();

loaded.print("Loaded Array", 4);
```

**Array Metadata:**

```zig
const arr = try ndarray.NDArray.ones(&[_]usize{3, 4, 5});
defer arr.deinit();

const dims = arr.ndim();        // Returns: 3
const total = arr.size();       // Returns: 60
const shape = arr.shape();      // Returns: &[_]usize{3, 4, 5}
```

## Building

Requirements:

- Zig 0.15.0 or later
- OpenBLAS: For optimized BLAS operations (required)
- OpenMP: For parallel operations (required)

Build and run examples:

```bash
zig build run              # Run basic example
zig build run-extended     # Run advanced example
zig build test             # Run tests
```

## Using as a Package

In your project's `build.zig.zon`:

```zig
.{
    .name = "myproject",
    .version = "0.1.0",
    .dependencies = .{
        .ndarray = .{
            .path = "../path/to/ndarray-c",
        },
    },
}
```

In your project's `build.zig`:

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Get the ndarray dependency
    const ndarray_dep = b.dependency("ndarray", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "myapp",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Import the ndarray module
    exe.root_module.addImport("ndarray", ndarray_dep.module("ndarray"));

    // Link required libraries
    exe.linkLibC();
    exe.linkSystemLibrary("omp");
    exe.linkSystemLibrary("openblas");

    b.installArtifact(exe);
}
```

In your `build.zig.zon`:

```zig
.dependencies = .{
    .ndarray = .{
        .url = "https://github.com/jailop/ndarray-c/archive/refs/tags/v1.0.0.tar.gz",
        .hash = "1220...", // Zig will provide the correct hash
    },
}
```

Copy `src/ndarray.zig` to your project and compile the C sources directly:

```zig
// In your build.zig
const ndarray_module = b.createModule(.{
    .root_source_file = b.path("src/ndarray.zig"),
});

exe.root_module.addImport("ndarray", ndarray_module);

exe.addCSourceFiles(.{
    .files = &.{
        "path/to/ndarray-c/src/ndarray_core.c",
        "path/to/ndarray-c/src/ndarray_creation.c",
        "path/to/ndarray-c/src/ndarray_arithmetic.c",
        "path/to/ndarray-c/src/ndarray_linalg.c",
        "path/to/ndarray-c/src/ndarray_manipulation.c",
        "path/to/ndarray-c/src/ndarray_aggregation.c",
        "path/to/ndarray-c/src/ndarray_print.c",
        "path/to/ndarray-c/src/ndarray_io.c",
    },
    .flags = &.{"-std=c99", "-O3", "-fopenmp"},
});

exe.linkLibC();
exe.linkSystemLibrary("omp");
exe.linkSystemLibrary("openblas");
```

## Project Structure

```
ndarray-c/
├── src/                    # C and Zig source files
│   ├── ndarray.h          # C header
│   ├── ndarray_internal.h # Internal C header
│   ├── ndarray_*.c        # C implementation files
│   └── ndarray.zig        # Zig bindings
├── tests/                  # Test files
│   └── test_ndarray.c     # C tests
├── benchmarks/             # Benchmark files
│   └── benchmark.c        # Performance benchmarks
├── examples/               # Example files
│   ├── example.c          # C example
│   ├── basic.zig          # Zig basic example
│   └── extended.zig       # Zig extended example
├── docs/                   # Generated documentation
├── Makefile               # C build system
├── build.zig              # Zig build system
├── README.md              # Main documentation (C)
└── README-zig.md          # This file (Zig documentation)
```
