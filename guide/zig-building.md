## Building

Requirements:

- Zig 0.15.0 or later
- OpenBLAS: For optimized BLAS operations (required)
- OpenMP: For parallel operations (required)

### Installing Dependencies

**macOS (Homebrew):**
```bash
brew install zig libomp openblas
```

**Ubuntu/Debian:**
```bash
sudo apt-get install zig libomp-dev libopenblas-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install zig libomp-devel openblas-devel
```

### Build Commands

**Examples:**
```bash
zig build                  # Build examples
zig build run              # Run basic example
zig build run-extended     # Run extended example
```

**Libraries:**
```bash
zig build lib              # Build both static and shared
zig build static           # Build static library only
zig build shared           # Build shared library only
```

Output: `zig-out/lib/libndarray.{a,dylib,so}` and `zig-out/include/ndarray.h`

**Tests:**
```bash
zig build test             # Run Zig unit tests
zig build test-c           # Run C unit tests (requires CUnit)
zig build test-all         # Run all tests
```

**Benchmarks:**
```bash
zig build bench            # Build benchmark executables

# Note: Due to Zig cache limitations with mixed compilation flags,
# use the automated script which supports all build systems:
cd benchmarks && ./run_benchmark.sh
```

## Using as a Package

Step 1: Add to `build.zig.zon`

```zig
.{
    .name = .myproject,
    .version = "0.1.0",
    .dependencies = .{
        .ndarray = .{
            .url = "https://github.com/jailop/ndarray-c/archive/refs/tags/v0.2.2.tar.gz",
            .hash = "1220000000000000000000000000000000000000000000000000000000000000",
        },
    },
}
```

Step 2: Run `zig build`

Zig will download the package and tell you the correct hash. Update your
`build.zig.zon` with the correct hash.

Step 3: Update your `build.zig`

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "myapp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Get ndarray dependency from package manager
    const ndarray_dep = b.dependency("ndarray", .{
        .target = target,
        .optimize = optimize,
    });

    // Get the Zig module and compiled library artifact
    const ndarray_module = ndarray_dep.module("ndarray");
    const ndarray_lib = ndarray_dep.artifact("ndarray");

    // Import the module
    exe.root_module.addImport("ndarray", ndarray_module);

    // Link the library (automatically includes OpenMP and OpenBLAS)
    exe.linkLibrary(ndarray_lib);

    b.installArtifact(exe);
}
```

Step 4: Use in your code

```zig
const ndarray = @import("ndarray");
const NDArray = ndarray.NDArray;

pub fn main() !void {
    const arr = try NDArray.randomNormal(&[_]usize{100, 100}, 0.0, 1.0);
    defer arr.deinit();
    
    _ = arr.mulScalar(2.0).addScalar(10.0);
    arr.print("My Array", 4);
}
```

System Requirements:

Users of your application need OpenMP and OpenBLAS installed on their system:

**macOS (Homebrew):**
```bash
brew install libomp openblas
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libomp-dev libopenblas-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install libomp-devel openblas-devel
```

**Arch Linux:**
```bash
sudo pacman -S openmp openblas
```

---
[← Zig Usage](zig-usage.md) | [Back to Main](../README.md)
