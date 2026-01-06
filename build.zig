const std = @import("std");

// Common C source files for the ndarray library
const c_source_files = &[_][]const u8{
    "src/ndarray_core.c",
    "src/ndarray_creation.c",
    "src/ndarray_arithmetic.c",
    "src/ndarray_linalg.c",
    "src/ndarray_manipulation.c",
    "src/ndarray_aggregation.c",
    "src/ndarray_print.c",
    "src/ndarray_io.c",
    "src/ndarray_comparison.c",
};

const c_flags = &[_][]const u8{
    "-std=c99",
    "-O3",
    "-march=native",
    "-fopenmp",
};

/// Configure include paths for Homebrew libraries on macOS
fn addHomebrewPaths(step: anytype, b: *std.Build, homebrew_prefix: ?[]const u8) void {
    if (homebrew_prefix) |prefix| {
        step.addIncludePath(.{ .cwd_relative = b.fmt("{s}/opt/libomp/include", .{prefix}) });
        step.addIncludePath(.{ .cwd_relative = b.fmt("{s}/opt/openblas/include", .{prefix}) });
        step.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/opt/libomp/lib", .{prefix}) });
        step.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/opt/openblas/lib", .{prefix}) });
    }
}

/// Configure common library and include setup for compile steps
fn configureCompileStep(step: anytype, b: *std.Build, homebrew_prefix: ?[]const u8) void {
    step.addCSourceFiles(.{
        .files = c_source_files,
        .flags = c_flags,
    });
    step.addIncludePath(.{ .cwd_relative = "src" });
    step.linkLibC();
    addHomebrewPaths(step, b, homebrew_prefix);
    step.linkSystemLibrary("omp");
    step.linkSystemLibrary("openblas");
}

/// Create a module with Homebrew include paths
fn createModuleWithPaths(b: *std.Build, comptime opts: type, options: opts, homebrew_prefix: ?[]const u8) *std.Build.Module {
    const module = b.createModule(options);
    module.addIncludePath(.{ .cwd_relative = "src" });
    addHomebrewPaths(module, b, homebrew_prefix);
    return module;
}

/// Create an executable with ndarray module imported
fn createExecutable(b: *std.Build, name: []const u8, source_file: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, ndarray_module: *std.Build.Module, homebrew_prefix: ?[]const u8) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = name,
        .root_module = b.createModule(.{
            .root_source_file = b.path(source_file),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("ndarray", ndarray_module);
    configureCompileStep(exe, b, homebrew_prefix);
    b.installArtifact(exe);
    return exe;
}

/// Add a run step for an executable
fn addRunStep(b: *std.Build, exe: *std.Build.Step.Compile, step_name: []const u8, description: []const u8) void {
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    const run_step = b.step(step_name, description);
    run_step.dependOn(&run_cmd.step);
}

/// Build a C library (static or dynamic)
fn buildLibrary(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, linkage: std.builtin.LinkMode, version: ?std.SemanticVersion, homebrew_prefix: ?[]const u8) *std.Build.Step.Compile {
    const lib_module = createModuleWithPaths(b, std.Build.Module.CreateOptions, .{
        .target = target,
        .optimize = optimize,
    }, homebrew_prefix);

    const lib = b.addLibrary(.{
        .name = "ndarray",
        .root_module = lib_module,
        .linkage = linkage,
        .version = version,
    });
    lib.addCSourceFiles(.{ .files = c_source_files, .flags = c_flags });
    lib.linkLibC();
    lib.linkSystemLibrary("m");
    addHomebrewPaths(lib, b, homebrew_prefix);
    lib.linkSystemLibrary("omp");
    lib.linkSystemLibrary("openblas");
    lib.installHeader(b.path("src/ndarray.h"), "ndarray.h");
    b.installArtifact(lib);
    return lib;
}

/// Build a benchmark executable
fn buildBenchmark(b: *std.Build, name: []const u8, with_openmp: bool, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, homebrew_prefix: ?[]const u8) *std.Build.Step.Compile {
    const bench = b.addExecutable(.{
        .name = name,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });
    
    const bench_flags_base = &[_][]const u8{
        "-std=c99",
        "-O3",
        "-march=native",
    };
    
    const bench_flags_omp = &[_][]const u8{
        "-std=c99",
        "-O3",
        "-march=native",
        "-fopenmp",
    };
    
    const flags = if (with_openmp) bench_flags_omp else bench_flags_base;
    
    // Add benchmark source
    bench.addCSourceFile(.{
        .file = b.path("benchmarks/benchmark.c"),
        .flags = flags,
    });
    
    // Add library sources
    bench.addCSourceFiles(.{
        .files = c_source_files,
        .flags = flags,
    });
    
    bench.addIncludePath(.{ .cwd_relative = "src" });
    bench.linkLibC();
    addHomebrewPaths(bench, b, homebrew_prefix);
    if (with_openmp) {
        bench.linkSystemLibrary("omp");
    }
    bench.linkSystemLibrary("openblas");
    b.installArtifact(bench);
    return bench;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Detect Homebrew paths on macOS for cross-platform compatibility
    const is_macos = target.result.os.tag == .macos or @import("builtin").os.tag == .macos;
    const homebrew_prefix: ?[]const u8 = if (is_macos) "/opt/homebrew" else null;

    // Create the module for export
    _ = b.addModule("ndarray", .{
        .root_source_file = b.path("src/ndarray.zig"),
    });

    // Module for internal use in examples
    const ndarray_module = createModuleWithPaths(b, std.Build.Module.CreateOptions, .{
        .root_source_file = b.path("src/ndarray.zig"),
    }, homebrew_prefix);

    // Build executables
    const example = createExecutable(b, "example", "examples/basic.zig", target, optimize, ndarray_module, homebrew_prefix);
    addRunStep(b, example, "run", "Run the example");

    const extended = createExecutable(b, "extended", "examples/extended.zig", target, optimize, ndarray_module, homebrew_prefix);
    addRunStep(b, extended, "run-extended", "Run the extended example");

    // Test executable (Zig tests)
    const zig_tests = b.addTest(.{
        .name = "ndarray-zig-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ndarray.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    configureCompileStep(zig_tests, b, homebrew_prefix);

    const run_zig_tests = b.addRunArtifact(zig_tests);
    const zig_test_step = b.step("test", "Run Zig unit tests");
    zig_test_step.dependOn(&run_zig_tests.step);

    // C test suite (requires CUnit)
    const c_test = b.addExecutable(.{
        .name = "ndarray_test",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });
    c_test.addCSourceFile(.{
        .file = b.path("tests/test_ndarray.c"),
        .flags = c_flags,
    });
    configureCompileStep(c_test, b, homebrew_prefix);
    c_test.linkSystemLibrary("cunit");
    b.installArtifact(c_test);

    const run_c_test = b.addRunArtifact(c_test);
    const c_test_step = b.step("test-c", "Run C unit tests (requires CUnit)");
    c_test_step.dependOn(&run_c_test.step);

    // Run all tests
    const all_tests_step = b.step("test-all", "Run all tests (Zig and C)");
    all_tests_step.dependOn(&run_zig_tests.step);
    all_tests_step.dependOn(&run_c_test.step);

    // Benchmarks
    const bench_seq = buildBenchmark(b, "benchmark_seq", false, target, optimize, homebrew_prefix);
    const bench_omp = buildBenchmark(b, "benchmark_omp", true, target, optimize, homebrew_prefix);
    
    const bench_step = b.step("bench", "Build benchmark executables");
    bench_step.dependOn(&bench_seq.step);
    bench_step.dependOn(&bench_omp.step);
    
    const run_bench_seq = b.addRunArtifact(bench_seq);
    const run_bench_seq_step = b.step("run-bench-seq", "Run sequential benchmark");
    run_bench_seq_step.dependOn(&run_bench_seq.step);
    
    const run_bench_omp = b.addRunArtifact(bench_omp);
    const run_bench_omp_step = b.step("run-bench-omp", "Run OpenMP benchmark");
    run_bench_omp_step.dependOn(&run_bench_omp.step);

    // Build C libraries
    _ = buildLibrary(b, target, optimize, .static, null, homebrew_prefix);
    _ = buildLibrary(b, target, optimize, .dynamic, .{ .major = 1, .minor = 0, .patch = 0 }, homebrew_prefix);

    // Build steps for libraries
    const lib_step = b.step("lib", "Build both static and shared libraries");
    lib_step.dependOn(b.getInstallStep());

    const static_step = b.step("static", "Build static library");
    static_step.dependOn(b.getInstallStep());

    const shared_step = b.step("shared", "Build shared library");
    shared_step.dependOn(b.getInstallStep());
}
