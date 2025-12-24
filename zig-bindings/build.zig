const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Path to C library sources
    const ndarray_src_path = "../src";

    // Create the module for export
    _ = b.addModule("ndarray", .{
        .root_source_file = b.path("src/ndarray.zig"),
    });

    // Module for internal use in examples
    const ndarray_module = b.createModule(.{
        .root_source_file = b.path("src/ndarray.zig"),
    });

    // Example executable
    const example_module = b.createModule(.{
        .root_source_file = b.path("examples/basic.zig"),
        .target = target,
        .optimize = optimize,
    });
    example_module.addImport("ndarray", ndarray_module);

    const example = b.addExecutable(.{
        .name = "example",
        .root_module = example_module,
    });

    example.addCSourceFiles(.{
        .files = &.{
            ndarray_src_path ++ "/ndarray_core.c",
            ndarray_src_path ++ "/ndarray_creation.c",
            ndarray_src_path ++ "/ndarray_arithmetic.c",
            ndarray_src_path ++ "/ndarray_linalg.c",
            ndarray_src_path ++ "/ndarray_manipulation.c",
            ndarray_src_path ++ "/ndarray_aggregation.c",
            ndarray_src_path ++ "/ndarray_print.c",
            ndarray_src_path ++ "/ndarray_io.c",
        },
        .flags = &.{
            "-std=c99",
            "-O3",
            "-march=native",
            "-fopenmp",
        },
    });

    example.addIncludePath(.{ .cwd_relative = ndarray_src_path });
    example.linkLibC();
    example.linkSystemLibrary("omp");
    example.linkSystemLibrary("openblas");

    b.installArtifact(example);

    const run_cmd = b.addRunArtifact(example);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the example");
    run_step.dependOn(&run_cmd.step);

    // Extended example executable
    const extended_module = b.createModule(.{
        .root_source_file = b.path("examples/extended.zig"),
        .target = target,
        .optimize = optimize,
    });
    extended_module.addImport("ndarray", ndarray_module);

    const extended = b.addExecutable(.{
        .name = "extended",
        .root_module = extended_module,
    });

    extended.addCSourceFiles(.{
        .files = &.{
            ndarray_src_path ++ "/ndarray_core.c",
            ndarray_src_path ++ "/ndarray_creation.c",
            ndarray_src_path ++ "/ndarray_arithmetic.c",
            ndarray_src_path ++ "/ndarray_linalg.c",
            ndarray_src_path ++ "/ndarray_manipulation.c",
            ndarray_src_path ++ "/ndarray_aggregation.c",
            ndarray_src_path ++ "/ndarray_print.c",
            ndarray_src_path ++ "/ndarray_io.c",
        },
        .flags = &.{
            "-std=c99",
            "-O3",
            "-march=native",
            "-fopenmp",
        },
    });

    extended.addIncludePath(.{ .cwd_relative = ndarray_src_path });
    extended.linkLibC();
    extended.linkSystemLibrary("omp");
    extended.linkSystemLibrary("openblas");

    b.installArtifact(extended);

    const run_extended_cmd = b.addRunArtifact(extended);
    run_extended_cmd.step.dependOn(b.getInstallStep());

    const run_extended_step = b.step("run-extended", "Run the extended example");
    run_extended_step.dependOn(&run_extended_cmd.step);

    // Create root module for tests
    const test_module = b.createModule(.{
        .root_source_file = b.path("src/ndarray.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Test executable
    const tests = b.addTest(.{
        .name = "ndarray-tests",
        .root_module = test_module,
    });

    tests.addCSourceFiles(.{
        .files = &.{
            ndarray_src_path ++ "/ndarray_core.c",
            ndarray_src_path ++ "/ndarray_creation.c",
            ndarray_src_path ++ "/ndarray_arithmetic.c",
            ndarray_src_path ++ "/ndarray_linalg.c",
            ndarray_src_path ++ "/ndarray_manipulation.c",
            ndarray_src_path ++ "/ndarray_aggregation.c",
            ndarray_src_path ++ "/ndarray_print.c",
            ndarray_src_path ++ "/ndarray_io.c",
        },
        .flags = &.{
            "-std=c99",
            "-O3",
            "-march=native",
            "-fopenmp",
        },
    });

    tests.addIncludePath(.{ .cwd_relative = ndarray_src_path });
    tests.linkLibC();
    tests.linkSystemLibrary("omp");
    tests.linkSystemLibrary("openblas");

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
