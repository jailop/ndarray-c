const std = @import("std");
const ndarray = @import("ndarray");

pub fn main() !void {
    std.debug.print("=== Zig NDArray Bindings Example ===\n\n", .{});

    // Create arrays from data
    std.debug.print("Creating matrices from data...\n", .{});
    const m1_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const m2_data = [_]f64{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };

    const m1 = try ndarray.NDArray.fromData(&[_]usize{ 2, 3 }, &m1_data);
    defer m1.deinit();

    const m2 = try ndarray.NDArray.fromData(&[_]usize{ 3, 2 }, &m2_data);
    defer m2.deinit();

    m1.print("Matrix M1 (2x3)", 1);
    m2.print("Matrix M2 (3x2)", 1);

    // Matrix multiplication
    std.debug.print("\nPerforming matrix multiplication...\n", .{});
    const result = try m1.matmul(m2);
    defer result.deinit();

    result.print("M1 @ M2", 1);

    // Save to file
    std.debug.print("\nSaving result to file...\n", .{});
    try result.save("result_zig.bin");
    std.debug.print("Saved to result_zig.bin\n", .{});

    // Load from file
    std.debug.print("\nLoading from file...\n", .{});
    const loaded = try ndarray.NDArray.load("result_zig.bin");
    defer loaded.deinit();

    loaded.print("Loaded Result", 1);

    // Create arrays with different initializations
    std.debug.print("\n--- Array Creation Methods ---\n", .{});

    const zeros = try ndarray.NDArray.zeros(&[_]usize{ 2, 3 });
    defer zeros.deinit();
    zeros.print("Zeros", 1);

    const ones = try ndarray.NDArray.ones(&[_]usize{ 2, 3 });
    defer ones.deinit();
    ones.print("Ones", 1);

    const filled = try ndarray.NDArray.full(&[_]usize{ 2, 3 }, 7.5);
    defer filled.deinit();
    filled.print("Filled with 7.5", 1);

    // Element-wise operations
    std.debug.print("\n--- Element-wise Operations ---\n", .{});

    const a = try ndarray.NDArray.ones(&[_]usize{ 2, 3 });
    defer a.deinit();

    const b = try ndarray.NDArray.full(&[_]usize{ 2, 3 }, 2.0);
    defer b.deinit();

    a.print("Array A (ones)", 1);
    b.print("Array B (twos)", 1);

    // Modify a in place
    a.add(b);
    a.print("A after adding B", 1);

    a.mulScalar(2.0);
    a.print("A after multiplying by 2", 1);

    // Transpose
    std.debug.print("\n--- Transpose ---\n", .{});
    const mat = try ndarray.NDArray.fromData(&[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 });
    defer mat.deinit();

    mat.print("Original", 1);

    const transposed = try mat.transpose();
    defer transposed.deinit();

    transposed.print("Transposed", 1);

    // Random arrays
    std.debug.print("\n--- Random Arrays ---\n", .{});

    const rand_unif = try ndarray.NDArray.randomUniform(&[_]usize{ 2, 3 }, 0.0, 1.0);
    defer rand_unif.deinit();
    rand_unif.print("Random Uniform [0, 1)", 4);

    const rand_norm = try ndarray.NDArray.randomNormal(&[_]usize{ 2, 3 }, 0.0, 1.0);
    defer rand_norm.deinit();
    rand_norm.print("Random Normal (mean=0, std=1)", 4);

    // Aggregation
    std.debug.print("\n--- Aggregation ---\n", .{});
    const arr = try ndarray.NDArray.fromData(&[_]usize{ 3, 4 }, &[_]f64{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
    });
    defer arr.deinit();

    arr.print("Array", 1);

    const sum_all = try ndarray.aggregate(arr, -1, .sum);
    defer sum_all.deinit();
    sum_all.print("Sum (all)", 1);

    const mean_axis0 = try ndarray.aggregate(arr, 0, .mean);
    defer mean_axis0.deinit();
    mean_axis0.print("Mean (axis 0)", 2);

    std.debug.print("\n=== All examples completed successfully ===\n", .{});
}
