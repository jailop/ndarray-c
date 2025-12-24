const std = @import("std");
const ndarray = @import("ndarray");

pub fn main() !void {
    std.debug.print("=== Testing Extended Zig NDArray Functions ===\n\n", .{});

    // Test creation functions
    std.debug.print("--- Array Creation Functions ---\n", .{});
    
    const range = try ndarray.NDArray.arange(&[_]usize{ 1, 10 }, 0.0, 10.0, 1.0);
    defer range.deinit();
    range.print("Arange [0, 10, step=1]", 1);
    
    const linsp = try ndarray.NDArray.linspace(&[_]usize{ 1, 5 }, 0.0, 1.0, 5);
    defer linsp.deinit();
    linsp.print("Linspace [0, 1, 5 points]", 3);

    // Test take (slicing)
    std.debug.print("\n--- Take (Slicing) ---\n", .{});
    
    const mat = try ndarray.NDArray.arange(&[_]usize{ 4, 5 }, 1.0, 21.0, 1.0);
    defer mat.deinit();
    mat.print("Original [4, 5]", 1);
    
    const rows = try mat.take(0, 1, 3);  // Rows 1 and 2
    defer rows.deinit();
    rows.print("Rows [1:3]", 1);
    
    const cols = try mat.take(1, 1, 4);  // Columns 1, 2, 3
    defer cols.deinit();
    cols.print("Columns [1:4]", 1);

    // Test tensordot
    std.debug.print("\n--- Tensordot ---\n", .{});
    
    const t1 = try ndarray.NDArray.ones(&[_]usize{ 2, 3, 4 });
    defer t1.deinit();
    
    const t2 = try ndarray.NDArray.full(&[_]usize{ 4, 5 }, 2.0);
    defer t2.deinit();
    
    // Contract on axis 2 of t1 and axis 0 of t2
    const tensor_result = try t1.tensordot(t2, &[_]i32{2}, &[_]i32{0});
    defer tensor_result.deinit();
    
    std.debug.print("T1 shape: [2, 3, 4], T2 shape: [4, 5]\n", .{});
    std.debug.print("Tensordot on axes (2, 0) -> shape: ", .{});
    std.debug.print("[{}, {}, {}]\n", .{
        tensor_result.ptr.*.dims[0],
        tensor_result.ptr.*.dims[1],
        tensor_result.ptr.*.dims[2],
    });
    tensor_result.print("Result", 1);

    // Test stack
    std.debug.print("\n--- Stack Arrays ---\n", .{});
    
    const s1 = try ndarray.NDArray.ones(&[_]usize{ 2, 3 });
    defer s1.deinit();
    
    const s2 = try ndarray.NDArray.full(&[_]usize{ 2, 3 }, 2.0);
    defer s2.deinit();
    
    const s3 = try ndarray.NDArray.full(&[_]usize{ 2, 3 }, 3.0);
    defer s3.deinit();
    
    const stacked = try ndarray.stack(0, &[_]ndarray.NDArray{ s1, s2, s3 });
    defer stacked.deinit();
    
    std.debug.print("Stacked 3 arrays [2,3] along axis 0\n", .{});
    stacked.print("Result shape [3, 2, 3]", 1);

    // Test concat
    std.debug.print("\n--- Concatenate Arrays ---\n", .{});
    
    const c1 = try ndarray.NDArray.ones(&[_]usize{ 2, 3 });
    defer c1.deinit();
    
    const c2 = try ndarray.NDArray.full(&[_]usize{ 2, 3 }, 2.0);
    defer c2.deinit();
    
    // Concat along axis 0 (rows)
    const concat_rows = try ndarray.concat(0, &[_]ndarray.NDArray{ c1, c2 });
    defer concat_rows.deinit();
    concat_rows.print("Concat along axis 0 (rows)", 1);
    
    // Concat along axis 1 (columns)
    const concat_cols = try ndarray.concat(1, &[_]ndarray.NDArray{ c1, c2 });
    defer concat_cols.deinit();
    concat_cols.print("Concat along axis 1 (cols)", 1);

    std.debug.print("\n=== All extended functions work! ===\n", .{});
}
