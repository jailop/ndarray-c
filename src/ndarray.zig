const std = @import("std");

// C library bindings
pub const c = @cImport({
    @cInclude("stddef.h");
    @cInclude("ndarray.h");
});

/// NDArray wrapper for Zig
pub const NDArray = struct {
    ptr: c.NDArray,

    /// Create a new ndarray with specified dimensions
    pub fn init(dims: []const usize) !NDArray {
        var c_dims = try std.heap.c_allocator.alloc(usize, dims.len + 1);
        defer std.heap.c_allocator.free(c_dims);
        
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0; // Sentinel
        
        const ptr = c.ndarray_new(c_dims.ptr);
        if (ptr == null) return error.AllocationFailed;
        
        return NDArray{ .ptr = ptr };
    }

    /// Free the ndarray
    pub fn deinit(self: NDArray) void {
        c.ndarray_free(self.ptr);
    }

    /// Create array filled with zeros
    pub fn zeros(dims: []const usize) !NDArray {
        var c_dims = try std.heap.c_allocator.alloc(usize, dims.len + 1);
        defer std.heap.c_allocator.free(c_dims);
        
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        
        const ptr = c.ndarray_new_zeros(c_dims.ptr);
        if (ptr == null) return error.AllocationFailed;
        
        return NDArray{ .ptr = ptr };
    }

    /// Create array filled with ones
    pub fn ones(dims: []const usize) !NDArray {
        var c_dims = try std.heap.c_allocator.alloc(usize, dims.len + 1);
        defer std.heap.c_allocator.free(c_dims);
        
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        
        const ptr = c.ndarray_new_ones(c_dims.ptr);
        if (ptr == null) return error.AllocationFailed;
        
        return NDArray{ .ptr = ptr };
    }

    /// Create array filled with a specific value
    pub fn full(dims: []const usize, value: f64) !NDArray {
        var c_dims = try std.heap.c_allocator.alloc(usize, dims.len + 1);
        defer std.heap.c_allocator.free(c_dims);
        
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        
        const ptr = c.ndarray_new_full(c_dims.ptr, value);
        if (ptr == null) return error.AllocationFailed;
        
        return NDArray{ .ptr = ptr };
    }

    /// Create array from existing data
    pub fn fromData(dims: []const usize, data: []const f64) !NDArray {
        var c_dims = try std.heap.c_allocator.alloc(usize, dims.len + 1);
        defer std.heap.c_allocator.free(c_dims);
        
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        
        const ptr = c.ndarray_new_from_data(c_dims.ptr, @constCast(data.ptr));
        if (ptr == null) return error.AllocationFailed;
        
        return NDArray{ .ptr = ptr };
    }

    /// Create array with random uniform values
    pub fn randomUniform(dims: []const usize, low: f64, high: f64) !NDArray {
        var c_dims = try std.heap.c_allocator.alloc(usize, dims.len + 1);
        defer std.heap.c_allocator.free(c_dims);
        
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        
        const ptr = c.ndarray_new_randunif(c_dims.ptr, low, high);
        if (ptr == null) return error.AllocationFailed;
        
        return NDArray{ .ptr = ptr };
    }

    /// Create array with random normal values
    pub fn randomNormal(dims: []const usize, mean: f64, stddev: f64) !NDArray {
        var c_dims = try std.heap.c_allocator.alloc(usize, dims.len + 1);
        defer std.heap.c_allocator.free(c_dims);
        
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        
        const ptr = c.ndarray_new_randnorm(c_dims.ptr, mean, stddev);
        if (ptr == null) return error.AllocationFailed;
        
        return NDArray{ .ptr = ptr };
    }

    /// Create array with evenly spaced values
    pub fn arange(dims: []const usize, start: f64, stop: f64, step: f64) !NDArray {
        var c_dims = try std.heap.c_allocator.alloc(usize, dims.len + 1);
        defer std.heap.c_allocator.free(c_dims);
        
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        
        const ptr = c.ndarray_new_arange(c_dims.ptr, start, stop, step);
        if (ptr == null) return error.AllocationFailed;
        
        return NDArray{ .ptr = ptr };
    }

    /// Create array with linearly spaced values
    pub fn linspace(dims: []const usize, start: f64, stop: f64, num: usize) !NDArray {
        var c_dims = try std.heap.c_allocator.alloc(usize, dims.len + 1);
        defer std.heap.c_allocator.free(c_dims);
        
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        
        const ptr = c.ndarray_new_linspace(c_dims.ptr, start, stop, num);
        if (ptr == null) return error.AllocationFailed;
        
        return NDArray{ .ptr = ptr };
    }

    /// Create a copy of the array
    pub fn copy(self: NDArray) !NDArray {
        const ptr = c.ndarray_new_copy(self.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Get value at position
    pub fn get(self: NDArray, pos: []const usize) f64 {
        const c_pos = std.heap.c_allocator.alloc(usize, pos.len) catch unreachable;
        defer std.heap.c_allocator.free(c_pos);
        @memcpy(c_pos, pos);
        return c.ndarray_get(self.ptr, c_pos.ptr);
    }

    /// Set value at position
    pub fn set(self: NDArray, pos: []const usize, value: f64) void {
        const c_pos = std.heap.c_allocator.alloc(usize, pos.len) catch unreachable;
        defer std.heap.c_allocator.free(c_pos);
        @memcpy(c_pos, pos);
        c.ndarray_set(self.ptr, c_pos.ptr, value);
    }

    /// Print the array
    pub fn print(self: NDArray, name: ?[:0]const u8, precision: i32) void {
        const c_name = if (name) |n| n.ptr else null;
        c.ndarray_print(self.ptr, c_name, precision);
    }

    /// Element-wise addition (modifies self in place)
    pub fn add(self: NDArray, other: NDArray) void {
        _ = c.ndarray_add(self.ptr, other.ptr);
    }

    /// Element-wise multiplication (modifies self in place)
    pub fn mul(self: NDArray, other: NDArray) void {
        _ = c.ndarray_mul(self.ptr, other.ptr);
    }

    /// Add scalar (modifies self in place)
    pub fn addScalar(self: NDArray, scalar: f64) void {
        _ = c.ndarray_add_scalar(self.ptr, scalar);
    }

    /// Multiply by scalar (modifies self in place)
    pub fn mulScalar(self: NDArray, scalar: f64) void {
        _ = c.ndarray_mul_scalar(self.ptr, scalar);
    }

    /// Matrix multiplication
    pub fn matmul(self: NDArray, other: NDArray) !NDArray {
        const ptr = c.ndarray_new_matmul(self.ptr, other.ptr);
        if (ptr == null) return error.MatmulFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Tensor contraction over specified axes
    pub fn tensordot(self: NDArray, other: NDArray, axes_a: []const i32, axes_b: []const i32) !NDArray {
        var c_axes_a = try std.heap.c_allocator.alloc(i32, axes_a.len + 1);
        defer std.heap.c_allocator.free(c_axes_a);
        
        var c_axes_b = try std.heap.c_allocator.alloc(i32, axes_b.len + 1);
        defer std.heap.c_allocator.free(c_axes_b);
        
        @memcpy(c_axes_a[0..axes_a.len], axes_a);
        c_axes_a[axes_a.len] = -1; // Sentinel
        
        @memcpy(c_axes_b[0..axes_b.len], axes_b);
        c_axes_b[axes_b.len] = -1; // Sentinel
        
        const ptr = c.ndarray_new_tensordot(self.ptr, other.ptr, c_axes_a.ptr, c_axes_b.ptr);
        if (ptr == null) return error.TensordotFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Transpose
    pub fn transpose(self: NDArray) !NDArray {
        const ptr = c.ndarray_new_transpose(self.ptr);
        if (ptr == null) return error.TransposeFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Take a slice along an axis
    pub fn take(self: NDArray, axis: i32, start: usize, end: usize) !NDArray {
        const ptr = c.ndarray_new_take(self.ptr, axis, start, end);
        if (ptr == null) return error.TakeFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Save to binary file
    pub fn save(self: NDArray, filename: [:0]const u8) !void {
        const result = c.ndarray_save(self.ptr, filename.ptr);
        if (result != 0) return error.SaveFailed;
    }

    /// Load from binary file
    pub fn load(filename: [:0]const u8) !NDArray {
        const ptr = c.ndarray_load(filename.ptr);
        if (ptr == null) return error.LoadFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Get number of dimensions
    pub fn ndim(self: NDArray) usize {
        return self.ptr.*.ndim;
    }

    /// Get dimension sizes
    pub fn shape(self: NDArray, allocator: std.mem.Allocator) ![]usize {
        const dims = try allocator.alloc(usize, self.ndim());
        for (0..self.ndim()) |i| {
            dims[i] = self.ptr.*.dims[i];
        }
        return dims;
    }
};

/// Aggregation types
pub const AggrType = enum(c_int) {
    sum = c.NDARRAY_AGGR_SUM,
    mean = c.NDARRAY_AGGR_MEAN,
    max = c.NDARRAY_AGGR_MAX,
    min = c.NDARRAY_AGGR_MIN,
    std = c.NDARRAY_AGGR_STD,
};

/// Aggregate over axis
pub fn aggregate(arr: NDArray, axis: i32, aggr_type: AggrType) !NDArray {
    const ptr = c.ndarray_new_axis_aggr(arr.ptr, axis, @intFromEnum(aggr_type));
    if (ptr == null) return error.AggregateFailed;
    return NDArray{ .ptr = ptr };
}

/// Stack arrays along a new axis
pub fn stack(axis: i32, arrays: []const NDArray) !NDArray {
    // Create null-terminated array of NDArray pointers
    var c_array_list = try std.heap.c_allocator.alloc(c.NDArray, arrays.len + 1);
    defer std.heap.c_allocator.free(c_array_list);
    
    for (arrays, 0..) |arr, i| {
        c_array_list[i] = arr.ptr;
    }
    c_array_list[arrays.len] = null; // Null terminator
    
    const ptr = c.ndarray_new_stack(axis, c_array_list.ptr);
    if (ptr == null) return error.StackFailed;
    return NDArray{ .ptr = ptr };
}

/// Concatenate arrays along an existing axis
pub fn concat(axis: i32, arrays: []const NDArray) !NDArray {
    // Create null-terminated array of NDArray pointers
    var c_array_list = try std.heap.c_allocator.alloc(c.NDArray, arrays.len + 1);
    defer std.heap.c_allocator.free(c_array_list);
    
    for (arrays, 0..) |arr, i| {
        c_array_list[i] = arr.ptr;
    }
    c_array_list[arrays.len] = null; // Null terminator
    
    const ptr = c.ndarray_new_concat(axis, c_array_list.ptr);
    if (ptr == null) return error.ConcatFailed;
    return NDArray{ .ptr = ptr };
}

test "basic array creation" {
    const arr = try NDArray.zeros(&[_]usize{ 2, 3 });
    defer arr.deinit();
    
    try std.testing.expect(arr.ndim() == 2);
}

test "array from data" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const arr = try NDArray.fromData(&[_]usize{ 2, 2 }, &data);
    defer arr.deinit();
    
    try std.testing.expectEqual(@as(f64, 1.0), arr.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 4.0), arr.get(&[_]usize{ 1, 1 }));
}

test "matrix multiplication" {
    const a = try NDArray.ones(&[_]usize{ 2, 3 });
    defer a.deinit();
    
    const b = try NDArray.ones(&[_]usize{ 3, 2 });
    defer b.deinit();
    
    const result = try a.matmul(b);
    defer result.deinit();
    
    try std.testing.expectEqual(@as(f64, 3.0), result.get(&[_]usize{ 0, 0 }));
}

test "save and load" {
    const arr = try NDArray.full(&[_]usize{ 2, 2 }, 42.0);
    defer arr.deinit();
    
    try arr.save("test_zig.bin");
    
    const loaded = try NDArray.load("test_zig.bin");
    defer loaded.deinit();
    
    try std.testing.expectEqual(@as(f64, 42.0), loaded.get(&[_]usize{ 0, 0 }));
}
