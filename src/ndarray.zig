const std = @import("std");

const MAX_DIMS = 16;
const MAX_ARRAYS = 64;

pub const c = @cImport({
    @cInclude("stddef.h");
    @cInclude("ndarray.h");
});


pub const NDArray = struct {
    ptr: c.NDArray,

    /// a new ndarray with specified dimensions
    pub fn init(dims: []const usize) !NDArray {
        if (dims.len > MAX_DIMS) return error.TooManyDimensions;
        var c_dims: [MAX_DIMS + 1]usize = undefined;
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0; // Sentinel
        const ptr = c.ndarray_new(&c_dims);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Free the ndarray
    pub fn deinit(self: NDArray) void {
        c.ndarray_free(self.ptr);
    }

    /// array filled with zeros
    pub fn zeros(dims: []const usize) !NDArray {
        if (dims.len > MAX_DIMS) return error.TooManyDimensions;
        var c_dims: [MAX_DIMS + 1]usize = undefined;
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        const ptr = c.ndarray_new_zeros(&c_dims);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// array filled with ones
    pub fn ones(dims: []const usize) !NDArray {
        if (dims.len > MAX_DIMS) return error.TooManyDimensions;
        var c_dims: [MAX_DIMS + 1]usize = undefined;
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        const ptr = c.ndarray_new_ones(&c_dims);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// array filled with a specific value
    pub fn full(dims: []const usize, value: f64) !NDArray {
        if (dims.len > MAX_DIMS) return error.TooManyDimensions;
        var c_dims: [MAX_DIMS + 1]usize = undefined;
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        const ptr = c.ndarray_new_full(&c_dims, value);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// array from existing data
    pub fn fromData(dims: []const usize, data: []const f64) !NDArray {
        if (dims.len > MAX_DIMS) return error.TooManyDimensions;
        var c_dims: [MAX_DIMS + 1]usize = undefined;
        
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        
        const ptr = c.ndarray_new_from_data(&c_dims, @constCast(data.ptr));
        if (ptr == null) return error.AllocationFailed;
        
        return NDArray{ .ptr = ptr };
    }

    /// array with random uniform values
    pub fn randomUniform(dims: []const usize, low: f64, high: f64) !NDArray {
        if (dims.len > MAX_DIMS) return error.TooManyDimensions;
        var c_dims: [MAX_DIMS + 1]usize = undefined;
        
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        
        const ptr = c.ndarray_new_randunif(&c_dims, low, high);
        if (ptr == null) return error.AllocationFailed;
        
        return NDArray{ .ptr = ptr };
    }

    /// array with random normal values
    pub fn randomNormal(dims: []const usize, mean: f64, stddev: f64) !NDArray {
        if (dims.len > MAX_DIMS) return error.TooManyDimensions;
        var c_dims: [MAX_DIMS + 1]usize = undefined;
        
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        
        const ptr = c.ndarray_new_randnorm(&c_dims, mean, stddev);
        if (ptr == null) return error.AllocationFailed;
        
        return NDArray{ .ptr = ptr };
    }

    /// array with evenly spaced values
    pub fn arange(dims: []const usize, start: f64, stop: f64, step: f64) !NDArray {
        if (dims.len > MAX_DIMS) return error.TooManyDimensions;
        var c_dims: [MAX_DIMS + 1]usize = undefined;
        
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        
        const ptr = c.ndarray_new_arange(&c_dims, start, stop, step);
        if (ptr == null) return error.AllocationFailed;
        
        return NDArray{ .ptr = ptr };
    }

    /// array with linearly spaced values
    pub fn linspace(dims: []const usize, start: f64, stop: f64, num: usize) !NDArray {
        if (dims.len > MAX_DIMS) return error.TooManyDimensions;
        var c_dims: [MAX_DIMS + 1]usize = undefined;
        
        @memcpy(c_dims[0..dims.len], dims);
        c_dims[dims.len] = 0;
        
        const ptr = c.ndarray_new_linspace(&c_dims, start, stop, num);
        if (ptr == null) return error.AllocationFailed;
        
        return NDArray{ .ptr = ptr };
    }

    /// a copy of the array
    pub fn copy(self: NDArray) !NDArray {
        const ptr = c.ndarray_new_copy(self.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Get value at position
    pub fn get(self: NDArray, pos: []const usize) f64 {
        var c_pos: [MAX_DIMS]usize = undefined;
        @memcpy(c_pos[0..pos.len], pos);
        return c.ndarray_get(self.ptr, &c_pos);
    }

    /// Set value at position
    pub fn set(self: NDArray, pos: []const usize, value: f64) void {
        var c_pos: [MAX_DIMS]usize = undefined;
        @memcpy(c_pos[0..pos.len], pos);
        c.ndarray_set(self.ptr, &c_pos, value);
    }

    /// Set values along a slice at a specific index on an axis
    /// For 2D: axis=0 sets a row, axis=1 sets a column
    /// For higher dimensions: sets the hyperplane perpendicular to the axis
    pub fn setSlice(self: NDArray, axis: i32, index: usize, values: []const f64) void {
        c.ndarray_set_slice(self.ptr, axis, index, values.ptr);
    }

    /// Fill a slice with a scalar value at a specific index on an axis
    /// For 2D: axis=0 fills a row, axis=1 fills a column
    /// For higher dimensions: fills the hyperplane perpendicular to the axis
    pub fn fillSlice(self: NDArray, axis: i32, index: usize, value: f64) void {
        c.ndarray_fill_slice(self.ptr, axis, index, value);
    }

    /// Print the array
    pub fn print(self: NDArray, name: ?[:0]const u8, precision: i32) void {
        const c_name = if (name) |n| n.ptr else null;
        c.ndarray_print(self.ptr, c_name, precision);
    }

    /// Element-wise addition (modifies self in place)
    pub fn add(self: NDArray, other: NDArray) NDArray {
        _ = c.ndarray_add(self.ptr, other.ptr);
        return self;
    }

    /// Element-wise multiplication (modifies self in place)
    pub fn mul(self: NDArray, other: NDArray) NDArray {
        _ = c.ndarray_mul(self.ptr, other.ptr);
        return self;
    }

    /// Add scalar (modifies self in place)
    pub fn addScalar(self: NDArray, scalar: f64) NDArray {
        _ = c.ndarray_add_scalar(self.ptr, scalar);
        return self;
    }

    /// Multiply by scalar (modifies self in place)
    pub fn mulScalar(self: NDArray, scalar: f64) NDArray {
        _ = c.ndarray_mul_scalar(self.ptr, scalar);
        return self;
    }

    /// Linear combination: self = alpha*self + beta*other
    /// Computes self = alpha*self + beta*other and stores result in self
    pub fn axpby(self: NDArray, alpha: f64, other: NDArray, beta: f64) NDArray {
        _ = c.ndarray_axpby(self.ptr, alpha, other.ptr, beta);
        return self;
    }

    /// Scale and shift: self = alpha*self + beta
    pub fn scaleShift(self: NDArray, alpha: f64, beta: f64) NDArray {
        _ = c.ndarray_scale_shift(self.ptr, alpha, beta);
        return self;
    }

    /// Element-wise multiply then scale: self = self * other * scalar
    pub fn mulScaled(self: NDArray, other: NDArray, scalar: f64) NDArray {
        _ = c.ndarray_mul_scaled(self.ptr, other.ptr, scalar);
        return self;
    }

    /// Apply function to each element in place: self = func(self)
    pub fn mapFn(self: NDArray, func: *const fn (f64) callconv(.c) f64) NDArray {
        _ = c.ndarray_mapfnc(self.ptr, func);
        return self;
    }

    /// Map function then multiply: self = func(self) * other * alpha
    pub fn mapMul(self: NDArray, func: *const fn (f64) callconv(.c) f64, 
                  other: NDArray, alpha: f64) void {
        _ = c.ndarray_map_mul(self.ptr, func, other.ptr, alpha);
    }

    /// Fused multiply-add: dest = alpha * (self * other) + beta * dest
    pub fn mulAdd(self: NDArray, other: NDArray, dest: NDArray, 
                  alpha: f64, beta: f64) void {
        _ = c.ndarray_mul_add(self.ptr, other.ptr, dest.ptr, alpha, beta);
    }

    /// Matrix-vector multiply: y = alpha * self * x + beta * y
    pub fn gemv(self: NDArray, x: NDArray, alpha: f64, beta: f64, y: NDArray) void {
        _ = c.ndarray_gemv(self.ptr, x.ptr, alpha, beta, y.ptr);
    }

    /// Clip values below minimum threshold
    pub fn clipMin(self: NDArray, min_val: f64) NDArray {
        _ = c.ndarray_clip_min(self.ptr, min_val);
        return self;
    }

    /// Clip values above maximum threshold
    pub fn clipMax(self: NDArray, max_val: f64) NDArray {
        _ = c.ndarray_clip_max(self.ptr, max_val);
        return self;
    }

    /// Clip values to range [min_val, max_val]
    pub fn clip(self: NDArray, min_val: f64, max_val: f64) void {
        _ = c.ndarray_clip(self.ptr, min_val, max_val);
    }

    /// Absolute value (modifies self in place)
    pub fn abs(self: NDArray) void {
        _ = c.ndarray_abs(self.ptr);
    }

    /// Sign function: -1, 0, or +1 (modifies self in place)
    pub fn sign(self: NDArray) void {
        _ = c.ndarray_sign(self.ptr);
    }

    /// Get pointer to a slice along an axis
    /// Returns pointer valid as long as array exists. User must respect bounds.
    pub fn getSlicePtr(self: NDArray, axis: i32, index: usize) [*]f64 {
        return c.ndarray_get_slice_ptr(self.ptr, axis, index);
    }

    /// Copy a slice from one array to another
    pub fn copySlice(src: NDArray, src_axis: i32, src_idx: usize,
                     dst: NDArray, dst_axis: i32, dst_idx: usize) void {
        c.ndarray_copy_slice(src.ptr, src_axis, src_idx, dst.ptr, dst_axis, dst_idx);
    }

    /// Get the size of a slice along an axis
    pub fn getSliceSize(self: NDArray, axis: i32) usize {
        return c.ndarray_get_slice_size(self.ptr, axis);
    }

    /// Element-wise equality comparison
    pub fn equal(self: NDArray, other: NDArray) !NDArray {
        const ptr = c.ndarray_new_equal(self.ptr, other.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Element-wise less-than comparison
    pub fn less(self: NDArray, other: NDArray) !NDArray {
        const ptr = c.ndarray_new_less(self.ptr, other.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Element-wise greater-than comparison
    pub fn greater(self: NDArray, other: NDArray) !NDArray {
        const ptr = c.ndarray_new_greater(self.ptr, other.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Scalar equality comparison
    pub fn equalScalar(self: NDArray, value: f64) !NDArray {
        const ptr = c.ndarray_new_equal_scalar(self.ptr, value);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Scalar less-than comparison
    pub fn lessScalar(self: NDArray, value: f64) !NDArray {
        const ptr = c.ndarray_new_less_scalar(self.ptr, value);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Scalar greater-than comparison
    pub fn greaterScalar(self: NDArray, value: f64) !NDArray {
        const ptr = c.ndarray_new_greater_scalar(self.ptr, value);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Logical AND
    pub fn logicalAnd(self: NDArray, other: NDArray) !NDArray {
        const ptr = c.ndarray_logical_and(self.ptr, other.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Logical OR
    pub fn logicalOr(self: NDArray, other: NDArray) !NDArray {
        const ptr = c.ndarray_logical_or(self.ptr, other.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Logical NOT
    pub fn logicalNot(self: NDArray) !NDArray {
        const ptr = c.ndarray_logical_not(self.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// NumPy-style where: result = condition ? x : y
    pub fn where(condition: NDArray, x: NDArray, y: NDArray) !NDArray {
        const ptr = c.ndarray_where(condition.ptr, x.ptr, y.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Matrix multiplication
    pub fn matmul(self: NDArray, other: NDArray) !NDArray {
        const ptr = c.ndarray_new_matmul(self.ptr, other.ptr);
        if (ptr == null) return error.MatmulFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Tensor contraction over specified axes
    pub fn tensordot(self: NDArray, other: NDArray, axes_a: []const i32, axes_b: []const i32) !NDArray {
        if (axes_a.len > MAX_DIMS or axes_b.len > MAX_DIMS) return error.TooManyDimensions;
        var c_axes_a: [MAX_DIMS + 1]i32 = undefined;
        var c_axes_b: [MAX_DIMS + 1]i32 = undefined;
        
        @memcpy(c_axes_a[0..axes_a.len], axes_a);
        c_axes_a[axes_a.len] = -1; // Sentinel
        
        @memcpy(c_axes_b[0..axes_b.len], axes_b);
        c_axes_b[axes_b.len] = -1; // Sentinel
        
        const ptr = c.ndarray_new_tensordot(self.ptr, other.ptr, &c_axes_a, &c_axes_b);
        if (ptr == null) return error.TensordotFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Transpose
    pub fn transpose(self: NDArray) !NDArray {
        const ptr = c.ndarray_new_transpose(self.ptr);
        if (ptr == null) return error.TransposeFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Reshape the ndarray in-place to new dimensions
    /// Use -1 for one dimension to automatically infer its size
    pub fn reshape(self: NDArray, new_dims: []const isize) !void {
        if (new_dims.len > MAX_DIMS) return error.TooManyDimensions;
        var c_dims: [MAX_DIMS + 1]usize = undefined;
        for (new_dims, 0..) |dim, i| {
            if (dim == -1) {
                c_dims[i] = @as(usize, @bitCast(@as(isize, -1)));
            } else {
                c_dims[i] = @intCast(dim);
            }
        }
        c_dims[new_dims.len] = 0; // Sentinel
        c.ndarray_reshape(self.ptr, &c_dims);
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
    sum = c.NDA_AGGR_SUM,
    mean = c.NDA_AGGR_MEAN,
    max = c.NDA_AGGR_MAX,
    min = c.NDA_AGGR_MIN,
    std = c.NDA_AGGR_STD,
};

/// Aggregate over axis
pub fn aggregate(arr: NDArray, axis: i32, aggr_type: AggrType) !NDArray {
    const ptr = c.ndarray_new_aggr(arr.ptr, axis, @intFromEnum(aggr_type));
    if (ptr == null) return error.AggregateFailed;
    return NDArray{ .ptr = ptr };
}

/// Aggregate all elements to a scalar value
pub fn scalarAggregate(arr: NDArray, aggr_type: AggrType) f64 {
    return c.ndarray_scalar_aggr(arr.ptr, @intFromEnum(aggr_type));
}

/// Stack arrays along a new axis
pub fn stack(axis: i32, arrays: []const NDArray) !NDArray {
    if (arrays.len > MAX_ARRAYS) return error.TooManyArrays;
    var c_array_list: [MAX_ARRAYS + 1]c.NDArray = undefined;
    
    for (arrays, 0..) |arr, i| {
        c_array_list[i] = arr.ptr;
    }
    c_array_list[arrays.len] = null; // Null terminator
    
    const ptr = c.ndarray_new_stack(axis, &c_array_list);
    if (ptr == null) return error.StackFailed;
    return NDArray{ .ptr = ptr };
}

/// Concatenate arrays along an existing axis
pub fn concat(axis: i32, arrays: []const NDArray) !NDArray {
    if (arrays.len > MAX_ARRAYS) return error.TooManyArrays;
    var c_array_list: [MAX_ARRAYS + 1]c.NDArray = undefined;
    
    for (arrays, 0..) |arr, i| {
        c_array_list[i] = arr.ptr;
    }
    c_array_list[arrays.len] = null; // Null terminator
    
    const ptr = c.ndarray_new_concat(axis, &c_array_list);
    if (ptr == null) return error.ConcatFailed;
    return NDArray{ .ptr = ptr };
}

test "init and deinit" {
    const arr = try NDArray.init(&[_]usize{ 2, 3 });
    defer arr.deinit();
    try std.testing.expect(arr.ndim() == 2);
}

test "zeros" {
    const arr = try NDArray.zeros(&[_]usize{ 2, 3 });
    defer arr.deinit();
    try std.testing.expectEqual(@as(f64, 0.0), arr.get(&[_]usize{ 0, 0 }));
}

test "ones" {
    const arr = try NDArray.ones(&[_]usize{ 2, 3 });
    defer arr.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), arr.get(&[_]usize{ 1, 2 }));
}

test "full" {
    const arr = try NDArray.full(&[_]usize{ 2, 2 }, 5.5);
    defer arr.deinit();
    try std.testing.expectEqual(@as(f64, 5.5), arr.get(&[_]usize{ 0, 0 }));
}

test "fromData" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const arr = try NDArray.fromData(&[_]usize{ 2, 2 }, &data);
    defer arr.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), arr.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 4.0), arr.get(&[_]usize{ 1, 1 }));
}

test "randomUniform" {
    const arr = try NDArray.randomUniform(&[_]usize{ 2, 3 }, 0.0, 1.0);
    defer arr.deinit();
    const val = arr.get(&[_]usize{ 0, 0 });
    try std.testing.expect(val >= 0.0 and val <= 1.0);
}

test "randomNormal" {
    const arr = try NDArray.randomNormal(&[_]usize{ 2, 3 }, 0.0, 1.0);
    defer arr.deinit();
    try std.testing.expect(arr.ndim() == 2);
}

test "arange" {
    const arr = try NDArray.arange(&[_]usize{ 2, 3 }, 0.0, 6.0, 1.0);
    defer arr.deinit();
    try std.testing.expectEqual(@as(f64, 0.0), arr.get(&[_]usize{ 0, 0 }));
}

test "linspace" {
    const arr = try NDArray.linspace(&[_]usize{ 2, 3 }, 0.0, 5.0, 6);
    defer arr.deinit();
    try std.testing.expectEqual(@as(f64, 0.0), arr.get(&[_]usize{ 0, 0 }));
}

test "copy" {
    const arr = try NDArray.ones(&[_]usize{ 2, 2 });
    defer arr.deinit();
    const copy = try arr.copy();
    defer copy.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), copy.get(&[_]usize{ 0, 0 }));
}

test "get and set" {
    const arr = try NDArray.zeros(&[_]usize{ 2, 2 });
    defer arr.deinit();
    arr.set(&[_]usize{ 1, 1 }, 7.5);
    try std.testing.expectEqual(@as(f64, 7.5), arr.get(&[_]usize{ 1, 1 }));
}

test "setSlice" {
    const arr = try NDArray.zeros(&[_]usize{ 2, 3 });
    defer arr.deinit();
    const values = [_]f64{ 1.0, 2.0, 3.0 };
    arr.setSlice(0, 0, &values);
    try std.testing.expectEqual(@as(f64, 1.0), arr.get(&[_]usize{ 0, 0 }));
}

test "fillSlice" {
    const arr = try NDArray.zeros(&[_]usize{ 2, 3 });
    defer arr.deinit();
    arr.fillSlice(0, 1, 9.0);
    try std.testing.expectEqual(@as(f64, 9.0), arr.get(&[_]usize{ 1, 0 }));
}

test "print" {
    const arr = try NDArray.ones(&[_]usize{ 2, 2 });
    defer arr.deinit();
    arr.print(null, 2);
}

test "add" {
    const a = try NDArray.ones(&[_]usize{ 2, 2 });
    defer a.deinit();
    const b = try NDArray.ones(&[_]usize{ 2, 2 });
    defer b.deinit();
    _ = a.add(b);
    try std.testing.expectEqual(@as(f64, 2.0), a.get(&[_]usize{ 0, 0 }));
}

test "mul" {
    const a = try NDArray.full(&[_]usize{ 2, 2 }, 3.0);
    defer a.deinit();
    const b = try NDArray.full(&[_]usize{ 2, 2 }, 2.0);
    defer b.deinit();
    _ = a.mul(b);
    try std.testing.expectEqual(@as(f64, 6.0), a.get(&[_]usize{ 0, 0 }));
}

test "addScalar" {
    const arr = try NDArray.ones(&[_]usize{ 2, 2 });
    defer arr.deinit();
    _ = arr.addScalar(5.0);
    try std.testing.expectEqual(@as(f64, 6.0), arr.get(&[_]usize{ 0, 0 }));
}

test "mulScalar" {
    const arr = try NDArray.full(&[_]usize{ 2, 2 }, 3.0);
    defer arr.deinit();
    _ = arr.mulScalar(2.0);
    try std.testing.expectEqual(@as(f64, 6.0), arr.get(&[_]usize{ 0, 0 }));
}

test "axpby" {
    const a = try NDArray.full(&[_]usize{ 2, 2 }, 2.0);
    defer a.deinit();
    const b = try NDArray.full(&[_]usize{ 2, 2 }, 3.0);
    defer b.deinit();
    _ = a.axpby(2.0, b, 3.0);
    try std.testing.expectEqual(@as(f64, 13.0), a.get(&[_]usize{ 0, 0 }));
}

test "scaleShift" {
    const arr = try NDArray.full(&[_]usize{ 2, 2 }, 2.0);
    defer arr.deinit();
    _ = arr.scaleShift(3.0, 1.0);
    try std.testing.expectEqual(@as(f64, 7.0), arr.get(&[_]usize{ 0, 0 }));
}

test "mulScaled" {
    const a = try NDArray.full(&[_]usize{ 2, 2 }, 2.0);
    defer a.deinit();
    const b = try NDArray.full(&[_]usize{ 2, 2 }, 3.0);
    defer b.deinit();
    _ = a.mulScaled(b, 2.0);
    try std.testing.expectEqual(@as(f64, 12.0), a.get(&[_]usize{ 0, 0 }));
}

fn testFunc(x: f64) callconv(.c) f64 {
    return x * 2.0;
}

test "mapFn" {
    const arr = try NDArray.full(&[_]usize{ 2, 2 }, 3.0);
    defer arr.deinit();
    _ = arr.mapFn(&testFunc);
    try std.testing.expectEqual(@as(f64, 6.0), arr.get(&[_]usize{ 0, 0 }));
}

test "mapMul" {
    const a = try NDArray.full(&[_]usize{ 2, 2 }, 3.0);
    defer a.deinit();
    const b = try NDArray.full(&[_]usize{ 2, 2 }, 2.0);
    defer b.deinit();
    a.mapMul(&testFunc, b, 1.0);
    try std.testing.expectEqual(@as(f64, 12.0), a.get(&[_]usize{ 0, 0 }));
}

test "mulAdd" {
    const a = try NDArray.full(&[_]usize{ 2, 2 }, 2.0);
    defer a.deinit();
    const b = try NDArray.full(&[_]usize{ 2, 2 }, 3.0);
    defer b.deinit();
    const dest = try NDArray.full(&[_]usize{ 2, 2 }, 1.0);
    defer dest.deinit();
    a.mulAdd(b, dest, 1.0, 1.0);
    try std.testing.expectEqual(@as(f64, 7.0), dest.get(&[_]usize{ 0, 0 }));
}

test "gemv" {
    const a = try NDArray.ones(&[_]usize{ 2, 3 });
    defer a.deinit();
    const x = try NDArray.ones(&[_]usize{ 3, 1 });
    defer x.deinit();
    const y = try NDArray.zeros(&[_]usize{ 2, 1 });
    defer y.deinit();
    a.gemv(x, 1.0, 0.0, y);
    try std.testing.expectEqual(@as(f64, 3.0), y.get(&[_]usize{ 0, 0 }));
}

test "clipMin" {
    const arr = try NDArray.full(&[_]usize{ 2, 2 }, 1.0);
    defer arr.deinit();
    _ = arr.clipMin(2.0);
    try std.testing.expectEqual(@as(f64, 2.0), arr.get(&[_]usize{ 0, 0 }));
}

test "clipMax" {
    const arr = try NDArray.full(&[_]usize{ 2, 2 }, 5.0);
    defer arr.deinit();
    _ = arr.clipMax(3.0);
    try std.testing.expectEqual(@as(f64, 3.0), arr.get(&[_]usize{ 0, 0 }));
}

test "clip" {
    const arr = try NDArray.full(&[_]usize{ 2, 2 }, 5.0);
    defer arr.deinit();
    arr.clip(2.0, 3.0);
    try std.testing.expectEqual(@as(f64, 3.0), arr.get(&[_]usize{ 0, 0 }));
}

test "abs" {
    const arr = try NDArray.full(&[_]usize{ 2, 2 }, -5.0);
    defer arr.deinit();
    arr.abs();
    try std.testing.expectEqual(@as(f64, 5.0), arr.get(&[_]usize{ 0, 0 }));
}

test "sign" {
    const arr = try NDArray.full(&[_]usize{ 2, 2 }, -5.0);
    defer arr.deinit();
    arr.sign();
    try std.testing.expectEqual(@as(f64, -1.0), arr.get(&[_]usize{ 0, 0 }));
}

test "getSlicePtr" {
    const arr = try NDArray.ones(&[_]usize{ 2, 3 });
    defer arr.deinit();
    const ptr = arr.getSlicePtr(0, 0);
    try std.testing.expectEqual(@as(f64, 1.0), ptr[0]);
}

test "copySlice" {
    const src = try NDArray.ones(&[_]usize{ 2, 3 });
    defer src.deinit();
    const dst = try NDArray.zeros(&[_]usize{ 2, 3 });
    defer dst.deinit();
    NDArray.copySlice(src, 0, 0, dst, 0, 1);
    try std.testing.expectEqual(@as(f64, 1.0), dst.get(&[_]usize{ 1, 0 }));
}

test "getSliceSize" {
    const arr = try NDArray.ones(&[_]usize{ 2, 3 });
    defer arr.deinit();
    const size = arr.getSliceSize(0);
    try std.testing.expectEqual(@as(usize, 3), size);
}

test "equal" {
    const a = try NDArray.ones(&[_]usize{ 2, 2 });
    defer a.deinit();
    const b = try NDArray.ones(&[_]usize{ 2, 2 });
    defer b.deinit();
    const result = try a.equal(b);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&[_]usize{ 0, 0 }));
}

test "less" {
    const a = try NDArray.ones(&[_]usize{ 2, 2 });
    defer a.deinit();
    const b = try NDArray.full(&[_]usize{ 2, 2 }, 2.0);
    defer b.deinit();
    const result = try a.less(b);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&[_]usize{ 0, 0 }));
}

test "greater" {
    const a = try NDArray.full(&[_]usize{ 2, 2 }, 2.0);
    defer a.deinit();
    const b = try NDArray.ones(&[_]usize{ 2, 2 });
    defer b.deinit();
    const result = try a.greater(b);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&[_]usize{ 0, 0 }));
}

test "equalScalar" {
    const arr = try NDArray.ones(&[_]usize{ 2, 2 });
    defer arr.deinit();
    const result = try arr.equalScalar(1.0);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&[_]usize{ 0, 0 }));
}

test "lessScalar" {
    const arr = try NDArray.ones(&[_]usize{ 2, 2 });
    defer arr.deinit();
    const result = try arr.lessScalar(2.0);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&[_]usize{ 0, 0 }));
}

test "greaterScalar" {
    const arr = try NDArray.full(&[_]usize{ 2, 2 }, 2.0);
    defer arr.deinit();
    const result = try arr.greaterScalar(1.0);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&[_]usize{ 0, 0 }));
}

test "logicalAnd" {
    const a = try NDArray.ones(&[_]usize{ 2, 2 });
    defer a.deinit();
    const b = try NDArray.ones(&[_]usize{ 2, 2 });
    defer b.deinit();
    const result = try a.logicalAnd(b);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&[_]usize{ 0, 0 }));
}

test "logicalOr" {
    const a = try NDArray.zeros(&[_]usize{ 2, 2 });
    defer a.deinit();
    const b = try NDArray.ones(&[_]usize{ 2, 2 });
    defer b.deinit();
    const result = try a.logicalOr(b);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&[_]usize{ 0, 0 }));
}

test "logicalNot" {
    const arr = try NDArray.zeros(&[_]usize{ 2, 2 });
    defer arr.deinit();
    const result = try arr.logicalNot();
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&[_]usize{ 0, 0 }));
}

test "where" {
    const condition = try NDArray.ones(&[_]usize{ 2, 2 });
    defer condition.deinit();
    const x = try NDArray.full(&[_]usize{ 2, 2 }, 10.0);
    defer x.deinit();
    const y = try NDArray.full(&[_]usize{ 2, 2 }, 20.0);
    defer y.deinit();
    const result = try NDArray.where(condition, x, y);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 10.0), result.get(&[_]usize{ 0, 0 }));
}

test "matmul" {
    const a = try NDArray.ones(&[_]usize{ 2, 3 });
    defer a.deinit();
    const b = try NDArray.ones(&[_]usize{ 3, 2 });
    defer b.deinit();
    const result = try a.matmul(b);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 3.0), result.get(&[_]usize{ 0, 0 }));
}

test "tensordot" {
    const a = try NDArray.ones(&[_]usize{ 2, 3, 4 });
    defer a.deinit();
    const b = try NDArray.ones(&[_]usize{ 4, 5 });
    defer b.deinit();
    const axes_a = [_]i32{2};
    const axes_b = [_]i32{0};
    const result = try a.tensordot(b, &axes_a, &axes_b);
    defer result.deinit();
    try std.testing.expect(result.ndim() == 3);
}

test "transpose" {
    const arr = try NDArray.ones(&[_]usize{ 2, 3 });
    defer arr.deinit();
    const result = try arr.transpose();
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&[_]usize{ 2, 1 }));
}

test "reshape" {
    const arr = try NDArray.ones(&[_]usize{ 2, 6 });
    defer arr.deinit();
    try arr.reshape(&[_]isize{ 3, 4 });
    try std.testing.expectEqual(@as(f64, 1.0), arr.get(&[_]usize{ 2, 3 }));
}

test "take" {
    const arr = try NDArray.ones(&[_]usize{ 3, 4 });
    defer arr.deinit();
    const result = try arr.take(0, 0, 2);
    defer result.deinit();
    try std.testing.expect(result.ndim() == 2);
}

test "save and load" {
    const arr = try NDArray.full(&[_]usize{ 2, 2 }, 42.0);
    defer arr.deinit();
    try arr.save("test_zig_saveload.bin");
    const loaded = try NDArray.load("test_zig_saveload.bin");
    defer loaded.deinit();
    try std.testing.expectEqual(@as(f64, 42.0), loaded.get(&[_]usize{ 0, 0 }));
}

test "ndim and shape" {
    const arr = try NDArray.ones(&[_]usize{ 2, 3, 4 });
    defer arr.deinit();
    try std.testing.expectEqual(@as(usize, 3), arr.ndim());
    const allocator = std.testing.allocator;
    const dims = try arr.shape(allocator);
    defer allocator.free(dims);
    try std.testing.expectEqual(@as(usize, 2), dims[0]);
    try std.testing.expectEqual(@as(usize, 3), dims[1]);
    try std.testing.expectEqual(@as(usize, 4), dims[2]);
}

test "aggregate sum" {
    const arr = try NDArray.ones(&[_]usize{ 2, 3 });
    defer arr.deinit();
    const result = try aggregate(arr, 0, AggrType.sum);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 2.0), result.get(&[_]usize{ 0, 0 }));
}

test "aggregate mean" {
    const arr = try NDArray.full(&[_]usize{ 2, 3 }, 4.0);
    defer arr.deinit();
    const result = try aggregate(arr, 1, AggrType.mean);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 4.0), result.get(&[_]usize{ 0, 0 }));
}

test "scalar aggregate sum" {
    const data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const arr = try NDArray.fromData(&[_]usize{ 3, 4 }, &data);
    defer arr.deinit();
    const result = scalarAggregate(arr, AggrType.sum);
    try std.testing.expectEqual(@as(f64, 78.0), result);
}

test "scalar aggregate mean" {
    const data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const arr = try NDArray.fromData(&[_]usize{ 3, 4 }, &data);
    defer arr.deinit();
    const result = scalarAggregate(arr, AggrType.mean);
    try std.testing.expectEqual(@as(f64, 6.5), result);
}

test "scalar aggregate max" {
    const data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const arr = try NDArray.fromData(&[_]usize{ 3, 4 }, &data);
    defer arr.deinit();
    const result = scalarAggregate(arr, AggrType.max);
    try std.testing.expectEqual(@as(f64, 12.0), result);
}

test "scalar aggregate min" {
    const data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const arr = try NDArray.fromData(&[_]usize{ 3, 4 }, &data);
    defer arr.deinit();
    const result = scalarAggregate(arr, AggrType.min);
    try std.testing.expectEqual(@as(f64, 1.0), result);
}

test "scalar aggregate std" {
    const data = [_]f64{ 2, 4, 4, 4, 5, 5, 5, 7, 9, 9 };
    const arr = try NDArray.fromData(&[_]usize{ 2, 5 }, &data);
    defer arr.deinit();
    const result = scalarAggregate(arr, AggrType.std);
    try std.testing.expectApproxEqRel(@as(f64, 2.1540659228538015), result, 1e-10);
}

test "scalar aggregate consistency" {
    const arr = try NDArray.ones(&[_]usize{ 10, 10 });
    defer arr.deinit();
    
    // Compare scalar aggregate with array aggregate
    const result_array = try aggregate(arr, c.NDA_ALL_AXES, AggrType.sum);
    defer result_array.deinit();
    const result_scalar = scalarAggregate(arr, AggrType.sum);
    
    try std.testing.expectEqual(result_array.get(&[_]usize{ 0, 0 }), result_scalar);
}

test "stack" {
    const a = try NDArray.ones(&[_]usize{ 2, 3 });
    defer a.deinit();
    const b = try NDArray.ones(&[_]usize{ 2, 3 });
    defer b.deinit();
    const arrays = [_]NDArray{ a, b };
    const result = try stack(0, &arrays);
    defer result.deinit();
    try std.testing.expect(result.ndim() == 3);
}

test "concat" {
    const a = try NDArray.ones(&[_]usize{ 2, 3 });
    defer a.deinit();
    const b = try NDArray.ones(&[_]usize{ 2, 3 });
    defer b.deinit();
    const arrays = [_]NDArray{ a, b };
    const result = try concat(0, &arrays);
    defer result.deinit();
    try std.testing.expect(result.ndim() == 2);
}
