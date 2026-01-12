const std = @import("std");

const MAX_DIMS = 64;
const MAX_ARRAYS = 64;

pub const c = @cImport({
    @cInclude("stddef.h");
    @cInclude("ndarray.h");
});

threadlocal var c_dims_buffer: [MAX_DIMS + 1]usize = undefined;

/// Convert Zig dimensions to C dimensions.
/// Appends a 0 sentinel at the end as required by C API.
fn zigDimsToCDims(dims: []const usize) ![*]usize {
    if (dims.len > MAX_DIMS) return error.TooManyDimensions;
    @memcpy(c_dims_buffer[0..dims.len], dims);
    c_dims_buffer[dims.len] = 0;
    return &c_dims_buffer;
}

/// Aggregation types for reduction operations.
/// 
/// Used with `initAggregate` and `scalarAggregate` methods.
pub const AggrType = enum(c_int) {
    /// Sum of elements
    sum = c.NDA_AGGR_SUM,
    /// Mean (average) of elements
    mean = c.NDA_AGGR_MEAN,
    /// Maximum element value
    max = c.NDA_AGGR_MAX,
    /// Minimum element value
    min = c.NDA_AGGR_MIN,
    /// Standard deviation of elements
    std = c.NDA_AGGR_STD,
};

/// Multi-dimensional array of doubles with N-dimensional support.
/// 
/// This is a wrapper around the C ndarray library that provides
/// a Zig-friendly interface. All arrays must have at least 2 dimensions.
/// 
/// Memory management: Call `deinit()` when done to free allocated memory.
pub const NDArray = struct {
    /// Pointer to the underlying C ndarray structure
    ptr: c.NDArray,

    /// Creates a new ndarray with specified dimensions.
    /// 
    /// All elements are uninitialized. Use `initZeros`, `initOnes`, or 
    /// `initFull` for initialized arrays.
    /// 
    /// **Parameters:**
    /// - `dims`: Slice of dimension sizes (must have at least 2 elements)
    /// 
    /// **Returns:** NDArray or error
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.init(&.{3, 4});
    /// defer arr.deinit();
    /// ```
    pub fn init(dims: []const usize) !NDArray {
        var arr: NDArray = undefined;
        const c_dims = try zigDimsToCDims(dims);
        const ptr = c.ndarray_new(c_dims);
        if (ptr == null) return error.AllocationFailed;
        arr.ptr = ptr;
        return arr;
    }

    /// Frees the memory allocated for the ndarray.
    /// 
    /// Must be called when done using the array to prevent memory leaks.
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.init(&.{3, 4});
    /// defer arr.deinit();
    /// ```
    pub fn deinit(self: NDArray) void {
        c.ndarray_free(self.ptr);
    }

    /// Creates a new ndarray filled with zeros.
    /// 
    /// **Parameters:**
    /// - `dims`: Slice of dimension sizes (must have at least 2 elements)
    /// 
    /// **Returns:** NDArray filled with 0.0 or error
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initZeros(&.{3, 4});
    /// defer arr.deinit();
    /// ```
    pub fn initZeros(dims: []const usize) !NDArray {
        var arr: NDArray = undefined;
        const c_dims = try zigDimsToCDims(dims);
        const ptr = c.ndarray_new_zeros(c_dims);
        if (ptr == null) return error.AllocationFailed;
        arr.ptr = ptr;
        return arr;
    }

    /// Creates a new ndarray filled with ones.
    /// 
    /// **Parameters:**
    /// - `dims`: Slice of dimension sizes (must have at least 2 elements)
    /// 
    /// **Returns:** NDArray filled with 1.0 or error
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initOnes(&.{3, 4});
    /// defer arr.deinit();
    /// ```
    pub fn initOnes(dims: []const usize) !NDArray {
        var arr: NDArray = undefined;
        const c_dims = try zigDimsToCDims(dims);
        const ptr = c.ndarray_new_ones(c_dims);
        if (ptr == null) return error.AllocationFailed;
        arr.ptr = ptr;
        return arr;
    }

    /// Creates a new ndarray filled with a specific value.
    /// 
    /// **Parameters:**
    /// - `dims`: Slice of dimension sizes (must have at least 2 elements)
    /// - `value`: The value to fill all elements with
    /// 
    /// **Returns:** NDArray filled with specified value or error
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initFull(&.{3, 4}, 5.0);
    /// defer arr.deinit();
    /// ```
    pub fn initFull(dims: []const usize, value: f64) !NDArray {
        var arr: NDArray = undefined;
        const c_dims = try zigDimsToCDims(dims);
        const ptr = c.ndarray_new_full(c_dims, value);
        if (ptr == null) return error.AllocationFailed;
        arr.ptr = ptr;
        return arr;
    }

    /// Creates a new ndarray from existing data.
    /// 
    /// The data is copied into the new array.
    /// 
    /// **Parameters:**
    /// - `dims`: Slice of dimension sizes (must have at least 2 elements)
    /// - `data`: Slice of f64 values (size must match product of dimensions)
    /// 
    /// **Returns:** NDArray containing copied data or error
    /// 
    /// **Example:**
    /// ```zig
    /// const data = [_]f64{1.0, 2.0, 3.0, 4.0};
    /// const arr = try NDArray.initFromData(&.{2, 2}, &data);
    /// defer arr.deinit();
    /// ```
    pub fn initFromData(dims: []const usize, data: []const f64) !NDArray {
        var arr: NDArray = undefined;
        const c_dims = try zigDimsToCDims(dims);
        const ptr = c.ndarray_new_from_data(c_dims, @constCast(data.ptr));
        if (ptr == null) return error.AllocationFailed;
        arr.ptr = ptr;
        return arr;
    }

    /// Creates a new ndarray with random uniform values.
    /// 
    /// Values are uniformly distributed between low (inclusive) and high (exclusive).
    /// 
    /// **Parameters:**
    /// - `dims`: Slice of dimension sizes (must have at least 2 elements)
    /// - `low`: Lower bound (inclusive)
    /// - `high`: Upper bound (exclusive)
    /// 
    /// **Returns:** NDArray with random uniform values or error
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initRandomUniform(&.{3, 4}, 0.0, 1.0);
    /// defer arr.deinit();
    /// ```
    pub fn initRandomUniform(dims: []const usize, low: f64, high: f64) !NDArray {
        var arr: NDArray = undefined;
        const c_dims = try zigDimsToCDims(dims);
        const ptr = c.ndarray_new_randunif(c_dims, low, high);
        if (ptr == null) return error.AllocationFailed;
        arr.ptr = ptr;
        return arr;
    }

    /// Creates a new ndarray with random normal values.
    /// 
    /// Values follow a Gaussian distribution with specified mean and standard deviation.
    /// 
    /// **Parameters:**
    /// - `dims`: Slice of dimension sizes (must have at least 2 elements)
    /// - `mean`: Mean of the distribution
    /// - `stddev`: Standard deviation of the distribution
    /// 
    /// **Returns:** NDArray with random normal values or error
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initRandomNormal(&.{3, 4}, 0.0, 1.0);
    /// defer arr.deinit();
    /// ```
    pub fn initRandomNormal(dims: []const usize, mean: f64, stddev: f64) !NDArray {
        var arr: NDArray = undefined;
        const c_dims = try zigDimsToCDims(dims);
        const ptr = c.ndarray_new_randnorm(c_dims, mean, stddev);
        if (ptr == null) return error.AllocationFailed;
        arr.ptr = ptr;
        return arr;
    }

    /// Creates a new ndarray with random poisson-distributed values.
    ///
    /// Values follow a Poisson distribution with lambda parameter.
    ///
    /// **Parameters:**
    /// - `dims`: Slice of dimension sizes (must have at least 2 elements)
    /// - `lambda`: mean of the distribution
    ///
    /// **Returns:** NDArray with random poisson-distributed values or error
    ///
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initRandomPoisson(&.{3, 4}, 5.0);
    /// defer arr.deinit();
    /// ```
    pub fn initRandomPoisson(dims: []const usize, lambda: f64) !NDArray {
        var arr: NDArray = undefined;
        const c_dims = try zigDimsToCDims(dims);
        const ptr = c.ndarray_new_randpoisson(c_dims, lambda);
        if (ptr == null) return error.AllocationFailed;
        arr.ptr = ptr;
        return arr;
    }

    /// Creates a new ndarray with evenly spaced values in a range.
    /// 
    /// Values are generated sequentially: start, start+step, start+2*step, ...
    /// and filled in row-major order.
    /// 
    /// **Parameters:**
    /// - `dims`: Slice of dimension sizes (must have at least 2 elements)
    /// - `start`: Starting value (inclusive)
    /// - `stop`: Ending value (exclusive)
    /// - `step`: Step size between values
    /// 
    /// **Returns:** NDArray with evenly spaced values or error
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initArange(&.{2, 5}, 0.0, 10.0, 1.0);
    /// defer arr.deinit();
    /// ```
    pub fn initArange(dims: []const usize, start: f64, stop: f64, step: f64) !NDArray {
        var arr: NDArray = undefined;
        const c_dims = try zigDimsToCDims(dims);
        const ptr = c.ndarray_new_arange(c_dims, start, stop, step);
        if (ptr == null) return error.AllocationFailed;
        arr.ptr = ptr;
        return arr;
    }

    /// Creates a new ndarray with linearly spaced values.
    /// 
    /// Values are evenly distributed between start and stop (both inclusive).
    /// 
    /// **Parameters:**
    /// - `dims`: Slice of dimension sizes (must have at least 2 elements)
    /// - `start`: Starting value (inclusive)
    /// - `stop`: Ending value (inclusive)
    /// - `num`: Number of values to generate
    /// 
    /// **Returns:** NDArray with linearly spaced values or error
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initLinspace(&.{2, 5}, 0.0, 1.0, 10);
    /// defer arr.deinit();
    /// ```
    pub fn initLinspace(dims: []const usize, start: f64, stop: f64, num: usize) !NDArray {
        var arr: NDArray = undefined;
        const c_dims = try zigDimsToCDims(dims);
        const ptr = c.ndarray_new_linspace(c_dims, start, stop, num);
        if (ptr == null) return error.AllocationFailed;
        arr.ptr = ptr;
        return arr;
    }

    /// Creates a copy of the array.
    /// 
    /// Allocates a new array with the same dimensions and copies all data.
    /// 
    /// **Returns:** New NDArray with copied data or error
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initOnes(&.{3, 4});
    /// defer arr.deinit();
    /// const copy = try arr.initCopy();
    /// defer copy.deinit();
    /// ```
    pub fn initCopy(self: NDArray) !NDArray {
        const ptr = c.ndarray_new_copy(self.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Gets the value at the specified position.
    /// 
    /// **Parameters:**
    /// - `pos`: Slice of indices for each dimension
    /// 
    /// **Returns:** The value at the specified position
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initOnes(&.{3, 4});
    /// defer arr.deinit();
    /// const val = arr.get(&.{1, 2});
    /// ```
    pub fn get(self: NDArray, pos: []const usize) f64 {
        var c_pos: [MAX_DIMS]usize = undefined;
        @memcpy(c_pos[0..pos.len], pos);
        return c.ndarray_get(self.ptr, &c_pos);
    }

    /// Sets the value at the specified position.
    /// 
    /// **Parameters:**
    /// - `pos`: Slice of indices for each dimension
    /// - `value`: The value to set
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.init(&.{3, 4});
    /// defer arr.deinit();
    /// arr.set(&.{0, 0}, 5.0);
    /// ```
    pub fn set(self: NDArray, pos: []const usize, value: f64) void {
        var c_pos: [MAX_DIMS]usize = undefined;
        @memcpy(c_pos[0..pos.len], pos);
        c.ndarray_set(self.ptr, &c_pos, value);
    }

    /// Sets values along a slice at a specific index on an axis.
    /// 
    /// For 2D: axis=0 sets a row, axis=1 sets a column.
    /// For higher dimensions: sets the hyperplane perpendicular to the axis.
    /// Returns self for method chaining.
    /// 
    /// **Parameters:**
    /// - `axis`: The axis along which to set the slice
    /// - `index`: The index along the axis
    /// - `values`: Slice of values to set (size must match slice size)
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.init(&.{3, 4});
    /// defer arr.deinit();
    /// const row_data = [_]f64{1.0, 2.0, 3.0, 4.0};
    /// _ = arr.setSlice(0, 0, &row_data);
    /// ```
    pub fn setSlice(self: NDArray, axis: i32, index: usize, values: []const f64) NDArray {
        _ = c.ndarray_set_slice(self.ptr, axis, index, values.ptr);
        return self;
    }

    /// Fills a slice with a scalar value at a specific index on an axis.
    /// 
    /// For 2D: axis=0 fills a row, axis=1 fills a column.
    /// For higher dimensions: fills the hyperplane perpendicular to the axis.
    /// Returns self for method chaining.
    /// 
    /// **Parameters:**
    /// - `axis`: The axis along which to fill the slice
    /// - `index`: The index along the axis
    /// - `value`: The scalar value to fill with
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.init(&.{3, 4});
    /// defer arr.deinit();
    /// _ = arr.fillSlice(0, 1, 5.0); // Fill second row with 5.0
    /// ```
    pub fn fillSlice(self: NDArray, axis: i32, index: usize, value: f64) NDArray {
        _ = c.ndarray_fill_slice(self.ptr, axis, index, value);
        return self;
    }

    /// Prints the array to stdout.
    /// 
    /// Automatically formats output based on dimensionality.
    /// 
    /// **Parameters:**
    /// - `name`: Optional name to display (null for no name)
    /// - `precision`: Number of decimal places
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initOnes(&.{3, 4});
    /// defer arr.deinit();
    /// arr.print("MyArray", 4);
    /// ```
    pub fn print(self: NDArray, name: ?[:0]const u8, precision: i32) void {
        const c_name = if (name) |n| n.ptr else null;
        c.ndarray_print(self.ptr, c_name, precision);
    }

    /// Performs element-wise addition (modifies self in place).
    /// 
    /// Computes self = self + other.
    /// Arrays must have identical dimensions.
    /// 
    /// **Parameters:**
    /// - `other`: The array to add
    /// 
    /// **Returns:** self (for method chaining)
    /// 
    /// **Example:**
    /// ```zig
    /// var a = try NDArray.initOnes(&.{3, 4});
    /// defer a.deinit();
    /// const b = try NDArray.initFull(&.{3, 4}, 2.0);
    /// defer b.deinit();
    /// _ = a.add(b); // a now contains 3.0
    /// ```
    pub fn add(self: NDArray, other: NDArray) NDArray {
        _ = c.ndarray_add(self.ptr, other.ptr);
        return self;
    }

    /// Performs element-wise multiplication (modifies self in place).
    /// 
    /// Computes self = self * other.
    /// Arrays must have identical dimensions.
    /// 
    /// **Parameters:**
    /// - `other`: The array to multiply with
    /// 
    /// **Returns:** self (for method chaining)
    /// 
    /// **Example:**
    /// ```zig
    /// var a = try NDArray.initFull(&.{3, 4}, 3.0);
    /// defer a.deinit();
    /// const b = try NDArray.initFull(&.{3, 4}, 2.0);
    /// defer b.deinit();
    /// _ = a.mul(b); // a now contains 6.0
    /// ```
    pub fn mul(self: NDArray, other: NDArray) NDArray {
        _ = c.ndarray_mul(self.ptr, other.ptr);
        return self;
    }

    /// Adds a scalar to all elements (modifies self in place).
    /// 
    /// Computes self = self + scalar.
    /// 
    /// **Parameters:**
    /// - `scalar`: The value to add to each element
    /// 
    /// **Returns:** self (for method chaining)
    /// 
    /// **Example:**
    /// ```zig
    /// var arr = try NDArray.initOnes(&.{3, 4});
    /// defer arr.deinit();
    /// _ = arr.addScalar(5.0); // All elements become 6.0
    /// ```
    pub fn addScalar(self: NDArray, scalar: f64) NDArray {
        _ = c.ndarray_add_scalar(self.ptr, scalar);
        return self;
    }

    /// Multiplies all elements by a scalar (modifies self in place).
    /// 
    /// Computes self = self * scalar.
    /// 
    /// **Parameters:**
    /// - `scalar`: The value to multiply each element by
    /// 
    /// **Returns:** self (for method chaining)
    /// 
    /// **Example:**
    /// ```zig
    /// var arr = try NDArray.initFull(&.{3, 4}, 2.0);
    /// defer arr.deinit();
    /// _ = arr.mulScalar(3.0); // All elements become 6.0
    /// ```
    pub fn mulScalar(self: NDArray, scalar: f64) NDArray {
        _ = c.ndarray_mul_scalar(self.ptr, scalar);
        return self;
    }

    /// Linear combination: self = alpha*self + beta*other.
    /// 
    /// Computes a scaled linear combination and stores result in self.
    /// Arrays must have identical dimensions.
    /// 
    /// **Parameters:**
    /// - `alpha`: Scaling factor for self
    /// - `other`: The second array
    /// - `beta`: Scaling factor for other
    /// 
    /// **Returns:** self (for method chaining)
    /// 
    /// **Example:**
    /// ```zig
    /// var a = try NDArray.initOnes(&.{3, 4});
    /// defer a.deinit();
    /// const b = try NDArray.initFull(&.{3, 4}, 2.0);
    /// defer b.deinit();
    /// _ = a.axpby(2.0, b, 3.0); // a = 2*a + 3*b = 8.0
    /// ```
    pub fn axpby(self: NDArray, alpha: f64, other: NDArray, beta: f64) NDArray {
        _ = c.ndarray_axpby(self.ptr, alpha, other.ptr, beta);
        return self;
    }

    /// Scale and shift: self = alpha*self + beta.
    /// 
    /// Efficiently computes an affine transformation.
    /// 
    /// **Parameters:**
    /// - `alpha`: Scaling factor
    /// - `beta`: Shift value
    /// 
    /// **Returns:** self (for method chaining)
    /// 
    /// **Example:**
    /// ```zig
    /// var arr = try NDArray.initOnes(&.{3, 4});
    /// defer arr.deinit();
    /// _ = arr.scaleShift(2.0, 3.0); // arr = 2*arr + 3 = 5.0
    /// ```
    pub fn scaleShift(self: NDArray, alpha: f64, beta: f64) NDArray {
        _ = c.ndarray_scale_shift(self.ptr, alpha, beta);
        return self;
    }

    /// Element-wise multiply then scale: self = self * other * scalar.
    /// 
    /// Computes element-wise product then scales.
    /// Arrays must have identical dimensions.
    /// 
    /// **Parameters:**
    /// - `other`: The array to multiply with
    /// - `scalar`: Scaling factor
    /// 
    /// **Returns:** self (for method chaining)
    /// 
    /// **Example:**
    /// ```zig
    /// var a = try NDArray.initFull(&.{3, 4}, 2.0);
    /// defer a.deinit();
    /// const b = try NDArray.initFull(&.{3, 4}, 3.0);
    /// defer b.deinit();
    /// _ = a.mulScaled(b, 2.0); // a = 2 * a * b = 12.0
    /// ```
    pub fn mulScaled(self: NDArray, other: NDArray, scalar: f64) NDArray {
        _ = c.ndarray_mul_scaled(self.ptr, other.ptr, scalar);
        return self;
    }

    /// Applies a function to each element in place.
    /// 
    /// Computes self = func(self).
    /// Function must have C calling convention.
    /// 
    /// **Parameters:**
    /// - `func`: Function pointer with signature `fn(f64) f64`
    /// 
    /// **Returns:** self (for method chaining)
    /// 
    /// **Example:**
    /// ```zig
    /// const std = @import("std");
    /// fn square(x: f64) callconv(.c) f64 { return x * x; }
    /// 
    /// var arr = try NDArray.initFull(&.{3, 4}, 2.0);
    /// defer arr.deinit();
    /// _ = arr.mapFn(square); // All elements become 4.0
    /// ```
    pub fn mapFn(self: NDArray, func: *const fn (f64) callconv(.c) f64) NDArray {
        _ = c.ndarray_mapfnc(self.ptr, func);
        return self;
    }

    /// Map function then multiply: self = func(self) * other * alpha.
    /// 
    /// Applies function element-wise to self, then multiplies by other and scales.
    /// Arrays must have identical dimensions.
    /// 
    /// **Parameters:**
    /// - `func`: Function pointer with signature `fn(f64) f64`
    /// - `other`: The array to multiply with
    /// - `alpha`: Scaling factor
    /// 
    /// **Example:**
    /// ```zig
    /// const std = @import("std");
    /// fn square(x: f64) callconv(.c) f64 { return x * x; }
    /// 
    /// var a = try NDArray.initFull(&.{3, 4}, 2.0);
    /// defer a.deinit();
    /// const b = try NDArray.initFull(&.{3, 4}, 3.0);
    /// defer b.deinit();
    /// a.mapMul(square, b, 0.5); // a = square(a) * b * 0.5 = 6.0
    /// ```
    pub fn mapMul(self: NDArray, func: *const fn (f64) callconv(.c) f64, other: NDArray, alpha: f64) void {
        _ = c.ndarray_map_mul(self.ptr, func, other.ptr, alpha);
    }

    /// Fused multiply-add: dest = alpha * (self * other) + beta * dest.
    /// 
    /// Element-wise multiply self and other, scale by alpha, add to beta*dest.
    /// All arrays must have identical dimensions.
    /// 
    /// **Parameters:**
    /// - `other`: The second source array
    /// - `dest`: The destination array (modified in place)
    /// - `alpha`: Scaling factor for self*other
    /// - `beta`: Scaling factor for existing dest values
    /// 
    /// **Example:**
    /// ```zig
    /// const a = try NDArray.initFull(&.{3, 4}, 2.0);
    /// defer a.deinit();
    /// const b = try NDArray.initFull(&.{3, 4}, 3.0);
    /// defer b.deinit();
    /// var c = try NDArray.initOnes(&.{3, 4});
    /// defer c.deinit();
    /// a.mulAdd(b, c, 1.0, 1.0); // c = c + a*b = 7.0
    /// ```
    pub fn mulAdd(self: NDArray, other: NDArray, dest: NDArray, alpha: f64, beta: f64) void {
        _ = c.ndarray_mul_add(self.ptr, other.ptr, dest.ptr, alpha, beta);
    }

    /// Matrix-vector multiply: y = alpha * self * x + beta * y.
    /// 
    /// Computes matrix-vector product using optimized BLAS.
    /// self must be 2D, x and y must be vectors.
    /// 
    /// **Parameters:**
    /// - `x`: Input vector
    /// - `alpha`: Scaling factor for self*x
    /// - `beta`: Scaling factor for existing y values
    /// - `y`: Output vector (modified in place)
    /// 
    /// **Example:**
    /// ```zig
    /// const a = try NDArray.init(&.{3, 4});
    /// defer a.deinit();
    /// const x = try NDArray.init(&.{4, 1});
    /// defer x.deinit();
    /// var y = try NDArray.init(&.{3, 1});
    /// defer y.deinit();
    /// a.matvecMul(y, x, 1.0, 0.0); // y = A * x
    /// ```
    // pub fn matvecMul(self: NDArray, y: NDArray, x: NDArray, alpha: f64, beta: f64) NDArray {
    //     return NDArray{ .ptr = c.ndarray_matvec_mul(y.ptr, self.ptr, x.ptr, alpha, beta) };
    // }

    /// Clips values below minimum threshold (modifies self in place).
    /// 
    /// All elements less than min_val are set to min_val.
    /// 
    /// **Parameters:**
    /// - `min_val`: Minimum value threshold
    /// 
    /// **Returns:** self (for method chaining)
    /// 
    /// **Example:**
    /// ```zig
    /// var arr = try NDArray.initRandomUniform(&.{3, 4}, -2.0, 2.0);
    /// defer arr.deinit();
    /// _ = arr.clipMin(0.0); // Non-negativity constraint (ReLU)
    /// ```
    pub fn clipMin(self: NDArray, min_val: f64) NDArray {
        _ = c.ndarray_clip_min(self.ptr, min_val);
        return self;
    }

    /// Clips values above maximum threshold (modifies self in place).
    /// 
    /// All elements greater than max_val are set to max_val.
    /// 
    /// **Parameters:**
    /// - `max_val`: Maximum value threshold
    /// 
    /// **Returns:** self (for method chaining)
    /// 
    /// **Example:**
    /// ```zig
    /// var arr = try NDArray.initRandomUniform(&.{3, 4}, -2.0, 2.0);
    /// defer arr.deinit();
    /// _ = arr.clipMax(1.0); // Cap values at 1.0
    /// ```
    pub fn clipMax(self: NDArray, max_val: f64) NDArray {
        _ = c.ndarray_clip_max(self.ptr, max_val);
        return self;
    }

    /// Clips values to range [min_val, max_val] (modifies self in place).
    /// 
    /// All elements are constrained to the specified range.
    /// 
    /// **Parameters:**
    /// - `min_val`: Minimum value threshold
    /// - `max_val`: Maximum value threshold
    /// 
    /// **Example:**
    /// ```zig
    /// var arr = try NDArray.initRandomUniform(&.{3, 4}, -2.0, 2.0);
    /// defer arr.deinit();
    /// arr.clip(0.0, 1.0); // Clip to [0, 1] range
    /// ```
    pub fn clip(self: NDArray, min_val: f64, max_val: f64) void {
        _ = c.ndarray_clip(self.ptr, min_val, max_val);
    }

    /// Computes absolute value (modifies self in place).
    /// 
    /// Computes self = |self| element-wise.
    /// 
    /// **Example:**
    /// ```zig
    /// var arr = try NDArray.initRandomUniform(&.{3, 4}, -2.0, 2.0);
    /// defer arr.deinit();
    /// arr.abs(); // All values become non-negative
    /// ```
    pub fn abs(self: NDArray) void {
        _ = c.ndarray_abs(self.ptr);
    }

    /// Computes sign function (modifies self in place).
    /// 
    /// Returns -1 for negative values, 0 for zero, +1 for positive values.
    /// 
    /// **Example:**
    /// ```zig
    /// var arr = try NDArray.initRandomUniform(&.{3, 4}, -2.0, 2.0);
    /// defer arr.deinit();
    /// arr.sign(); // Extract sign information: -1, 0, or 1
    /// ```
    pub fn sign(self: NDArray) void {
        _ = c.ndarray_sign(self.ptr);
    }

    /// Gets a pointer to a slice along an axis.
    /// 
    /// Returns pointer valid as long as array exists. User must respect bounds.
    /// 
    /// **Parameters:**
    /// - `axis`: The axis along which to get the slice
    /// - `index`: The index along the axis
    /// 
    /// **Returns:** Pointer to the start of the slice data
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.init(&.{3, 4});
    /// defer arr.deinit();
    /// const row_ptr = arr.getSlicePtr(0, 1); // Pointer to row 1
    /// row_ptr[0] = 1.0; // Modify first element of row 1
    /// ```
    pub fn getSlicePtr(self: NDArray, axis: i32, index: usize) [*]f64 {
        return c.ndarray_get_slice_ptr(self.ptr, axis, index);
    }

    /// Copies a slice from one array to another.
    /// 
    /// The slice sizes must match.
    /// Returns self (destination) for method chaining.
    /// 
    /// **Parameters:**
    /// - `self_axis`: Axis in destination (self)
    /// - `self_idx`: Index along destination axis
    /// - `src`: Source array
    /// - `src_axis`: Axis in source
    /// - `src_idx`: Index along source axis
    /// 
    /// **Example:**
    /// ```zig
    /// const a = try NDArray.init(&.{3, 4});
    /// defer a.deinit();
    /// var b = try NDArray.init(&.{3, 4});
    /// defer b.deinit();
    /// _ = b.copySlice(0, 2, a, 0, 0); // Copy row 0 from a to row 2 of b
    /// ```
    pub fn copySlice(self: NDArray, self_axis: i32, self_idx: usize, src: NDArray, src_axis: i32, src_idx: usize) NDArray {
        _ = c.ndarray_copy_slice(self.ptr, self_axis, self_idx, src.ptr, src_axis, src_idx);
        return self;
    }

    /// Gets the size of a slice along an axis.
    /// 
    /// Returns the number of elements in a slice perpendicular to the given axis.
    /// 
    /// **Parameters:**
    /// - `axis`: The axis
    /// 
    /// **Returns:** Number of elements in a slice
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.init(&.{3, 4});
    /// defer arr.deinit();
    /// const row_size = arr.getSliceSize(0); // Returns 4
    /// const col_size = arr.getSliceSize(1); // Returns 3
    /// ```
    pub fn getSliceSize(self: NDArray, axis: i32) usize {
        return c.ndarray_get_slice_size(self.ptr, axis);
    }

    /// Creates element-wise equality comparison array.
    /// 
    /// Returns 1.0 where elements are equal, 0.0 otherwise.
    /// Arrays must have identical dimensions.
    /// 
    /// **Parameters:**
    /// - `other`: Array to compare with
    /// 
    /// **Returns:** New array with comparison results or error
    /// 
    /// **Example:**
    /// ```zig
    /// const a = try NDArray.initOnes(&.{3, 4});
    /// defer a.deinit();
    /// const b = try NDArray.initOnes(&.{3, 4});
    /// defer b.deinit();
    /// const result = try a.initEqual(b); // All 1.0
    /// defer result.deinit();
    /// ```
    pub fn initEqual(self: NDArray, other: NDArray) !NDArray {
        const ptr = c.ndarray_new_equal(self.ptr, other.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Creates element-wise less-than comparison array.
    /// 
    /// Returns 1.0 where self < other, 0.0 otherwise.
    /// Arrays must have identical dimensions.
    /// 
    /// **Parameters:**
    /// - `other`: Array to compare with
    /// 
    /// **Returns:** New array with comparison results or error
    /// 
    /// **Example:**
    /// ```zig
    /// const a = try NDArray.initOnes(&.{3, 4});
    /// defer a.deinit();
    /// const b = try NDArray.initFull(&.{3, 4}, 2.0);
    /// defer b.deinit();
    /// const result = try a.initLess(b); // All 1.0 (a < b)
    /// defer result.deinit();
    /// ```
    pub fn initLess(self: NDArray, other: NDArray) !NDArray {
        const ptr = c.ndarray_new_less(self.ptr, other.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Creates element-wise greater-than comparison array.
    /// 
    /// Returns 1.0 where self > other, 0.0 otherwise.
    /// Arrays must have identical dimensions.
    /// 
    /// **Parameters:**
    /// - `other`: Array to compare with
    /// 
    /// **Returns:** New array with comparison results or error
    /// 
    /// **Example:**
    /// ```zig
    /// const a = try NDArray.initFull(&.{3, 4}, 2.0);
    /// defer a.deinit();
    /// const b = try NDArray.initOnes(&.{3, 4});
    /// defer b.deinit();
    /// const result = try a.initGreater(b); // All 1.0 (a > b)
    /// defer result.deinit();
    /// ```
    pub fn initGreater(self: NDArray, other: NDArray) !NDArray {
        const ptr = c.ndarray_new_greater(self.ptr, other.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Creates scalar equality comparison array.
    /// 
    /// Returns 1.0 where elements equal value, 0.0 otherwise.
    /// 
    /// **Parameters:**
    /// - `value`: Scalar value to compare against
    /// 
    /// **Returns:** New array with comparison results or error
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initZeros(&.{3, 4});
    /// defer arr.deinit();
    /// const mask = try arr.initEqualScalar(0.0); // Find all zeros
    /// defer mask.deinit();
    /// ```
    pub fn initEqualScalar(self: NDArray, value: f64) !NDArray {
        const ptr = c.ndarray_new_equal_scalar(self.ptr, value);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Creates scalar less-than comparison array.
    /// 
    /// Returns 1.0 where elements are less than value, 0.0 otherwise.
    /// 
    /// **Parameters:**
    /// - `value`: Scalar value to compare against
    /// 
    /// **Returns:** New array with comparison results or error
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initRandomUniform(&.{3, 4}, 0.0, 10.0);
    /// defer arr.deinit();
    /// const mask = try arr.initLessScalar(5.0); // Find elements < 5.0
    /// defer mask.deinit();
    /// ```
    pub fn initLessScalar(self: NDArray, value: f64) !NDArray {
        const ptr = c.ndarray_new_less_scalar(self.ptr, value);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Creates scalar greater-than comparison array.
    /// 
    /// Returns 1.0 where elements are greater than value, 0.0 otherwise.
    /// 
    /// **Parameters:**
    /// - `value`: Scalar value to compare against
    /// 
    /// **Returns:** New array with comparison results or error
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initRandomUniform(&.{3, 4}, 0.0, 10.0);
    /// defer arr.deinit();
    /// const mask = try arr.initGreaterScalar(5.0); // Find elements > 5.0
    /// defer mask.deinit();
    /// ```
    pub fn initGreaterScalar(self: NDArray, value: f64) !NDArray {
        const ptr = c.ndarray_new_greater_scalar(self.ptr, value);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Creates logical AND array.
    /// 
    /// Returns 1.0 where both elements are non-zero, 0.0 otherwise.
    /// Arrays must have identical dimensions.
    /// 
    /// **Parameters:**
    /// - `other`: Second array (treated as boolean)
    /// 
    /// **Returns:** New array with logical AND results or error
    /// 
    /// **Example:**
    /// ```zig
    /// const data = try NDArray.initRandomUniform(&.{3, 4}, 0.0, 10.0);
    /// defer data.deinit();
    /// const a = try data.initGreaterScalar(0.0);
    /// defer a.deinit();
    /// const b = try data.initLessScalar(10.0);
    /// defer b.deinit();
    /// const mask = try a.initLogicalAnd(b); // 0 < data < 10
    /// defer mask.deinit();
    /// ```
    pub fn initLogicalAnd(self: NDArray, other: NDArray) !NDArray {
        const ptr = c.ndarray_new_logical_and(self.ptr, other.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Creates logical OR array.
    /// 
    /// Returns 1.0 where at least one element is non-zero, 0.0 otherwise.
    /// Arrays must have identical dimensions.
    /// 
    /// **Parameters:**
    /// - `other`: Second array (treated as boolean)
    /// 
    /// **Returns:** New array with logical OR results or error
    /// 
    /// **Example:**
    /// ```zig
    /// const data = try NDArray.initRandomUniform(&.{3, 4}, 0.0, 10.0);
    /// defer data.deinit();
    /// const a = try data.initLessScalar(0.0);
    /// defer a.deinit();
    /// const b = try data.initGreaterScalar(10.0);
    /// defer b.deinit();
    /// const mask = try a.initLogicalOr(b); // data < 0 OR data > 10
    /// defer mask.deinit();
    /// ```
    pub fn initLogicalOr(self: NDArray, other: NDArray) !NDArray {
        const ptr = c.ndarray_new_logical_or(self.ptr, other.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Creates logical NOT array.
    /// 
    /// Returns 1.0 where element is zero, 0.0 where non-zero.
    /// 
    /// **Returns:** New array with logical NOT results or error
    /// 
    /// **Example:**
    /// ```zig
    /// const data = try NDArray.initZeros(&.{3, 4});
    /// defer data.deinit();
    /// const mask = try data.initEqualScalar(0.0);
    /// defer mask.deinit();
    /// const not_mask = try mask.initLogicalNot(); // Invert mask
    /// defer not_mask.deinit();
    /// ```
    pub fn initLogicalNot(self: NDArray) !NDArray {
        const ptr = c.ndarray_new_logical_not(self.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Creates ternary selection array: result = condition ? x : y.
    /// 
    /// NumPy-style where function. Selects elements from x or y based on condition.
    /// All arrays must have identical dimensions.
    /// 
    /// **Parameters:**
    /// - `condition`: Boolean array (non-zero = true)
    /// - `x`: Array to select from when condition is true
    /// - `y`: Array to select from when condition is false
    /// 
    /// **Returns:** New array with selected values or error
    /// 
    /// **Example:**
    /// ```zig
    /// const a = try NDArray.initRandomUniform(&.{3, 4}, -2.0, 2.0);
    /// defer a.deinit();
    /// const positive = try a.initGreaterScalar(0.0);
    /// defer positive.deinit();
    /// const zeros = try NDArray.initZeros(&.{3, 4});
    /// defer zeros.deinit();
    /// const result = try NDArray.initWhere(positive, a, zeros); // Keep only positive
    /// defer result.deinit();
    /// ```
    pub fn initWhere(condition: NDArray, x: NDArray, y: NDArray) !NDArray {
        const ptr = c.ndarray_new_where(condition.ptr, x.ptr, y.ptr);
        if (ptr == null) return error.AllocationFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Creates matrix multiplication result.
    /// 
    /// Operates on the last two dimensions and broadcasts over leading dimensions.
    /// Uses cache-optimized blocked algorithm.
    /// 
    /// **Parameters:**
    /// - `other`: The second array
    /// 
    /// **Returns:** New array with multiplication result or error
    /// 
    /// **Example:**
    /// ```zig
    /// const a = try NDArray.init(&.{3, 4});
    /// defer a.deinit();
    /// const b = try NDArray.init(&.{4, 5});
    /// defer b.deinit();
    /// const c = try a.initMatmul(b); // Shape: [3, 5]
    /// defer c.deinit();
    /// ```
    pub fn initMatmul(self: NDArray, other: NDArray) !NDArray {
        const ptr = c.ndarray_new_matmul(self.ptr, other.ptr);
        if (ptr == null) return error.MatmulFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Creates tensor contraction result over specified axes.
    /// 
    /// Contracts specified axes between two tensors using Einstein summation convention.
    /// For standard matrix multiplication, use `initMatmul` which is optimized.
    /// 
    /// **Parameters:**
    /// - `other`: The second tensor
    /// - `axes_a`: Slice of axes from self to contract
    /// - `axes_b`: Slice of axes from other to contract (must match axes_a length)
    /// 
    /// **Returns:** New tensor with contracted dimensions or error
    /// 
    /// **Example:**
    /// ```zig
    /// const a = try NDArray.init(&.{2, 3, 4});
    /// defer a.deinit();
    /// const b = try NDArray.init(&.{4, 5});
    /// defer b.deinit();
    /// // Single axis contraction: A[i,j,k] * B[k,l] -> C[i,j,l]
    /// const c = try a.initTensorDot(b, &.{2}, &.{0});
    /// defer c.deinit();
    /// ```
    pub fn initTensorDot(self: NDArray, other: NDArray, axes_a: []const i32, axes_b: []const i32) !NDArray {
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

    /// Creates a transposed copy of the array.
    /// 
    /// For N-dimensional arrays, reverses all axes.
    /// For example, shape [2,3,4] becomes [4,3,2].
    /// 
    /// **Returns:** New transposed array or error
    /// 
    /// **Example:**
    /// ```zig
    /// const a = try NDArray.init(&.{2, 3, 4});
    /// defer a.deinit();
    /// const t = try a.initTranspose(); // Shape: [4, 3, 2]
    /// defer t.deinit();
    /// ```
    pub fn initTranspose(self: NDArray) !NDArray {
        const ptr = c.ndarray_new_transpose(self.ptr);
        if (ptr == null) return error.TransposeFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Reshapes the array in-place to new dimensions.
    /// 
    /// The total number of elements must remain the same.
    /// Data remains in row-major (C-order) layout.
    /// Use -1 for one dimension to automatically infer its size.
    /// 
    /// **Parameters:**
    /// - `new_dims`: Slice of new dimension sizes
    /// 
    /// **Example:**
    /// ```zig
    /// var arr = try NDArray.init(&.{2, 6});
    /// defer arr.deinit();
    /// try arr.reshape(&.{3, 4}); // Shape becomes [3, 4]
    /// try arr.reshape(&.{2, 2, 3}); // Shape becomes [2, 2, 3]
    /// ```
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

    /// Creates a subregion from the array.
    /// 
    /// Extracts elements from start to end indices (end is exclusive).
    /// 
    /// **Parameters:**
    /// - `axis`: Axis along which to take the subregion
    /// - `start`: Starting index (inclusive)
    /// - `end`: Ending index (exclusive)
    /// 
    /// **Returns:** New array with copied subregion or error
    /// 
    /// **Example:**
    /// ```zig
    /// const a = try NDArray.init(&.{4, 5});
    /// defer a.deinit();
    /// const rows = try a.initTake(0, 1, 3); // Extract rows 1 and 2: [2,5]
    /// defer rows.deinit();
    /// const cols = try a.initTake(1, 2, 5); // Extract columns 2,3,4: [4,3]
    /// defer cols.deinit();
    /// ```
    pub fn initTake(self: NDArray, axis: i32, start: usize, end: usize) !NDArray {
        const ptr = c.ndarray_new_take(self.ptr, axis, start, end);
        if (ptr == null) return error.TakeFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Saves the array to a binary file.
    /// 
    /// Creates a binary file with custom format including magic number,
    /// version, dimensions, and data in row-major order.
    /// 
    /// **Parameters:**
    /// - `filename`: Path to the output file (use .bin extension)
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initRandomUniform(&.{3, 4}, 0.0, 1.0);
    /// defer arr.deinit();
    /// try arr.save("mydata.bin");
    /// ```
    pub fn save(self: NDArray, filename: [:0]const u8) !void {
        const result = c.ndarray_save(self.ptr, filename.ptr);
        if (result != 0) return error.SaveFailed;
    }

    /// Loads an array from a binary file.
    /// 
    /// Reads the file format created by `save()`.
    /// 
    /// **Parameters:**
    /// - `filename`: Path to the input file (.bin extension)
    /// 
    /// **Returns:** New array loaded from file or error
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initLoad("mydata.bin");
    /// defer arr.deinit();
    /// arr.print("Loaded", 4);
    /// ```
    pub fn initLoad(filename: [:0]const u8) !NDArray {
        const ptr = c.ndarray_new_load(filename.ptr);
        if (ptr == null) return error.LoadFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Gets the number of dimensions.
    /// 
    /// **Returns:** Number of dimensions
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.init(&.{2, 3, 4});
    /// defer arr.deinit();
    /// const n = arr.ndim(); // Returns 3
    /// ```
    pub fn ndim(self: NDArray) usize {
        return self.ptr.*.ndim;
    }

    /// Gets the dimension sizes.
    /// 
    /// Allocates and returns a slice containing the size of each dimension.
    /// Caller is responsible for freeing the returned slice.
    /// 
    /// **Parameters:**
    /// - `allocator`: Allocator to use for the returned slice
    /// 
    /// **Returns:** Slice of dimension sizes or error
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.init(&.{2, 3, 4});
    /// defer arr.deinit();
    /// const dims = try arr.shape(std.testing.allocator);
    /// defer std.testing.allocator.free(dims);
    /// // dims is &.{2, 3, 4}
    /// ```
    pub fn shape(self: NDArray, allocator: std.mem.Allocator) ![]usize {
        const dims = try allocator.alloc(usize, self.ndim());
        for (0..self.ndim()) |i| {
            dims[i] = self.ptr.*.dims[i];
        }
        return dims;
    }

    /// Creates aggregation result over an axis.
    /// 
    /// Result maintains ndim >= 2 constraint by adding dimension of 1 if needed.
    /// Use -1 for axis to aggregate over all axes (returns [1, 1] shape).
    /// 
    /// **Parameters:**
    /// - `axis`: Axis to aggregate over (0 to ndim-1, or -1 for all axes)
    /// - `aggr_type`: Type of aggregation (sum, mean, max, min, std)
    /// 
    /// **Returns:** New array with aggregation result or error
    /// 
    /// **Example:**
    /// ```zig
    /// const a = try NDArray.init(&.{3, 4});
    /// defer a.deinit();
    /// const sum0 = try a.initAggregate(0, .sum); // Sum along axis 0: [1, 4]
    /// defer sum0.deinit();
    /// const mean_all = try a.initAggregate(-1, .mean); // Mean of all: [1, 1]
    /// defer mean_all.deinit();
    /// ```
    pub fn initAggregate(self: NDArray, axis: i32, aggr_type: AggrType) !NDArray {
        const ptr = c.ndarray_new_aggr(self.ptr, axis, @intFromEnum(aggr_type));
        if (ptr == null) return error.AggregateFailed;
        return NDArray{ .ptr = ptr };
    }

    /// Aggregates all elements to a scalar value.
    /// 
    /// **Parameters:**
    /// - `aggr_type`: Type of aggregation (sum, mean, max, min, std)
    /// 
    /// **Returns:** Scalar aggregation result
    /// 
    /// **Example:**
    /// ```zig
    /// const arr = try NDArray.initRandomUniform(&.{3, 4}, 0.0, 10.0);
    /// defer arr.deinit();
    /// const sum = arr.scalarAggregate(.sum);
    /// const mean = arr.scalarAggregate(.mean);
    /// const max_val = arr.scalarAggregate(.max);
    /// ```
    pub fn scalarAggregate(self: NDArray, aggr_type: AggrType) f64 {
        return c.ndarray_scalar_aggr(self.ptr, @intFromEnum(aggr_type));
    }

    /// Stacks arrays along a new axis.
    /// 
    /// Creates a new dimension at the specified position.
    /// All arrays must have the same shape.
    /// 
    /// **Parameters:**
    /// - `axis`: Position for the new dimension (0 to ndim)
    /// - `arrays`: Slice of arrays to stack
    /// 
    /// **Returns:** New stacked array or error
    /// 
    /// **Example:**
    /// ```zig
    /// const a = try NDArray.init(&.{2, 3});
    /// defer a.deinit();
    /// const b = try NDArray.init(&.{2, 3});
    /// defer b.deinit();
    /// const c = try NDArray.init(&.{2, 3});
    /// defer c.deinit();
    /// const stacked = try NDArray.initStack(0, &.{a, b, c}); // Shape: [3, 2, 3]
    /// defer stacked.deinit();
    /// ```
    pub fn initStack(axis: i32, arrays: []const NDArray) !NDArray {
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

    /// Concatenates arrays along an existing axis.
    /// 
    /// All dimensions except the concatenation axis must match.
    /// 
    /// **Parameters:**
    /// - `axis`: Axis along which to concatenate (0 to ndim-1)
    /// - `arrays`: Slice of arrays to concatenate
    /// 
    /// **Returns:** New concatenated array or error
    /// 
    /// **Example:**
    /// ```zig
    /// const a = try NDArray.init(&.{2, 3, 4});
    /// defer a.deinit();
    /// const b = try NDArray.init(&.{2, 5, 4});
    /// defer b.deinit();
    /// const concatenated = try NDArray.initConcat(1, &.{a, b}); // Shape: [2, 8, 4]
    /// defer concatenated.deinit();
    /// ```
    pub fn initConcat(axis: i32, arrays: []const NDArray) !NDArray {
        if (arrays.len > MAX_ARRAYS) return error.TooManyArrays;
        var c_array_list: [MAX_ARRAYS + 1]c.NDArray = undefined;
        for (arrays, 0..) |arr, i| {
            c_array_list[i] = arr.ptr;
        }
        c_array_list[arrays.len] = null;
        const ptr = c.ndarray_new_concat(axis, &c_array_list);
        if (ptr == null) return error.ConcatFailed;
        return NDArray{ .ptr = ptr };
    }
};

test "init and deinit" {
    const arr = try NDArray.init(&.{ 2, 3 });
    defer arr.deinit();
    try std.testing.expect(arr.ndim() == 2);
}

test "zeros" {
    const arr = try NDArray.initZeros(&.{ 2, 3 });
    defer arr.deinit();
    try std.testing.expectEqual(@as(f64, 0.0), arr.get(&.{ 0, 0 }));
}

test "ones" {
    const arr = try NDArray.initOnes(&.{ 2, 3 });
    defer arr.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), arr.get(&.{ 1, 2 }));
}

test "full" {
    const arr = try NDArray.initFull(&.{ 2, 2 }, 5.5);
    defer arr.deinit();
    try std.testing.expectEqual(@as(f64, 5.5), arr.get(&.{ 0, 0 }));
}

test "fromData" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const arr = try NDArray.initFromData(&.{ 2, 2 }, &data);
    defer arr.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), arr.get(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 4.0), arr.get(&.{ 1, 1 }));
}

test "randomUniform" {
    const arr = try NDArray.initRandomUniform(&.{ 2, 3 }, 0.0, 1.0);
    defer arr.deinit();
    const val = arr.get(&.{ 0, 0 });
    try std.testing.expect(val >= 0.0 and val <= 1.0);
}

test "randomNormal" {
    const arr = try NDArray.initRandomNormal(&.{ 2, 3 }, 0.0, 1.0);
    defer arr.deinit();
    try std.testing.expect(arr.ndim() == 2);
}

test "randomPoisson" {
    const arr = try NDArray.initRandomPoisson(&.{ 2, 3 }, 5.0);
    defer arr.deinit();
    try std.testing.expect(arr.ndim() == 2);
}

test "arange" {
    const arr = try NDArray.initArange(&.{ 2, 3 }, 0.0, 6.0, 1.0);
    defer arr.deinit();
    try std.testing.expectEqual(@as(f64, 0.0), arr.get(&.{ 0, 0 }));
}

test "linspace" {
    const arr = try NDArray.initLinspace(&.{ 2, 3 }, 0.0, 5.0, 6);
    defer arr.deinit();
    try std.testing.expectEqual(@as(f64, 0.0), arr.get(&.{ 0, 0 }));
}

test "copy" {
    const arr = try NDArray.initOnes(&.{ 2, 2 });
    defer arr.deinit();
    const copy = try arr.initCopy();
    defer copy.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), copy.get(&.{ 0, 0 }));
}

test "get and set" {
    const arr = try NDArray.initZeros(&.{ 2, 2 });
    defer arr.deinit();
    arr.set(&.{ 1, 1 }, 7.5);
    try std.testing.expectEqual(@as(f64, 7.5), arr.get(&.{ 1, 1 }));
}

test "setSlice" {
    const arr = try NDArray.initZeros(&.{ 2, 3 });
    defer arr.deinit();
    const values = [_]f64{ 1.0, 2.0, 3.0 };
    _ = arr.setSlice(0, 0, &values);
    try std.testing.expectEqual(@as(f64, 1.0), arr.get(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 2.0), arr.get(&.{ 0, 1 }));
    try std.testing.expectEqual(@as(f64, 3.0), arr.get(&.{ 0, 2 }));
}

test "fillSlice" {
    const arr = try NDArray.initZeros(&.{ 2, 3 });
    defer arr.deinit();
    _ = arr.fillSlice(0, 1, 9.0);
    try std.testing.expectEqual(@as(f64, 9.0), arr.get(&.{ 1, 0 }));
    try std.testing.expectEqual(@as(f64, 9.0), arr.get(&.{ 1, 1 }));
    try std.testing.expectEqual(@as(f64, 9.0), arr.get(&.{ 1, 2 }));
}

test "setSlice and fillSlice chaining" {
    const arr = try NDArray.initZeros(&.{ 3, 4 });
    defer arr.deinit();
    const row_data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    _ = arr.setSlice(0, 0, &row_data).fillSlice(0, 1, 5.0);
    // Check first row was set
    try std.testing.expectEqual(@as(f64, 1.0), arr.get(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 4.0), arr.get(&.{ 0, 3 }));
    // Check second row was filled
    try std.testing.expectEqual(@as(f64, 5.0), arr.get(&.{ 1, 0 }));
    try std.testing.expectEqual(@as(f64, 5.0), arr.get(&.{ 1, 3 }));
}

test "print" {
    const arr = try NDArray.initOnes(&.{ 2, 2 });
    defer arr.deinit();
    arr.print(null, 2);
}

test "add" {
    const a = try NDArray.initOnes(&.{ 2, 2 });
    defer a.deinit();
    const b = try NDArray.initOnes(&.{ 2, 2 });
    defer b.deinit();
    _ = a.add(b);
    try std.testing.expectEqual(@as(f64, 2.0), a.get(&.{ 0, 0 }));
}

test "mul" {
    const a = try NDArray.initFull(&.{ 2, 2 }, 3.0);
    defer a.deinit();
    const b = try NDArray.initFull(&.{ 2, 2 }, 2.0);
    defer b.deinit();
    _ = a.mul(b);
    try std.testing.expectEqual(@as(f64, 6.0), a.get(&.{ 0, 0 }));
}

test "addScalar" {
    const arr = try NDArray.initOnes(&.{ 2, 2 });
    defer arr.deinit();
    _ = arr.addScalar(5.0);
    try std.testing.expectEqual(@as(f64, 6.0), arr.get(&.{ 0, 0 }));
}

test "mulScalar" {
    const arr = try NDArray.initFull(&.{ 2, 2 }, 3.0);
    defer arr.deinit();
    _ = arr.mulScalar(2.0);
    try std.testing.expectEqual(@as(f64, 6.0), arr.get(&.{ 0, 0 }));
}

test "axpby" {
    const a = try NDArray.initFull(&.{ 2, 2 }, 2.0);
    defer a.deinit();
    const b = try NDArray.initFull(&.{ 2, 2 }, 3.0);
    defer b.deinit();
    _ = a.axpby(2.0, b, 3.0);
    try std.testing.expectEqual(@as(f64, 13.0), a.get(&.{ 0, 0 }));
}

test "scaleShift" {
    const arr = try NDArray.initFull(&.{ 2, 2 }, 2.0);
    defer arr.deinit();
    _ = arr.scaleShift(3.0, 1.0);
    try std.testing.expectEqual(@as(f64, 7.0), arr.get(&.{ 0, 0 }));
}

test "mulScaled" {
    const a = try NDArray.initFull(&.{ 2, 2 }, 2.0);
    defer a.deinit();
    const b = try NDArray.initFull(&.{ 2, 2 }, 3.0);
    defer b.deinit();
    _ = a.mulScaled(b, 2.0);
    try std.testing.expectEqual(@as(f64, 12.0), a.get(&.{ 0, 0 }));
}

fn testFunc(x: f64) callconv(.c) f64 {
    return x * 2.0;
}

test "mapFn" {
    const arr = try NDArray.initFull(&.{ 2, 2 }, 3.0);
    defer arr.deinit();
    _ = arr.mapFn(&testFunc);
    try std.testing.expectEqual(@as(f64, 6.0), arr.get(&.{ 0, 0 }));
}

test "mapMul" {
    const a = try NDArray.initFull(&.{ 2, 2 }, 3.0);
    defer a.deinit();
    const b = try NDArray.initFull(&.{ 2, 2 }, 2.0);
    defer b.deinit();
    a.mapMul(&testFunc, b, 1.0);
    try std.testing.expectEqual(@as(f64, 12.0), a.get(&.{ 0, 0 }));
}

test "mulAdd" {
    const a = try NDArray.initFull(&.{ 2, 2 }, 2.0);
    defer a.deinit();
    const b = try NDArray.initFull(&.{ 2, 2 }, 3.0);
    defer b.deinit();
    const dest = try NDArray.initFull(&.{ 2, 2 }, 1.0);
    defer dest.deinit();
    a.mulAdd(b, dest, 1.0, 1.0);
    try std.testing.expectEqual(@as(f64, 7.0), dest.get(&.{ 0, 0 }));
}

// test "matvecMul" {
//     const a = try NDArray.initOnes(&.{ 2, 3 });
//     defer a.deinit();
//     const x = try NDArray.initOnes(&.{ 3, 1 });
//     defer x.deinit();
//     const y = try NDArray.initZeros(&.{ 2, 1 });
//     defer y.deinit();
//     _ = a.matvecMul(y, x, 1.0, 0.0);
//     try std.testing.expectEqual(@as(f64, 3.0), y.get(&.{ 0, 0 }));
// }

test "clipMin" {
    const arr = try NDArray.initFull(&.{ 2, 2 }, 1.0);
    defer arr.deinit();
    _ = arr.clipMin(2.0);
    try std.testing.expectEqual(@as(f64, 2.0), arr.get(&.{ 0, 0 }));
}

test "clipMax" {
    const arr = try NDArray.initFull(&.{ 2, 2 }, 5.0);
    defer arr.deinit();
    _ = arr.clipMax(3.0);
    try std.testing.expectEqual(@as(f64, 3.0), arr.get(&.{ 0, 0 }));
}

test "clip" {
    const arr = try NDArray.initFull(&.{ 2, 2 }, 5.0);
    defer arr.deinit();
    arr.clip(2.0, 3.0);
    try std.testing.expectEqual(@as(f64, 3.0), arr.get(&.{ 0, 0 }));
}

test "abs" {
    const arr = try NDArray.initFull(&.{ 2, 2 }, -5.0);
    defer arr.deinit();
    arr.abs();
    try std.testing.expectEqual(@as(f64, 5.0), arr.get(&.{ 0, 0 }));
}

test "sign" {
    const arr = try NDArray.initFull(&.{ 2, 2 }, -5.0);
    defer arr.deinit();
    arr.sign();
    try std.testing.expectEqual(@as(f64, -1.0), arr.get(&.{ 0, 0 }));
}

test "getSlicePtr" {
    const arr = try NDArray.initOnes(&.{ 2, 3 });
    defer arr.deinit();
    const ptr = arr.getSlicePtr(0, 0);
    try std.testing.expectEqual(@as(f64, 1.0), ptr[0]);
}

test "copySlice" {
    const src = try NDArray.initOnes(&.{ 2, 3 });
    defer src.deinit();
    const dst = try NDArray.initZeros(&.{ 2, 3 });
    defer dst.deinit();
    const result = dst.copySlice(0, 1, src, 0, 0);
    try std.testing.expect(result.ptr == dst.ptr);
    try std.testing.expectEqual(@as(f64, 1.0), dst.get(&.{ 1, 0 }));
}

test "getSliceSize" {
    const arr = try NDArray.initOnes(&.{ 2, 3 });
    defer arr.deinit();
    const size = arr.getSliceSize(0);
    try std.testing.expectEqual(@as(usize, 3), size);
}

test "equal" {
    const a = try NDArray.initOnes(&.{ 2, 2 });
    defer a.deinit();
    const b = try NDArray.initOnes(&.{ 2, 2 });
    defer b.deinit();
    const result = try a.initEqual(b);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&.{ 0, 0 }));
}

test "less" {
    const a = try NDArray.initOnes(&.{ 2, 2 });
    defer a.deinit();
    const b = try NDArray.initFull(&.{ 2, 2 }, 2.0);
    defer b.deinit();
    const result = try a.initLess(b);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&.{ 0, 0 }));
}

test "greater" {
    const a = try NDArray.initFull(&.{ 2, 2 }, 2.0);
    defer a.deinit();
    const b = try NDArray.initOnes(&.{ 2, 2 });
    defer b.deinit();
    const result = try a.initGreater(b);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&.{ 0, 0 }));
}

test "equalScalar" {
    const arr = try NDArray.initOnes(&.{ 2, 2 });
    defer arr.deinit();
    const result = try arr.initEqualScalar(1.0);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&.{ 0, 0 }));
}

test "lessScalar" {
    const arr = try NDArray.initOnes(&.{ 2, 2 });
    defer arr.deinit();
    const result = try arr.initLessScalar(2.0);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&.{ 0, 0 }));
}

test "greaterScalar" {
    const arr = try NDArray.initFull(&.{ 2, 2 }, 2.0);
    defer arr.deinit();
    const result = try arr.initGreaterScalar(1.0);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&.{ 0, 0 }));
}

test "logicalAnd" {
    const a = try NDArray.initOnes(&.{ 2, 2 });
    defer a.deinit();
    const b = try NDArray.initOnes(&.{ 2, 2 });
    defer b.deinit();
    const result = try a.initLogicalAnd(b);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&.{ 0, 0 }));
}

test "logicalOr" {
    const a = try NDArray.initZeros(&.{ 2, 2 });
    defer a.deinit();
    const b = try NDArray.initOnes(&.{ 2, 2 });
    defer b.deinit();
    const result = try a.initLogicalOr(b);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&.{ 0, 0 }));
}

test "logicalNot" {
    const arr = try NDArray.initZeros(&.{ 2, 2 });
    defer arr.deinit();
    const result = try arr.initLogicalNot();
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&.{ 0, 0 }));
}

test "where" {
    const condition = try NDArray.initOnes(&.{ 2, 2 });
    defer condition.deinit();
    const x = try NDArray.initFull(&.{ 2, 2 }, 10.0);
    defer x.deinit();
    const y = try NDArray.initFull(&.{ 2, 2 }, 20.0);
    defer y.deinit();
    const result = try NDArray.initWhere(condition, x, y);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 10.0), result.get(&.{ 0, 0 }));
}

test "matmul" {
    const a = try NDArray.initOnes(&.{ 2, 3 });
    defer a.deinit();
    const b = try NDArray.initOnes(&.{ 3, 2 });
    defer b.deinit();
    const result = try a.initMatmul(b);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 3.0), result.get(&.{ 0, 0 }));
}

test "tensordot" {
    const a = try NDArray.initOnes(&.{ 2, 3, 4 });
    defer a.deinit();
    const b = try NDArray.initOnes(&.{ 4, 5 });
    defer b.deinit();
    const axes_a = [_]i32{2};
    const axes_b = [_]i32{0};
    const result = try a.initTensorDot(b, &axes_a, &axes_b);
    defer result.deinit();
    try std.testing.expect(result.ndim() == 3);
}

test "transpose" {
    const arr = try NDArray.initOnes(&.{ 2, 3 });
    defer arr.deinit();
    const result = try arr.initTranspose();
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), result.get(&.{ 2, 1 }));
}

test "reshape" {
    const arr = try NDArray.initOnes(&.{ 2, 6 });
    defer arr.deinit();
    try arr.reshape(&[_]isize{ 3, 4 });
    try std.testing.expectEqual(@as(f64, 1.0), arr.get(&.{ 2, 3 }));
}

test "take" {
    const arr = try NDArray.initOnes(&.{ 3, 4 });
    defer arr.deinit();
    const result = try arr.initTake(0, 0, 2);
    defer result.deinit();
    try std.testing.expect(result.ndim() == 2);
}

test "save and load" {
    const arr = try NDArray.initFull(&.{ 2, 2 }, 42.0);
    defer arr.deinit();
    try arr.save("test_zig_saveload.bin");
    const loaded = try NDArray.initLoad("test_zig_saveload.bin");
    defer loaded.deinit();
    try std.testing.expectEqual(@as(f64, 42.0), loaded.get(&.{ 0, 0 }));
}

test "ndim and shape" {
    const arr = try NDArray.initOnes(&.{ 2, 3, 4 });
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
    const arr = try NDArray.initOnes(&.{ 2, 3 });
    defer arr.deinit();
    const result = try arr.initAggregate(0, AggrType.sum);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 2.0), result.get(&.{ 0, 0 }));
}

test "aggregate mean" {
    const arr = try NDArray.initFull(&.{ 2, 3 }, 4.0);
    defer arr.deinit();
    const result = try arr.initAggregate(1, AggrType.mean);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 4.0), result.get(&.{ 0, 0 }));
}

test "scalar aggregate sum" {
    const data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const arr = try NDArray.initFromData(&.{ 3, 4 }, &data);
    defer arr.deinit();
    const result = arr.scalarAggregate(AggrType.sum);
    try std.testing.expectEqual(@as(f64, 78.0), result);
}

test "scalar aggregate mean" {
    const data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const arr = try NDArray.initFromData(&.{ 3, 4 }, &data);
    defer arr.deinit();
    const result = arr.scalarAggregate(AggrType.mean);
    try std.testing.expectEqual(@as(f64, 6.5), result);
}

test "scalar aggregate max" {
    const data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const arr = try NDArray.initFromData(&.{ 3, 4 }, &data);
    defer arr.deinit();
    const result = arr.scalarAggregate(AggrType.max);
    try std.testing.expectEqual(@as(f64, 12.0), result);
}

test "scalar aggregate min" {
    const data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const arr = try NDArray.initFromData(&.{ 3, 4 }, &data);
    defer arr.deinit();
    const result = arr.scalarAggregate(AggrType.min);
    try std.testing.expectEqual(@as(f64, 1.0), result);
}

test "scalar aggregate std" {
    const data = [_]f64{ 2, 4, 4, 4, 5, 5, 5, 7, 9, 9 };
    const arr = try NDArray.initFromData(&.{ 2, 5 }, &data);
    defer arr.deinit();
    const result = arr.scalarAggregate(AggrType.std);
    try std.testing.expectApproxEqRel(@as(f64, 2.1540659228538015), result, 1e-10);
}

test "scalar aggregate consistency" {
    const arr = try NDArray.initOnes(&.{ 10, 10 });
    defer arr.deinit();

    // Compare scalar aggregate with array aggregate
    const result_array = try arr.initAggregate(c.NDA_ALL_AXES, AggrType.sum);
    defer result_array.deinit();
    const result_scalar = arr.scalarAggregate(AggrType.sum);

    try std.testing.expectEqual(result_array.get(&.{ 0, 0 }), result_scalar);
}

test "stack" {
    const a = try NDArray.initOnes(&.{ 2, 3 });
    defer a.deinit();
    const b = try NDArray.initOnes(&.{ 2, 3 });
    defer b.deinit();
    const arrays = [_]NDArray{ a, b };
    const result = try NDArray.initStack(0, &arrays);
    defer result.deinit();
    try std.testing.expect(result.ndim() == 3);
}

test "concat" {
    const a = try NDArray.initOnes(&.{ 2, 3 });
    defer a.deinit();
    const b = try NDArray.initOnes(&.{ 2, 3 });
    defer b.deinit();
    const arrays = [_]NDArray{ a, b };
    const result = try NDArray.initConcat(0, &arrays);
    defer result.deinit();
    try std.testing.expect(result.ndim() == 2);
}
