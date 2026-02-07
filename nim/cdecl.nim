import defs

{.push dynlib: "libndarray.so".}

# Low-level C bindings
proc c_new*(dims: ptr csize_t): NdarrayPtr {.
  importc: "ndarray_new", header: "ndarray.h".}

proc c_set*(t: NdarrayPtr, pos: ptr csize_t, value: cdouble) {.
  importc: "ndarray_set", header: "ndarray.h".}

proc c_get*(t: NdarrayPtr, pos: ptr csize_t): cdouble {.
  importc: "ndarray_get", header: "ndarray.h".}

proc c_set_slice*(arr: NdarrayPtr, axis: cint, index: csize_t, values: ptr cdouble): NdarrayPtr {.
  importc: "ndarray_set_slice", header: "ndarray.h".}

proc c_fill_slice*(arr: NdarrayPtr, axis: cint, index: csize_t, value: cdouble): NdarrayPtr {.
  importc: "ndarray_fill_slice", header: "ndarray.h".}

proc c_get_slice_ptr*(arr: NdarrayPtr, axis: cint, index: csize_t): ptr cdouble {.
  importc: "ndarray_get_slice_ptr", header: "ndarray.h".}

proc c_copy_slice*(dst: NdarrayPtr, dstAxis: cint, dstIdx: csize_t,
                  src: NdarrayPtr, srcAxis: cint, srcIdx: csize_t): NdarrayPtr {.
  importc: "ndarray_copy_slice", header: "ndarray.h".}

proc c_get_slice_size*(arr: NdarrayPtr, axis: cint): csize_t {.
  importc: "ndarray_get_slice_size", header: "ndarray.h".}

proc c_print*(arr: NdarrayPtr, name: cstring, precision: cint) {.
  importc: "ndarray_print", header: "ndarray.h".}

proc c_new_zeros*(dims: ptr csize_t): NdarrayPtr {.
  importc: "ndarray_new_zeros", header: "ndarray.h".}

proc c_new_from_data*(dims: ptr csize_t, data: ptr cdouble): NdarrayPtr {.
  importc: "ndarray_new_from_data", header: "ndarray.h".}

proc c_new_ones*(dims: ptr csize_t): NdarrayPtr {.
  importc: "ndarray_new_ones", header: "ndarray.h".}

proc c_new_full*(dims: ptr csize_t, value: cdouble): NdarrayPtr {.
  importc: "ndarray_new_full", header: "ndarray.h".}

proc c_new_arange*(dims: ptr csize_t, start: cdouble, stop: cdouble, step: cdouble): NdarrayPtr {.
  importc: "ndarray_new_arange", header: "ndarray.h".}

proc c_new_linspace*(dims: ptr csize_t, start: cdouble, stop: cdouble, num: csize_t): NdarrayPtr {.
  importc: "ndarray_new_linspace", header: "ndarray.h".}

proc c_new_randnorm*(dims: ptr csize_t, mean: cdouble, stddev: cdouble): NdarrayPtr {.
  importc: "ndarray_new_randnorm", header: "ndarray.h".}

proc c_new_randunif*(dims: ptr csize_t, low: cdouble, high: cdouble): NdarrayPtr {.
  importc: "ndarray_new_randunif", header: "ndarray.h".}

proc c_new_randpoisson*(dims: ptr csize_t, lambda: cdouble): NdarrayPtr {.
  importc: "ndarray_new_randpoisson", header: "ndarray.h".}

# Arithmetic operations
proc c_add*(A: NdarrayPtr, B: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_add", header: "ndarray.h".}

proc c_mul*(A: NdarrayPtr, B: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_mul", header: "ndarray.h".}

proc c_add_scalar*(A: NdarrayPtr, scalar: cdouble): NdarrayPtr {.
  importc: "ndarray_add_scalar", header: "ndarray.h".}

proc c_mul_scalar*(A: NdarrayPtr, scalar: cdouble): NdarrayPtr {.
  importc: "ndarray_mul_scalar", header: "ndarray.h".}

proc c_mapfnc*(A: NdarrayPtr, fn: proc(x: cdouble): cdouble {.cdecl.}): NdarrayPtr {.
  importc: "ndarray_mapfnc", header: "ndarray.h".}

proc c_axpby*(A: NdarrayPtr, alpha: cdouble, B: NdarrayPtr, beta: cdouble): NdarrayPtr {.
  importc: "ndarray_axpby", header: "ndarray.h".}

proc c_scale_shift*(A: NdarrayPtr, alpha: cdouble, beta: cdouble): NdarrayPtr {.
  importc: "ndarray_scale_shift", header: "ndarray.h".}

proc c_mul_scaled*(A: NdarrayPtr, B: NdarrayPtr, scalar: cdouble): NdarrayPtr {.
  importc: "ndarray_mul_scaled", header: "ndarray.h".}

proc c_map_mul*(A: NdarrayPtr, fn: proc(x: cdouble): cdouble {.cdecl.}, 
               B: NdarrayPtr, alpha: cdouble): NdarrayPtr {.
  importc: "ndarray_map_mul", header: "ndarray.h".}

proc c_mul_add*(A: NdarrayPtr, B: NdarrayPtr, C: NdarrayPtr, alpha: cdouble, beta: cdouble): NdarrayPtr {.
  importc: "ndarray_mul_add", header: "ndarray.h".}

# proc c_gemv(A: NdarrayPtr, x: NdarrayPtr, alpha: cdouble, beta: cdouble, y: NdarrayPtr): NdarrayPtr {.
#   importc: "ndarray_gemv", header: "ndarray.h".}

proc c_clip_min*(A: NdarrayPtr, minVal: cdouble): NdarrayPtr {.
  importc: "ndarray_clip_min", header: "ndarray.h".}

proc c_clip_max*(A: NdarrayPtr, maxVal: cdouble): NdarrayPtr {.
  importc: "ndarray_clip_max", header: "ndarray.h".}

proc c_clip*(A: NdarrayPtr, minVal: cdouble, maxVal: cdouble): NdarrayPtr {.
  importc: "ndarray_clip", header: "ndarray.h".}

proc c_abs*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_abs", header: "ndarray.h".}

proc c_sign*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_sign", header: "ndarray.h".}

proc c_exp*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_exp", header: "ndarray.h".}

proc c_log*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_log", header: "ndarray.h".}

proc c_sqrt*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_sqrt", header: "ndarray.h".}

proc c_sin*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_sin", header: "ndarray.h".}

proc c_cos*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_cos", header: "ndarray.h".}

proc c_tan*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_tan", header: "ndarray.h".}

proc c_pow*(A: NdarrayPtr, power: cdouble): NdarrayPtr {.
  importc: "ndarray_pow", header: "ndarray.h".}

# Comparison operations
proc c_new_equal*(A: NdarrayPtr, B: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_new_equal", header: "ndarray.h".}

proc c_new_less*(A: NdarrayPtr, B: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_new_less", header: "ndarray.h".}

proc c_new_greater*(A: NdarrayPtr, B: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_new_greater", header: "ndarray.h".}

proc c_new_equal_scalar*(A: NdarrayPtr, value: cdouble): NdarrayPtr {.
  importc: "ndarray_new_equal_scalar", header: "ndarray.h".}

proc c_new_less_scalar*(A: NdarrayPtr, value: cdouble): NdarrayPtr {.
  importc: "ndarray_new_less_scalar", header: "ndarray.h".}

proc c_new_greater_scalar*(A: NdarrayPtr, value: cdouble): NdarrayPtr {.
  importc: "ndarray_new_greater_scalar", header: "ndarray.h".}

proc c_logical_and*(A: NdarrayPtr, B: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_new_logical_and", header: "ndarray.h".}

proc c_logical_or*(A: NdarrayPtr, B: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_new_logical_or", header: "ndarray.h".}

proc c_logical_not*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_new_logical_not", header: "ndarray.h".}

proc c_where*(condition: NdarrayPtr, x: NdarrayPtr, y: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_new_where", header: "ndarray.h".}

# Linear algebra operations
proc c_new_tensordot*(A: NdarrayPtr, B: NdarrayPtr, axesA: ptr cint, axesB: ptr cint): NdarrayPtr {.
  importc: "ndarray_new_tensordot", header: "ndarray.h".}

proc c_new_matmul*(A: NdarrayPtr, B: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_new_matmul", header: "ndarray.h".}

# Manipulation operations
proc c_new_stack*(axis: cint, arrList: ptr NdarrayPtr) : NdarrayPtr {.
  importc: "ndarray_new_stack", header: "ndarray.h".}

proc c_new_concat*(axis: cint, arrList: ptr NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_new_concat", header: "ndarray.h".}

proc c_new_take*(arr: NdarrayPtr, axis: cint, start: csize_t, endIdx: csize_t): NdarrayPtr {.
  importc: "ndarray_new_take", header: "ndarray.h".}

proc c_new_transpose*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_new_transpose", header: "ndarray.h".}

proc c_reshape*(arr: NdarrayPtr, newDims: ptr csize_t) {.
  importc: "ndarray_reshape", header: "ndarray.h".}

# Aggregation operations
proc c_new_aggr*(A: NdarrayPtr, axis: cint, aggrType: cint): NdarrayPtr {.
  importc: "ndarray_new_aggr", header: "ndarray.h".}

proc c_scalar_aggr*(A: NdarrayPtr, aggrType: cint): cdouble {.
  importc: "ndarray_scalar_aggr", header: "ndarray.h".}

# Additional mathematical functions
proc c_mapfnc_par*(A: NdarrayPtr, fn: proc (a: cdouble, b: cdouble): cdouble {.cdecl.}, v: cdouble): NdarrayPtr {.
  importc: "ndarray_mapfnc_par", header: "ndarray.h".}

proc c_sinh*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_sinh", header: "ndarray.h".}

proc c_cosh*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_cosh", header: "ndarray.h".}

proc c_tanh*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_tanh", header: "ndarray.h".}

proc c_asin*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_asin", header: "ndarray.h".}

proc c_acos*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_acos", header: "ndarray.h".}

proc c_atan*(A: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_atan", header: "ndarray.h".}

# I/O operations
proc c_save*(arr: NdarrayPtr, filename: cstring): cint {.
  importc: "ndarray_save", header: "ndarray.h".}

proc c_load*(filename: cstring): NdarrayPtr {.
  importc: "ndarray_new_load", header: "ndarray.h".}

{.pop.}
