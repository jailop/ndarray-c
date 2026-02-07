import defs
import args
import cdecl

proc newArange(dims: openArray[csize_t], start: float, stop: float, step: float): NDArray =
  ## Creates a new ndarray with evenly spaced values in a range.
  ##
  ## Values are generated sequentially: start, start+step, start+2*step, ...
  ## and filled in row-major order.
  ##
  ## **Parameters:**
  ## * `dims` - Array of dimension sizes (must have at least 2 elements)
  ## * `start` - Starting value (inclusive)
  ## * `stop` - Ending value (exclusive)
  ## * `step` - Step size between values
  ##
  ## **Returns:** New NDArray with evenly spaced values
  ##
  ## **Raises:** ValueError if creation fails
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newArange(@[2, 5], 0.0, 10.0, 1.0)  # Values 0 to 9
  var c_dims = toCDims(dims)
  let handle = c_new_arange(addr c_dims[0], cdouble(start), cdouble(stop), cdouble(step))
  if handle.isNil:
    raise newException(ValueError, "Failed to create ndarray")
  wrapHandle(handle)

proc newLinspace(dims: openArray[csize_t], start: float, stop: float, num: csize_t): NDArray =
  ## Creates a new ndarray with linearly spaced values.
  ##
  ## Values are evenly distributed between start and stop (both inclusive).
  ##
  ## **Parameters:**
  ## * `dims` - Array of dimension sizes (must have at least 2 elements)
  ## * `start` - Starting value (inclusive)
  ## * `stop` - Ending value (inclusive)
  ## * `num` - Number of values to generate
  ##
  ## **Returns:** New NDArray with linearly spaced values
  ##
  ## **Raises:** ValueError if creation fails
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newLinspace(@[2, 5], 0.0, 1.0, 10)  # 10 values from 0 to 1
  var c_dims = toCDims(dims)
  let handle = c_new_linspace(addr c_dims[0], cdouble(start), cdouble(stop), num)
  if handle.isNil:
    raise newException(ValueError, "Failed to create ndarray")
  wrapHandle(handle)

proc newArange*(dims: openArray[int], start: float, stop: float, step: float): NDArray =
  ## Creates a new ndarray with evenly spaced values (int version).
  ##
  ## Convenience overload that accepts int arrays instead of csize_t.
  ##
  ## See also:
  ## * `newArange<#newArange,openArray[csize_t],float,float,float>`_
  var c_dims: seq[csize_t]
  for d in dims: c_dims.add(csize_t(d))
  newArange(c_dims, start, stop, step)

proc newLinspace*(dims: openArray[int], start: float, stop: float, num: csize_t): NDArray =
  ## Creates a new ndarray with linearly spaced values (int version).
  ##
  ## Convenience overload that accepts int arrays instead of csize_t.
  ##
  ## See also:
  ## * `newLinspace<#newLinspace,openArray[csize_t],float,float,csize_t>`_
  var c_dims: seq[csize_t]
  for d in dims: c_dims.add(csize_t(d))
  result = newLinspace(c_dims, start, stop, num)