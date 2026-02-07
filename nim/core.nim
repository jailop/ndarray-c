import defs
import args
import cdecl

proc newNDArray(dims: openArray[csize_t]): NDArray =
  var c_dims = toCDims(dims)
  let handle = c_new(addr c_dims[0])
  if handle.isNil:
    raise newException(ValueError, "Failed to create ndarray")
  wrapHandle(handle)

## Creates a new ndarray with specified dimensions.
##
## All elements are uninitialized. Use `newZeros`, `newOnes`, or
## `newFull` for initialized arrays.
##
## **Parameters:**
## * `dims` - Array of dimension sizes (must have at least 2 elements)
##
## **Returns:** New NDArray with uninitialized data
##
## **Raises:** ValueError if creation fails
##
## Example:
## 
## .. code-block:: nim
##   let arr = newNDArray(@[3, 4])  # 3x4 uninitialized array
proc newNDArray*(dims: openArray[int]): NDArray =
  var c_dims: seq[csize_t]
  for d in dims: c_dims.add(csize_t(d))
  newNDArray(c_dims)

proc newZeros(dims: openArray[csize_t]): NDArray =
  var c_dims = toCDims(dims)
  let handle = c_new_zeros(addr c_dims[0])
  if handle.isNil:
    raise newException(ValueError, "Failed to create ndarray")
  wrapHandle(handle)

## Creates a new ndarray filled with zeros.
##
## **Parameters:**
## * `dims` - Array of dimension sizes (must have at least 2 elements)
##
## **Returns:** New NDArray filled with 0.0
##
## **Raises:** ValueError if creation fails
##
## Example:
## 
## .. code-block:: nim
##   let arr = newZeros(@[3, 4])  # 3x4 array of zeros
proc newZeros*(dims: openArray[int]): NDArray =
  var c_dims: seq[csize_t]
  for d in dims: c_dims.add(csize_t(d))
  newZeros(c_dims)

proc newOnes(dims: openArray[csize_t]): NDArray =
  var c_dims = toCDims(dims)
  let handle = c_new_ones(addr c_dims[0])
  if handle.isNil:
    raise newException(ValueError, "Failed to create ndarray")
  wrapHandle(handle)

## Creates a new ndarray filled with ones.
##
## **Parameters:**
## * `dims` - Array of dimension sizes (must have at least 2 elements)
##
## **Returns:** New NDArray filled with 1.0
##
## **Raises:** ValueError if creation fails
##
## Example:
## 
## .. code-block:: nim
##   let arr = newOnes(@[3, 4])  # 3x4 array of ones
proc newOnes*(dims: openArray[int]): NDArray =
  ## Creates a new ndarray filled with ones (int version).
  ##
  ## Convenience overload that accepts int arrays instead of csize_t.
  ##
  ## See also:
  ## * `newOnes<#newOnes,openArray[csize_t]>`_
  var c_dims: seq[csize_t]
  for d in dims: c_dims.add(csize_t(d))
  newOnes(c_dims)

proc newFull(dims: openArray[csize_t], value: float): NDArray =
  var c_dims = toCDims(dims)
  let handle = c_new_full(addr c_dims[0], cdouble(value))
  if handle.isNil:
    raise newException(ValueError, "Failed to create ndarray")
  wrapHandle(handle)

## Creates a new ndarray filled with a specific value.
##
## **Parameters:**
## * `dims` - Array of dimension sizes (must have at least 2 elements)
## * `value` - The value to fill the array with
##
## **Returns:** New NDArray filled with the specified value
##
## **Raises:** ValueError if creation fails
##
## Example:
## 
## .. code-block:: nim
##   let arr = newFull(@[3, 4], 5.0)  # 3x4 array filled with 5.0
proc newFull*(dims: openArray[int], value: float): NDArray =
  var c_dims: seq[csize_t]
  for d in dims: c_dims.add(csize_t(d))
  newFull(c_dims, value)

proc newFromData(dims: openArray[csize_t], data: openArray[float]): NDArray =
  var c_dims = toCDims(dims)
  let handle = c_new_from_data(addr c_dims[0], unsafeAddr data[0])
  if handle.isNil:
    raise newException(ValueError, "Failed to create ndarray")
  wrapHandle(handle)

## Creates a new ndarray from existing data.
##
## The data is copied into the new array. Data should be in row-major order.
##
## **Parameters:**
## * `dims` - Array of dimension sizes (must have at least 2 elements)
## * `data` - Array of values to copy (size must match product of dims)
##
## **Returns:** New NDArray with copied data
##
## **Raises:** ValueError if creation fails
##
## Example:
## 
## .. code-block:: nim
##   let data = @[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
##   let arr = newFromData(@[2, 3], data)  # 2x3 array from data
proc newFromData*(dims: openArray[int], data: openArray[float]): NDArray =
  var c_dims: seq[csize_t]
  for d in dims: c_dims.add(csize_t(d))
  newFromData(c_dims, data)
