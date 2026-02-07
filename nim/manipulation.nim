import defs
import cdecl

proc newTranspose*(arr: NDArray): NDArray =
  ## Transposes array (returns new array).
  ##
  ## Reverses the order of axes. For 2D arrays, swaps rows and columns.
  ##
  ## **Returns:** New array with transposed data
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newArange(@[2, 3], 0.0, 6.0, 1.0)  # 2x3 matrix
  ##   let transposed = arr.newTranspose()          # 3x2 matrix
  let handle = c_new_transpose(arr.handle)
  if handle.isNil:
    raise newException(ValueError, "Transpose failed")
  wrapHandle(handle)

proc reshape*(arr: NDArray, newDims: openArray[int]) =
  ## Reshapes the array in-place to new dimensions.
  ##
  ## Total number of elements must remain the same.
  ## Use -1 for one dimension to automatically infer its size.
  ##
  ## **Parameters:**
  ## * `newDims` - New shape (use -1 for auto-infer)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newArange(@[2, 6], 0.0, 12.0, 1.0)  # 2x6
  ##   arr.reshape(@[3, 4])                          # Now 3x4
  ##   arr.reshape(@[2, -1])                         # Now 2x6 (inferred)
  var c_dims: array[MAX_DIMS + 1, csize_t]
  if newDims.len > MAX_DIMS:
    raise newException(ValueError, "Too many dimensions")
  for i in 0..<newDims.len:
    if newDims[i] == -1:
      c_dims[i] = cast[csize_t](-1)
    else:
      c_dims[i] = csize_t(newDims[i])
  c_dims[newDims.len] = 0
  c_reshape(arr.handle, addr c_dims[0])

proc newTake*(arr: NDArray, axis: int, start: int, `end`: int): NDArray =
  ## Extracts a slice along an axis (returns new array).
  ##
  ## **Parameters:**
  ## * `axis` - Axis along which to slice
  ## * `start` - Starting index (inclusive)
  ## * `end` - Ending index (exclusive)
  ##
  ## **Returns:** New array with extracted slice
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newArange(@[4, 5], 0.0, 20.0, 1.0)
  ##   let slice = arr.newTake(0, 1, 3)  # Rows 1 and 2 (indices 1, 2)
  let handle = c_new_take(arr.handle, cint(axis), csize_t(start), csize_t(`end`))
  if handle.isNil:
    raise newException(ValueError, "Take failed")
  wrapHandle(handle)