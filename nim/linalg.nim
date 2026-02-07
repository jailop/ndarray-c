import defs
import args
import cdecl

proc newMatmul*(arr: NDArray, other: NDArray): NDArray =
  ## Matrix multiplication (returns new array).
  ##
  ## Performs matrix multiplication using BLAS-optimized routines.
  ##
  ## **Parameters:**
  ## * `other` - Matrix to multiply with (inner dimensions must match)
  ##
  ## **Returns:** New array with matrix product
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let a = newOnes(@[2, 3])  # 2x3 matrix
  ##   let b = newOnes(@[3, 4])  # 3x4 matrix
  ##   let c = a.newMatmul(b)    # 2x4 result
  let handle = c_new_matmul(arr.handle, other.handle)
  if handle.isNil:
    raise newException(ValueError, "Matrix multiplication failed")
  wrapHandle(handle)

proc newTensordot*(arr: NDArray, other: NDArray, axesA: openArray[int], axesB: openArray[int]): NDArray =
  ## Tensor contraction over specified axes (returns new array).
  ##
  ## Generalized tensor product contracting specified axes.
  ##
  ## **Parameters:**
  ## * `other` - Tensor to contract with
  ## * `axesA` - Axes of arr to contract
  ## * `axesB` - Axes of other to contract (must match length of axesA)
  ##
  ## **Returns:** New array with contracted tensor
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let a = newOnes(@[2, 3, 4])
  ##   let b = newOnes(@[4, 5])
  ##   let c = a.newTensordot(b, @[2], @[0])  # Contract axis 2 of a with axis 0 of b
  var c_axes_a = toCAxes(axesA)
  var c_axes_b = toCAxes(axesB)
  let handle = c_new_tensordot(arr.handle, other.handle, addr c_axes_a[0], addr c_axes_b[0])
  if handle.isNil:
    raise newException(ValueError, "Tensor dot failed")
  wrapHandle(handle)