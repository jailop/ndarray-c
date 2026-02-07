import defs
import cdecl

proc newEqual*(arr: NDArray, other: NDArray): NDArray =
  ## Element-wise equality comparison (returns new array).
  ##
  ## Returns 1.0 where arr == other, 0.0 elsewhere.
  ##
  ## **Parameters:**
  ## * `other` - Array to compare (must have same shape)
  ##
  ## **Returns:** New array with comparison results
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let a = newOnes(@[2, 3])
  ##   let b = newOnes(@[2, 3])
  ##   let result = a.newEqual(b)  # All 1.0
  let handle = c_new_equal(arr.handle, other.handle)
  if handle.isNil:
    raise newException(ValueError, "Comparison failed")
  wrapHandle(handle)

proc newLess*(arr: NDArray, other: NDArray): NDArray =
  ## Element-wise less-than comparison (returns new array).
  ##
  ## Returns 1.0 where arr < other, 0.0 elsewhere.
  ##
  ## **Parameters:**
  ## * `other` - Array to compare (must have same shape)
  ##
  ## **Returns:** New array with comparison results
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let a = newArange(@[2, 3], 0.0, 6.0, 1.0)
  ##   let b = newFull(@[2, 3], 3.0)
  ##   let result = a.newLess(b)  # [1, 1, 1, 0, 0, 0]
  let handle = c_new_less(arr.handle, other.handle)
  if handle.isNil:
    raise newException(ValueError, "Comparison failed")
  wrapHandle(handle)

proc newGreater*(arr: NDArray, other: NDArray): NDArray =
  ## Element-wise greater-than comparison (returns new array).
  ##
  ## Returns 1.0 where arr > other, 0.0 elsewhere.
  ##
  ## **Parameters:**
  ## * `other` - Array to compare (must have same shape)
  ##
  ## **Returns:** New array with comparison results
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let a = newArange(@[2, 3], 0.0, 6.0, 1.0)
  ##   let b = newFull(@[2, 3], 2.5)
  ##   let result = a.newGreater(b)  # [0, 0, 0, 1, 1, 1]
  let handle = c_new_greater(arr.handle, other.handle)
  if handle.isNil:
    raise newException(ValueError, "Comparison failed")
  wrapHandle(handle)

proc newEqualScalar*(arr: NDArray, value: float): NDArray =
  ## Scalar equality comparison (returns new array).
  ##
  ## Returns 1.0 where arr == value, 0.0 elsewhere.
  ##
  ## **Parameters:**
  ## * `value` - Scalar value to compare
  ##
  ## **Returns:** New array with comparison results
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newArange(@[2, 3], 0.0, 6.0, 1.0)
  ##   let result = arr.newEqualScalar(3.0)  # 1.0 only at position with value 3
  let handle = c_new_equal_scalar(arr.handle, cdouble(value))
  if handle.isNil:
    raise newException(ValueError, "Comparison failed")
  wrapHandle(handle)

proc newLessScalar*(arr: NDArray, value: float): NDArray =
  ## Scalar less-than comparison (returns new array).
  ##
  ## Returns 1.0 where arr < value, 0.0 elsewhere.
  ##
  ## **Parameters:**
  ## * `value` - Scalar value to compare
  ##
  ## **Returns:** New array with comparison results
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newArange(@[2, 3], 0.0, 6.0, 1.0)
  ##   let result = arr.newLessScalar(3.0)  # 1.0 where values < 3
  let handle = c_new_less_scalar(arr.handle, cdouble(value))
  if handle.isNil:
    raise newException(ValueError, "Comparison failed")
  wrapHandle(handle)

proc newGreaterScalar*(arr: NDArray, value: float): NDArray =
  ## Scalar greater-than comparison (returns new array).
  ##
  ## Returns 1.0 where arr > value, 0.0 elsewhere.
  ##
  ## **Parameters:**
  ## * `value` - Scalar value to compare
  ##
  ## **Returns:** New array with comparison results
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newArange(@[2, 3], 0.0, 6.0, 1.0)
  ##   let result = arr.newGreaterScalar(2.5)  # 1.0 where values > 2.5
  let handle = c_new_greater_scalar(arr.handle, cdouble(value))
  if handle.isNil:
    raise newException(ValueError, "Comparison failed")
  wrapHandle(handle)

proc newLogicalAnd*(arr: NDArray, other: NDArray): NDArray =
  ## Logical AND operation (returns new array).
  ##
  ## Returns 1.0 where both arr and other are non-zero, 0.0 elsewhere.
  ##
  ## **Parameters:**
  ## * `other` - Array to AND with (must have same shape)
  ##
  ## **Returns:** New array with logical AND results
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let a = newArange(@[2, 3], 0.0, 6.0, 1.0)
  ##   let b = newGreaterScalar(a, 2.5)
  ##   let c = newLessScalar(a, 4.5)
  ##   let result = b.newLogicalAnd(c)  # 1.0 where 2.5 < val < 4.5
  let handle = c_logical_and(arr.handle, other.handle)
  if handle.isNil:
    raise newException(ValueError, "Logical operation failed")
  wrapHandle(handle)

proc newLogicalOr*(arr: NDArray, other: NDArray): NDArray =
  ## Logical OR operation (returns new array).
  ##
  ## Returns 1.0 where either arr or other is non-zero, 0.0 elsewhere.
  ##
  ## **Parameters:**
  ## * `other` - Array to OR with (must have same shape)
  ##
  ## **Returns:** New array with logical OR results
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let a = newArange(@[2, 3], 0.0, 6.0, 1.0)
  ##   let b = newLessScalar(a, 2.0)
  ##   let c = newGreaterScalar(a, 4.0)
  ##   let result = b.newLogicalOr(c)  # 1.0 where val < 2 OR val > 4
  let handle = c_logical_or(arr.handle, other.handle)
  if handle.isNil:
    raise newException(ValueError, "Logical operation failed")
  wrapHandle(handle)

proc newLogicalNot*(arr: NDArray): NDArray =
  ## Logical NOT operation (returns new array).
  ##
  ## Returns 1.0 where arr is zero, 0.0 where arr is non-zero.
  ##
  ## **Returns:** New array with logical NOT results
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newArange(@[2, 3], 0.0, 6.0, 1.0)
  ##   let mask = newLessScalar(arr, 3.0)
  ##   let inverted = mask.newLogicalNot()  # Inverts the mask
  let handle = c_logical_not(arr.handle)
  if handle.isNil:
    raise newException(ValueError, "Logical operation failed")
  wrapHandle(handle)

proc clip*(arr: var NDArray, minVal: float, maxVal: float): var NDArray {.discardable.} =
  ## Clips values to range [minVal, maxVal] (modifies arr in place).
  ##
  ## Values below minVal become minVal, values above maxVal become maxVal.
  ##
  ## **Parameters:**
  ## * `minVal` - Minimum allowed value
  ## * `maxVal` - Maximum allowed value
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   var arr = newArange(@[2, 3], -1.0, 5.0, 1.0)
  ##   arr.clip(0.0, 3.0)  # Values clamped to [0, 3]
  discard c_clip(arr.handle, cdouble(minVal), cdouble(maxVal))
  arr

proc newWhere*(condition: NDArray, x: NDArray, y: NDArray): NDArray =
  ## NumPy-style conditional selection (returns new array).
  ##
  ## Returns x where condition is non-zero, y where condition is zero.
  ##
  ## **Parameters:**
  ## * `condition` - Boolean array (non-zero = true)
  ## * `x` - Array to select from when condition is true
  ## * `y` - Array to select from when condition is false
  ##
  ## **Returns:** New array with selected values
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let data = newArange(@[2, 3], 0.0, 6.0, 1.0)
  ##   let mask = data.newGreaterScalar(2.5)
  ##   let zeros = newZeros(@[2, 3])
  ##   let filtered = newWhere(mask, data, zeros)  # Keep values > 2.5, rest = 0
  let handle = c_where(condition.handle, x.handle, y.handle)
  if handle.isNil:
    raise newException(ValueError, "Where operation failed")
  wrapHandle(handle)

proc clipMin*(arr: var NDArray, minVal: float): var NDArray {.discardable.} =
  ## Clips values below minimum threshold (modifies arr in place).
  ##
  ## Sets any value less than minVal to minVal.
  ##
  ## **Parameters:**
  ## * `minVal` - Minimum allowed value
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   var arr = newArange(@[2, 3], -2.0, 4.0, 1.0)
  ##   arr.clipMin(0.0)  # Negative values become 0
  discard c_clip_min(arr.handle, cdouble(minVal))
  arr

proc clipMax*(arr: var NDArray, maxVal: float): var NDArray {.discardable.} =
  ## Clips values above maximum threshold (modifies arr in place).
  ##
  ## Sets any value greater than maxVal to maxVal.
  ##
  ## **Parameters:**
  ## * `maxVal` - Maximum allowed value
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   var arr = newArange(@[2, 3], 0.0, 6.0, 1.0)
  ##   arr.clipMax(3.0)  # Values > 3 become 3
  discard c_clip_max(arr.handle, cdouble(maxVal))
  arr



