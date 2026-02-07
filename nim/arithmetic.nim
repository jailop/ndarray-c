import defs
import cdecl

proc add*(arr: var NDArray, other: NDArray): var NDArray {.discardable.} =
  ## Element-wise addition (modifies arr in place).
  ##
  ## Performs arr = arr + other element by element.
  ##
  ## **Parameters:**
  ## * `other` - Array to add (must have same shape)
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   var a = newOnes(@[2, 3])
  ##   let b = newFull(@[2, 3], 2.0)
  ##   a.add(b)  # a now contains 3.0 everywhere
  discard c_add(arr.handle, other.handle)
  arr

proc mul*(arr: var NDArray, other: NDArray): var NDArray {.discardable.} =
  ## Element-wise multiplication (modifies arr in place).
  ##
  ## Performs arr = arr * other element by element.
  ##
  ## **Parameters:**
  ## * `other` - Array to multiply (must have same shape)
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   var a = newFull(@[2, 3], 5.0)
  ##   let b = newFull(@[2, 3], 2.0)
  ##   a.mul(b)  # a now contains 10.0 everywhere
  discard c_mul(arr.handle, other.handle)
  arr

proc addScalar*(arr: var NDArray, scalar: float): var NDArray {.discardable.} =
  ## Adds scalar to all elements (modifies arr in place).
  ##
  ## Performs arr = arr + scalar for all elements.
  ##
  ## **Parameters:**
  ## * `scalar` - Value to add to each element
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   var arr = newZeros(@[2, 3])
  ##   arr.addScalar(5.0)  # All elements now 5.0
  discard c_add_scalar(arr.handle, cdouble(scalar))
  arr

proc mulScalar*(arr: var NDArray, scalar: float): var NDArray {.discardable.} =
  ## Multiplies all elements by scalar (modifies arr in place).
  ##
  ## Performs arr = arr * scalar for all elements.
  ##
  ## **Parameters:**
  ## * `scalar` - Value to multiply each element by
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   var arr = newOnes(@[2, 3])
  ##   arr.mulScalar(10.0)  # All elements now 10.0
  discard c_mul_scalar(arr.handle, cdouble(scalar))
  arr

proc axpby*(arr: var NDArray, alpha: float, other: NDArray, beta: float): var NDArray {.discardable.} =
  ## Linear combination: arr = alpha*arr + beta*other.
  ##
  ## BLAS-style operation for efficient linear combinations.
  ##
  ## **Parameters:**
  ## * `alpha` - Scalar multiplier for arr
  ## * `other` - Second array (must have same shape)
  ## * `beta` - Scalar multiplier for other
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   var a = newFull(@[2, 3], 2.0)
  ##   let b = newFull(@[2, 3], 3.0)
  ##   a.axpby(2.0, b, 3.0)  # a = 2*a + 3*b = 4 + 9 = 13
  discard c_axpby(arr.handle, cdouble(alpha), other.handle, cdouble(beta))
  arr

proc scaleShift*(arr: var NDArray, alpha: float, beta: float): var NDArray {.discardable.} =
  ## Scale and shift: arr = alpha*arr + beta.
  ##
  ## Combines scaling and addition in one operation.
  ##
  ## **Parameters:**
  ## * `alpha` - Scale factor
  ## * `beta` - Shift amount
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   var arr = newOnes(@[2, 3])
  ##   arr.scaleShift(2.0, 3.0)  # arr = 2*1 + 3 = 5
  discard c_scale_shift(arr.handle, cdouble(alpha), cdouble(beta))
  arr

proc mulScaled*(arr: var NDArray, other: NDArray, scalar: float): var NDArray {.discardable.} =
  ## Element-wise multiply then scale: arr = arr * other * scalar.
  ##
  ## **Parameters:**
  ## * `other` - Array to multiply (must have same shape)
  ## * `scalar` - Additional scale factor
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   var a = newFull(@[2, 3], 2.0)
  ##   let b = newFull(@[2, 3], 3.0)
  ##   a.mulScaled(b, 0.5)  # a = 2 * 3 * 0.5 = 3
  discard c_mul_scaled(arr.handle, other.handle, cdouble(scalar))
  arr