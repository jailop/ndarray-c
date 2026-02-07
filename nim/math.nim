import defs
import cdecl

proc exp*(arr: var NDArray): var NDArray {.discardable.} =
  ## Applies exponential function element-wise (modifies arr in place).
  ##
  ## Performs arr = exp(arr) for all elements using optimized SIMD operations.
  ## This is equivalent to arr.mapFn(exp) but avoids FFI overhead.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example: 
  ## 
  ## .. code-block:: nim
  ##   var arr = newOnes(@[2, 3])
  ##   arr.exp()  # All elements become e = 2.71828
  discard c_exp(arr.handle)
  arr

proc log*(arr: var NDArray): var NDArray {.discardable.} =
  ## Applies natural logarithm function element-wise (modifies arr in place).
  ##
  ## Performs arr = log(arr) for all elements using optimized SIMD operations.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example: 
  ## 
  ## .. code-block:: nim
  ##   var arr = newFull(@[2, 3], M_E)
  ##   arr.log()  # All elements become 1.0
  discard c_log(arr.handle)
  arr

proc sqrt*(arr: var NDArray): var NDArray {.discardable.} =
  ## Applies square root function element-wise (modifies arr in place).
  ##
  ## Performs arr = sqrt(arr) for all elements using optimized SIMD operations.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example: 
  ## 
  ## .. code-block:: nim
  ##   var arr = newFull(@[2, 3], 4.0)
  ##   arr.sqrt()  # All elements become 2.0
  discard c_sqrt(arr.handle)
  arr

proc sin*(arr: var NDArray): var NDArray {.discardable.} =
  ## Applies sine function element-wise (modifies arr in place).
  ##
  ## Performs arr = sin(arr) for all elements using optimized SIMD operations.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example: 
  ## 
  ## .. code-block:: nim
  ##   var arr = newLinspace(@[1, 4], 0.0, PI, 4)
  ##   arr.sin()  # Apply sine to [0, π/3, 2π/3, π]
  discard c_sin(arr.handle)
  arr

proc cos*(arr: var NDArray): var NDArray {.discardable.} =
  ## Applies cosine function element-wise (modifies arr in place).
  ##
  ## Performs arr = cos(arr) for all elements using optimized SIMD operations.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example: 
  ## 
  ## .. code-block:: nim
  ##   var arr = newLinspace(@[1, 4], 0.0, PI, 4)
  ##   arr.cos()  # Apply cosine to [0, π/3, 2π/3, π]
  discard c_cos(arr.handle)
  arr

proc tan*(arr: var NDArray): var NDArray {.discardable.} =
  ## Applies tangent function element-wise (modifies arr in place).
  ##
  ## Performs arr = tan(arr) for all elements using optimized SIMD operations.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example: 
  ## 
  ## .. code-block:: nim
  ##   var arr = newLinspace(@[1, 4], -PI/4, PI/4)
  ##   arr.tan()  # Apply tangent to [-π/4, 0, π/4]
  discard c_tan(arr.handle)
  arr

proc pow*(arr: var NDArray, power: float): var NDArray {.discardable.} =
  ## Applies power function element-wise (modifies arr in place).
  ##
  ## Performs arr = arr^power for all elements using optimized SIMD operations.
  ##
  ## **Parameters:**
  ## * `power` - Power to raise each element to.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example: 
  ## 
  ## .. code-block:: nim
  ##   var arr = newFull(@[2, 3], 2.0)
  ##   arr.pow(3.0)  # All elements become 8.0 (2^3)
  discard c_pow(arr.handle, cdouble(power))
  arr

proc sinh*(arr: var NDArray): var NDArray {.discardable.} =
  ## Applies hyperbolic sine function element-wise (modifies arr in place).
  ##
  ## Performs arr = sinh(arr) for all elements using optimized SIMD operations.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example: 
  ## 
  ## .. code-block:: nim
  ##   var arr = newFull(@[2, 3], 1.0)
  ##   arr.sinh()  # All elements become sinh(1.0)
  discard c_sinh(arr.handle)
  arr

proc cosh*(arr: var NDArray): var NDArray {.discardable.} =
  ## Applies hyperbolic cosine function element-wise (modifies arr in place).
  ##
  ## Performs arr = cosh(arr) for all elements using optimized SIMD operations.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example: 
  ## 
  ## .. code-block:: nim
  ##   var arr = newFull(@[2, 3], 1.0)
  ##   arr.cosh()  # All elements become cosh(1.0)
  discard c_cosh(arr.handle)
  arr

proc tanh*(arr: var NDArray): var NDArray {.discardable.} =
  ## Applies hyperbolic tangent function element-wise (modifies arr in place).
  ##
  ## Performs arr = tanh(arr) for all elements using optimized SIMD operations.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example: 
  ## 
  ## .. code-block:: nim
  ##   var arr = newFull(@[2, 3], 1.0)
  ##   arr.tanh()  # All elements become tanh(1.0)
  discard c_tanh(arr.handle)
  arr

proc asin*(arr: var NDArray): var NDArray {.discardable.} =
  ## Applies inverse sine function element-wise (modifies arr in place).
  ##
  ## Performs arr = asin(arr) for all elements using optimized SIMD operations.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example: 
  ## 
  ## .. code-block:: nim
  ##   var arr = newFull(@[2, 3], 0.5)
  ##   arr.asin()  # All elements become asin(0.5)
  discard c_asin(arr.handle)
  arr

proc acos*(arr: var NDArray): var NDArray {.discardable.} =
  ## Applies inverse cosine function element-wise (modifies arr in place).
  ##
  ## Performs arr = acos(arr) for all elements using optimized SIMD operations.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example: 
  ## 
  ## .. code-block:: nim
  ##   var arr = newFull(@[2, 3], 0.5)
  ##   arr.acos()  # All elements become acos(0.5)
  discard c_acos(arr.handle)
  arr

proc atan*(arr: var NDArray): var NDArray {.discardable.} =
  ## Applies inverse tangent function element-wise (modifies arr in place).
  ##
  ## Performs arr = atan(arr) for all elements using optimized SIMD operations.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example: 
  ## 
  ## .. code-block:: nim
  ##   var arr = newFull(@[2, 3], 1.0)
  ##   arr.atan()  # All elements become atan(1.0) = π/4
  discard c_atan(arr.handle)
  arr

proc absValue*(arr: var NDArray): var NDArray {.discardable.} =
  ## Applies absolute value function element-wise (modifies arr in place).
  ##
  ## Performs arr = |arr| for all elements using optimized SIMD operations.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example: 
  ## 
  ## .. code-block:: nim
  ##   var arr = newFull(@[2, 3], -2.5)
  ##   arr.absValue()  # All elements become 2.5
  discard c_abs(arr.handle)
  arr

proc sign*(arr: var NDArray): var NDArray {.discardable.} =
  ## Sign function: -1, 0, or +1 (modifies arr in place).
  ##
  ## Returns -1 for negative values, 0 for zero, +1 for positive values.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   var arr = newArange(@[2, 3], -2.0, 4.0, 1.0)
  ##   arr.sign()  # Values become -1, -1, 0, 1, 1, 1
  discard c_sign(arr.handle)
  arr

proc abs*(arr: var NDArray): var NDArray {.discardable.} =
  ## Absolute value (modifies arr in place).
  ##
  ## Replaces each element with its absolute value.
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   var arr = newArange(@[2, 3], -2.0, 4.0, 1.0)
  ##   arr.abs()  # All values now non-negative
  discard c_abs(arr.handle)
  arr