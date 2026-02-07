## Nim bindings for ndarray-c library
## 
## A numpy-like ndarray library for C with multi-dimensional arrays,
## OpenMP parallelization, and BLAS-optimized operations.
## 
## Features automatic memory management using Nim's destructors and move semantics.
## 
## Examples
## --------
## 
## Basic usage:
## 
## .. code-block:: nim
##   import ndarray
##   
##   # Create a 2x3 array of ones (simple int syntax)
##   let arr = newOnes(@[2, 3])
##   
##   # Set value at position (1, 2)
##   arr.set(@[1, 2], 42.0)
##   
##   # Get value at position
##   let val = arr.get(@[1, 2])
##   
##   # Print the array
##   arr.print("My Array", 2)

import defs
import cdecl
import args
import core
import random
import creation
import slice
import arithmetic
import math
import comparison
import linalg
import manipulation
import aggregation
import combining
import io
import properties

export defs
export core
export random
export creation
export slice
export arithmetic
export math
export comparison
export linalg
export manipulation
export aggregation
export combining
export io
export properties

# Core methods (get/set with C types kept private)
proc get(arr: NDArray, pos: openArray[csize_t]): float =
  ## Gets the value at the specified position (C version).
  ##
  ## **Parameters:**
  ## * `pos` - Array of indices for each dimension
  ##
  ## **Returns:** The value at the specified position
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newOnes(@[3, 4])
  ##   let val = arr.get(@[1.csize_t, 2])
  var c_pos = toCPos(pos)
  float(c_get(arr.handle, addr c_pos[0]))

proc set(arr: NDArray, pos: openArray[csize_t], value: float) =
  ## Sets the value at the specified position (C version).
  ##
  ## **Parameters:**
  ## * `pos` - Array of indices for each dimension
  ## * `value` - The value to set
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newZeros(@[3, 4])
  ##   arr.set(@[0.csize_t, 0], 42.0)
  var c_pos = toCPos(pos)
  c_set(arr.handle, addr c_pos[0], cdouble(value))

proc get*(arr: NDArray, pos: openArray[int]): float =
  ## Gets the value at the specified position (int version).
  ##
  ## Convenience overload that accepts int arrays instead of csize_t.
  ##
  ## See also:
  ## * `get<#get,NDArray,openArray[csize_t]>`_
  var c_pos: seq[csize_t]
  for p in pos: c_pos.add(csize_t(p))
  arr.get(c_pos)

proc set*(arr: NDArray, pos: openArray[int], value: float) =
  ## Sets the value at the specified position (int version).
  ##
  ## Convenience overload that accepts int arrays instead of csize_t.
  ##
  ## See also:
  ## * `set<#set,NDArray,openArray[csize_t],float>`_
  var c_pos: seq[csize_t]
  for p in pos: c_pos.add(csize_t(p))
  arr.set(c_pos, value)

# Map functions (requires C calling convention)
proc mapFn*(arr: var NDArray, fn: proc(x: cdouble): cdouble {.cdecl.}):
    var NDArray {.discardable.} =
  ## Applies a function to each element (modifies arr in place).
  ##
  ## **Parameters:**
  ## * `fn` - C-compatible function to apply to each element
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   proc square(x: cdouble): cdouble {.cdecl.} = x * x
  ##   var arr = newArange(@[2, 3], 1.0, 7.0, 1.0)
  ##   arr.mapFn(square)  # Squares all elements
  discard c_mapfnc(arr.handle, fn)
  arr

proc mapFnPar*(arr: var NDArray, fn: proc(a: cdouble, b: cdouble): cdouble {.cdecl.}, value: float): var NDArray {.discardable.} =
  ## Apply a binary function with a constant value in parallel (modifies arr in place).
  ##
  ## Applies fn(element, value) to each element of array in parallel.
  ## This is useful for operations like adding a constant, scaling, etc.
  ##
  ## **Parameters:**
  ## * `fn` - C-compatible binary function to apply
  ## * `value` - Constant value to pass as second argument to fn
  ##
  ## **Returns:** Modified array (for method chaining)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   proc addConstant(x, c: cdouble): cdouble {.cdecl.} = x + c
  ##   var arr = newOnes(@[2, 3])
  ##   arr.mapFnPar(addConstant, 5.0)  # All elements become 6.0
  discard c_mapfnc_par(arr.handle, fn, cdouble(value))
  arr

proc mapMul*(arr: var NDArray, fn: proc(x: cdouble): cdouble {.cdecl.}, 
              other: NDArray, alpha: cdouble): var NDArray {.discardable.} =
  ## Map function then multiply: arr = func(arr) * other * alpha.
  ##
  ## Combines function mapping with element-wise multiplication and scaling.
  ##
  ## **Parameters:**
  ## * `fn` - C-compatible function to apply
  ## * `other` - Array to multiply with (must have same shape)
  ## * `alpha` - Additional scale factor
  ##
  ## **Returns:** Modified array (for method chaining)
  discard c_map_mul(arr.handle, fn, other.handle, alpha)
  arr

proc mulAdd*(arr: var NDArray, other: NDArray, dest: NDArray, alpha: cdouble, beta: cdouble): var NDArray {.discardable.} =
  ## Fused multiply-add: dest = alpha * (arr * other) + beta * dest.
  ##
  ## Efficient combined operation useful in neural networks and optimization.
  ##
  ## **Parameters:**
  ## * `other` - Array to multiply with (must have same shape as arr)
  ## * `dest` - Destination array (must have same shape)
  ## * `alpha` - Scale factor for product
  ## * `beta` - Scale factor for dest
  ##
  ## **Returns:** Modified arr (for method chaining)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   var a = newFull(@[2, 2], 2.0)
  ##   let b = newFull(@[2, 2], 3.0)
  ##   var c = newOnes(@[2, 2])
  ##   a.mulAdd(b, c, 1.0, 2.0)  # c = 1*(2*3) + 2*1 = 8
  discard c_mul_add(arr.handle, other.handle, dest.handle, alpha, beta)
  arr