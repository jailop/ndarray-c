import defs
import args
import cdecl

proc newStack*(axis: int, arrays: openArray[NDArray]): NDArray =
  ## Stacks arrays along a new axis (returns new array).
  ##
  ## Creates a new dimension and stacks arrays along it.
  ##
  ## **Parameters:**
  ## * `axis` - Position of new axis in result
  ## * `arrays` - Arrays to stack (must have same shape)
  ##
  ## **Returns:** New array with stacked data
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let a = newOnes(@[2, 3])
  ##   let b = newZeros(@[2, 3])
  ##   let stacked = newStack(0, @[a, b])  # Shape [2, 2, 3]
  var c_array_list = toCArrayList(arrays)
  let handle = c_new_stack(cint(axis), addr c_array_list[0])
  if handle.isNil:
    raise newException(ValueError, "Stack failed")
  wrapHandle(handle)

proc newConcat*(axis: int, arrays: openArray[NDArray]): NDArray =
  ## Concatenates arrays along an existing axis (returns new array).
  ##
  ## Joins arrays along the specified axis.
  ##
  ## **Parameters:**
  ## * `axis` - Axis along which to concatenate
  ## * `arrays` - Arrays to concatenate (shapes must match except on concat axis)
  ##
  ## **Returns:** New array with concatenated data
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let a = newOnes(@[2, 3])
  ##   let b = newZeros(@[2, 3])
  ##   let concatenated = newConcat(0, @[a, b])  # Shape [4, 3]
  var c_array_list = toCArrayList(arrays)
  let handle = c_new_concat(cint(axis), addr c_array_list[0])
  if handle.isNil:
    raise newException(ValueError, "Concatenation failed")
  wrapHandle(handle)