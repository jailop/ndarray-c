import defs
import cdecl

proc ndim*(arr: NDArray): int =
  ## Gets the number of dimensions.
  ##
  ## **Returns:** Number of array dimensions
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newOnes(@[2, 3, 4])
  ##   echo arr.ndim  # 3
  int(arr.handle.ndim)

proc shape*(arr: NDArray): seq[int] =
  ## Gets the dimension sizes of the array.
  ##
  ## **Returns:** Sequence of dimension sizes
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newOnes(@[2, 3, 4])
  ##   let dims = arr.shape()  # @[2, 3, 4]
  ##   echo "Shape: ", dims
  result = newSeq[int](arr.handle.ndim)
  let dimsPtr = cast[ptr UncheckedArray[csize_t]](arr.handle.dims)
  for i in 0..<arr.handle.ndim:
    result[i] = int(dimsPtr[i])

proc print*(arr: NDArray, name: cstring = nil, precision: cint = 2) =
  ## Prints the array to standard output.
  ##
  ## **Parameters:**
  ## * `name` - Optional name to display
  ## * `precision` - Number of decimal places (default: 2)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newOnes(@[2, 3])
  ##   arr.print("My Array", 2)
  c_print(arr.handle, name, precision)