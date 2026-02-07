import defs
import cdecl

proc setSlice*(arr: var NDArray, axis: int, index: int, values: openArray[float]): var NDArray {.discardable.} =
  ## Sets values along a slice at a specific index on an axis.
  ##
  ## For 2D: axis=0 sets a row, axis=1 sets a column.
  ## For higher dimensions: sets hyperplane perpendicular to axis.
  ## Returns array to enable method chaining.
  ##
  ## **Parameters:**
  ## * `axis` - The axis along which to set the slice
  ## * `index` - The index along the axis
  ## * `values` - Array of values to set (size must match slice size)
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   var arr = newZeros(@[3, 4])
  ##   let rowData = @[1.0, 2.0, 3.0, 4.0]
  ##   arr.setSlice(0, 0, rowData)  # Set first row
  discard c_set_slice(arr.handle, cint(axis), csize_t(index), unsafeAddr values[0])
  return arr

proc fillSlice*(arr: var NDArray, axis: int, index: int, value: float): var NDArray {.discardable.} =
  ## Fills a slice with a scalar value at a specific index on an axis.
  ##
  ## For 2D: axis=0 fills a row, axis=1 fills a column.
  ## For higher dimensions: fills hyperplane perpendicular to axis.
  ## Returns array to enable method chaining.
  ##
  ## **Parameters:**
  ## * `axis` - The axis along which to fill the slice
  ## * `index` - The index along the axis
  ## * `value` - The scalar value to fill with
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   var arr = newZeros(@[3, 4])
  ##   arr.fillSlice(0, 1, 5.0)  # Fill second row with 5.0
  discard c_fill_slice(arr.handle, cint(axis), csize_t(index), cdouble(value))
  return arr

proc getSlicePtr*(arr: NDArray, axis: int, index: int): ptr float =
  ## Gets a pointer to a slice along an axis.
  ##
  ## **Warning:** Advanced operation for direct memory access.
  ## Use with caution as it bypasses bounds checking.
  ##
  ## **Parameters:**
  ## * `axis` - The axis along which to get the slice
  ## * `index` - The index along the axis
  ##
  ## **Returns:** Pointer to first element of slice
  c_get_slice_ptr(arr.handle, cint(axis), csize_t(index))

proc copySlice*(dst: var NDArray, dstAxis: int, dstIdx: int,
                src: NDArray, srcAxis: int, srcIdx: int): var NDArray {.discardable.} =
  ## Copies a slice from one array to another.
  ##
  ## **Parameters:**
  ## * `dst` - Destination array
  ## * `dstAxis` - Axis in destination array  
  ## * `dstIdx` - Index along destination axis
  ## * `src` - Source array
  ## * `srcAxis` - Axis in source array
  ## * `srcIdx` - Index along source axis
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let src = newOnes(@[3, 4])
  ##   var dst = newZeros(@[3, 4])
  ##   copySlice(dst, 0, 1, src, 0, 0)  # Copy row 0 from src to row 1 of dst
  discard c_copy_slice(dst.handle, cint(dstAxis), csize_t(dstIdx), src.handle, cint(srcAxis), csize_t(srcIdx))
  return dst

proc getSliceSize*(arr: NDArray, axis: int): int =
  ## Gets size of a slice along an axis.
  ##
  ## **Parameters:**
  ## * `axis` - The axis to query
  ##
  ## **Returns:** Number of elements in a slice along that axis
  int(c_get_slice_size(arr.handle, cint(axis)))