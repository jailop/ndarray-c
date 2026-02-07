import defs
import cdecl

proc save*(arr: NDArray, filename: string): bool =
  ## Saves array to binary file.
  ##
  ## **Parameters:**
  ## * `filename` - Path to output file
  ##
  ## **Returns:** true on success, false on failure
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newOnes(@[3, 4])
  ##   if arr.save("myarray.nda"):
  ##     echo "Saved successfully"
  c_save(arr.handle, filename.cstring) == 0

proc newLoad*(filename: string): NDArray =
  ## Loads array from binary file.
  ##
  ## **Parameters:**
  ## * `filename` - Path to input file
  ##
  ## **Returns:** New NDArray with loaded data
  ##
  ## **Raises:** IOError if file cannot be read
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newLoad("myarray.nda")
  ##   arr.print("Loaded array", 2)
  let handle = c_load(filename.cstring)
  if handle.isNil:
    raise newException(IOError, "Failed to load ndarray from file")
  wrapHandle(handle)