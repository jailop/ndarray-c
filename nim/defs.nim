# High-level Nim API with idiomatic names
const MAX_DIMS* = 16
const MAX_ARRAYS* = 64

type
  NdarrayInternal {.importc: "NDArray_", header: "ndarray.h".} = object
    data: ptr cdouble
    dims*: ptr csize_t
    ndim*: csize_t
  
  NdarrayPtr* = ptr NdarrayInternal
  
  NDArray* = object
    ## A handle to a ndarray structure with automatic memory management
    handle*: NdarrayPtr

  AggrType* = enum
    ## Aggregation types
    aggrSum = 0
    aggrMean = 1
    aggrStd = 2
    aggrMax = 3
    aggrMin = 4

const
  ALL_AXES* = -1
    ## Constant to indicate operations on all axes

proc c_free*(t: NdarrayPtr) {.
  importc: "ndarray_free", header: "ndarray.h".}

proc c_new_copy*(t: NdarrayPtr): NdarrayPtr {.
  importc: "ndarray_new_copy", header: "ndarray.h".}



# Destructor for automatic memory management
proc `=destroy`(arr: var NDArray) =
  ## Automatic destructor called by Nim's memory management
  if arr.handle != nil:
    when defined(debugDestructor):
      echo "=destroy called, freeing handle: ", cast[uint](arr.handle)
    c_free(arr.handle)
    arr.handle = nil

proc `=copy`(dest: var NDArray, src: NDArray) =
  ## Copy hook - creates a deep copy of the array
  if dest.handle != nil and dest.handle != src.handle:
    c_free(dest.handle)
  if src.handle != nil:
    dest.handle = c_new_copy(src.handle)
  else:
    dest.handle = nil

proc `=sink`(dest: var NDArray, src: NDArray) =
  ## Sink/move hook - transfers ownership
  if dest.handle != nil and dest.handle != src.handle:
    c_free(dest.handle)
  dest.handle = src.handle

# Helper to create NDArray from raw pointer
proc wrapHandle*(handle: NdarrayPtr): NDArray =
  NDArray(handle: handle)

proc copy*(arr: NDArray): NDArray =
  ## Creates a deep copy of the array.
  ##
  ## Allocates a new array with the same dimensions and copies all data.
  ##
  ## **Returns:** New NDArray with copied data
  ##
  ## **Raises:** ValueError if copy fails
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newOnes(@[3, 4])
  ##   let arrCopy = arr.copy()  # Independent copy
  ##   arrCopy.set(@[0, 0], 5.0)  # Doesn't affect arr
  let handle = c_new_copy(arr.handle)
  if handle.isNil:
    raise newException(ValueError, "Failed to copy ndarray")
  wrapHandle(handle)


