import defs

func toCDims*(dims: openArray[csize_t]): array[MAX_DIMS + 1, csize_t] =
  ## Convert Nim array to C-style zero-terminated array
  if dims.len > MAX_DIMS:
    raise newException(ValueError, "Too many dimensions")
  for i in 0..<dims.len:
    result[i] = dims[i]
  result[dims.len] = 0

func toCPos*(pos: openArray[csize_t]): array[MAX_DIMS, csize_t] =
  ## Convert Nim array to C position array
  if pos.len > MAX_DIMS:
    raise newException(ValueError, "Too many dimensions")
  for i in 0..<pos.len:
    result[i] = pos[i]

func toCAxes*(axes: openArray[cint]): array[MAX_DIMS + 1, cint] =
  ## Convert Nim array to C-style axes array with -1 sentinel
  if axes.len > MAX_DIMS:
    raise newException(ValueError, "Too many axes")
  for i in 0..<axes.len:
    result[i] = axes[i]
  result[axes.len] = -1

func toCAxes*(axes: openArray[int]): array[MAX_DIMS + 1, cint] =
  ## Convert Nim int array to C-style axes array with -1 sentinel
  if axes.len > MAX_DIMS:
    raise newException(ValueError, "Too many axes")
  for i in 0..<axes.len:
    result[i] = cint(axes[i])
  result[axes.len] = -1

func toCArrayList*(arrays: openArray[NDArray]): array[MAX_ARRAYS + 1, NdarrayPtr] =
  ## Convert Nim array to C-style NULL-terminated array
  if arrays.len > MAX_ARRAYS:
    raise newException(ValueError, "Too many arrays")
  for i in 0..<arrays.len:
    result[i] = arrays[i].handle
  result[arrays.len] = nil
