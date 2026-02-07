import defs
import args
import cdecl

proc newRandomUniform(dims: openArray[csize_t], low: float, high: float): NDArray =
  var c_dims = toCDims(dims)
  let handle = c_new_randunif(addr c_dims[0], cdouble(low), cdouble(high))
  if handle.isNil:
    raise newException(ValueError, "Failed to create ndarray")
  wrapHandle(handle)

## Creates a new ndarray with random values from uniform distribution.
##
## Values are uniformly distributed in the range [low, high).
##
## **Parameters:**
## * `dims` - Array of dimension sizes (must have at least 2 elements)
## * `low` - Lower bound (inclusive)
## * `high` - Upper bound (exclusive)
##
## **Returns:** New NDArray with random uniform values
##
## **Raises:** ValueError if creation fails
##
## Example:
## 
## .. code-block:: nim
##   let arr = newRandomUniform(@[3, 4], 0.0, 1.0)  # Random values in [0, 1)
proc newRandomUniform*(dims: openArray[int], low: float, high: float): NDArray =
  var c_dims: seq[csize_t]
  for d in dims: c_dims.add(csize_t(d))
  newRandomUniform(c_dims, low, high)

proc newRandomNormal(dims: openArray[csize_t], mean: float, stddev: float): NDArray =
  var c_dims = toCDims(dims)
  let handle = c_new_randnorm(addr c_dims[0], cdouble(mean), cdouble(stddev))
  if handle.isNil:
    raise newException(ValueError, "Failed to create ndarray")
  wrapHandle(handle)

## Creates a new ndarray with random values from normal distribution.
##
## Values follow a Gaussian distribution with specified mean and standard deviation.
##
## **Parameters:**
## * `dims` - Array of dimension sizes (must have at least 2 elements)
## * `mean` - Mean of the distribution
## * `stddev` - Standard deviation of the distribution
##
## **Returns:** New NDArray with random normal values
##
## **Raises:** ValueError if creation fails
##
## Example:
## 
## .. code-block:: nim
##   let arr = newRandomNormal(@[3, 4], 0.0, 1.0)  # Standard normal distribution
proc newRandomNormal*(dims: openArray[int], mean: float, stddev: float): NDArray =
  var c_dims: seq[csize_t]
  for d in dims: c_dims.add(csize_t(d))
  newRandomNormal(c_dims, mean, stddev)


