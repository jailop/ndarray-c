import defs
import cdecl

proc newAggregate*(arr: NDArray, axis: int, aggrType: AggrType): NDArray =
  ## Aggregates along an axis (returns new array).
  ##
  ## Reduces the array along the specified axis using the given aggregation type.
  ##
  ## **Parameters:**
  ## * `axis` - Axis to aggregate along (use ALL_AXES for all)
  ## * `aggrType` - Type of aggregation (sum, mean, max, min, std)
  ##
  ## **Returns:** New array with aggregated values
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newArange(@[3, 4], 0.0, 12.0, 1.0)
  ##   let rowSums = arr.newAggregate(0, aggrSum)     # Sum along axis 0
  ##   let colMeans = arr.newAggregate(1, aggrMean)   # Mean along axis 1
  let handle = c_new_aggr(arr.handle, cint(axis), cint(aggrType))
  if handle.isNil:
    raise newException(ValueError, "Aggregation failed")
  wrapHandle(handle)

proc scalarAggregate*(arr: NDArray, aggrType: AggrType): float =
  ## Aggregates all elements to a scalar value.
  ##
  ## **Parameters:**
  ## * `aggrType` - Type of aggregation (sum, mean, max, min, std)
  ##
  ## **Returns:** Scalar result of aggregation
  ##
  ## Example:
  ## 
  ## .. code-block:: nim
  ##   let arr = newArange(@[3, 4], 0.0, 12.0, 1.0)
  ##   let total = arr.scalarAggregate(aggrSum)   # Sum of all elements = 66
  ##   let average = arr.scalarAggregate(aggrMean) # Mean = 5.5
  float(c_scalar_aggr(arr.handle, cint(aggrType)))