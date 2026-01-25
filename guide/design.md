# Design Considerations

- The priority is a clean and simple API over performance
- Provides helper macros for list manipulation to express dimensions,
  positions, axes, and many ndarrays.
- Support only basic array operations.
- Uses assertions for runtime checks; can be disabled with `NDEBUG`
- All arrays must have at least 2 dimensions (ndim >= 2). For 1D arrays, use shape `[1, n]` or `[n, 1]`
- For most of the operations, it is prefered to overwrite existing
  arrays instead of creating new ones (in place operations)
- Operations that create new arrays have `_new_` in their name (e.g.,
  `ndarray_new_matmul`). In that way, it is easy to identify which
  objects need to be freed later.
- Uses C99 standard; requires C99-compatible compiler
- Uses `double` as the default data type for array elements
- Uses row-major order for array storage
- Integrated with OpenMP for parallel operations (required)
- Uses CBLAS (OpenBLAS) for optimized linear algebra operations (required)
