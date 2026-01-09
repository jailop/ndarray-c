# ndarray for C and Zig

A numpy-like ndarray library for C, with Zig bindings.

- Multi-dimensional arrays (ndim >= 2)
- OpenMP parallelization
- BLAS-optimized operations

Dependencies:

- OpenMP
- OpenBLAS

Documentation:

- [Design Considerations](guide/design.md)  
- [C Building](guide/building.md)
- [Zig Building](guide/zig-building.md)
- [API Reference](https://jailop.github.io/ndarray-c/)  

To add this library to your zig project:

```sh
zig fetch --save "git+https://github.com/jailop/ndarray-c#main"
```

This library has also bidings for Nim:

- [Nim Bindings](https://github.com/jailop/ndarray-nim)

## Disclaimers

- This is a project for learning.
- The API can change at any moment.
- It is not intended for production use.
- Feedback is welcomed

## Pending decisions

- It has not being decided the error management approach. At this
  moment, only asserts are applied.
- The intention is that the name of the function indicates if the result
  is a new allocated array or it is only an scalar. Functions that
  doesn't have an indication about its return value perform inplace
  operations over the first argument. This still needs to be refined.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) file for details.
