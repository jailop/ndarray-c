## Documentation

Generate API documentation:

```bash
make docs
# Open docs/html/index.html in a browser
```

Requires Doxygen to be installed for local generation.

OpenMP Support:

OpenMP is required by default. All builds include OpenMP support:

```bash
# Build with OpenMP support (default)
make

# Build library with OpenMP (default)
make lib

# Build tests with OpenMP (default)
make test
```

When compiled with OpenMP:

- Aggregation operations (sum, mean, std, max, min) are parallelized
- Transpose operations are parallelized
- Large matrix operations benefit from multi-threading
- Performance improvements are most noticeable with arrays larger than 1000x1000

Requires OpenMP-compatible compiler (gcc, clang with libomp).

Debug vs Release Builds:

The library uses assertions for runtime checks. Control them with `NDEBUG`.

```bash
# Debug build (default - assertions enabled)
make

# Release build (assertions disabled for performance)
make CFLAGS="-O3 -DNDEBUG -std=c99 -march=native -fopenmp"
```

Debug builds (without `-DNDEBUG`):

- Enable runtime assertions for parameter validation
- Check array dimensions, axis bounds, NULL pointers
- Slower but safer for development

Release builds (with `-DNDEBUG`):

- Disable assertions for maximum performance
- About 5-10% faster for small operations
- Use only with well-tested code

Custom Compiler Flags:

You can override the default flags:

```bash
# Custom optimization level
make CFLAGS="-O2 -Wall -std=c99 -fopenmp"

# Enable sanitizers for debugging
make CFLAGS="-O0 -g -fsanitize=address -std=c99 -fopenmp"

# Profile-guided optimization
make CFLAGS="-O3 -fprofile-generate -std=c99 -fopenmp"
```

## Building Your Program

After installing the library, compile your program:

```bash
gcc -o myprogram myprogram.c -lndarray -lm
```
