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
# Using installed library (CMake install or Makefile install)
gcc -fopenmp -o myprogram myprogram.c -I/usr/local/include -L/usr/local/lib -lndarray -lopenblas -lm

# On macOS with Homebrew
gcc-15 -fopenmp -o myprogram myprogram.c \
    -I/usr/local/include \
    -I/opt/homebrew/opt/libomp/include \
    -I/opt/homebrew/opt/openblas/include \
    -L/usr/local/lib \
    -L/opt/homebrew/opt/libomp/lib \
    -L/opt/homebrew/opt/openblas/lib \
    -lndarray -lopenblas -lgomp -lm

# Or link directly with source files (no installation needed)
gcc -fopenmp -o myprogram myprogram.c -Isrc src/ndarray_*.c -lopenblas -lm
```

### Using with CMake

Create a `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject C)

find_package(ndarray REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(myprogram myprogram.c)
target_link_libraries(myprogram PRIVATE ndarray::ndarray OpenMP::OpenMP_C)
```

Then build:
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

---
[← Building](building.md) | [Back to Main](../README.md)
