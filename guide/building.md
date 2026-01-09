## Building

Requirements:

- OpenMP: For parallel operations (required on Linux/macOS, optional on Windows)
- OpenBLAS: For optimized BLAS operations (required)
- CUnit: For running tests (optional, only needed for tests)

### Platform-Specific Notes

Windows: This library uses C99 features (VLAs) that require GCC or
Clang. MSVC is not supported. Use MinGW-w64 or MSYS2 with GCC. See
[Windows build instructions](#windows-build-with-vcpkg) below.

### Build System Options

The library supports three build systems for maximum cross-platform
compatibility:

#### 1. CMake (Recommended for cross-platform)

```bash
# Configure (auto-detects GCC and Homebrew libraries on macOS)
mkdir build && cd build
cmake ..

# Build everything
cmake --build .

# Build specific targets
cmake --build . --target ndarray_static    # Static library
cmake --build . --target ndarray_shared    # Shared library
cmake --build . --target example           # Example program
cmake --build . --target ndarray_test      # Test suite
cmake --build . --target benchmark_seq     # Sequential benchmark
cmake --build . --target benchmark_omp     # OpenMP benchmark

# Install
sudo cmake --build . --target install

# Run tests
ctest
```

CMake options:
```bash
cmake -DBUILD_SHARED_LIBS=ON      # Build shared library (default: ON)
cmake -DBUILD_STATIC_LIBS=ON      # Build static library (default: ON)
cmake -DBUILD_EXAMPLES=ON         # Build examples (default: ON)
cmake -DBUILD_TESTS=OFF           # Build tests (default: ON)
cmake -DBUILD_BENCHMARKS=ON       # Build benchmarks (default: ON)
cmake -DCMAKE_INSTALL_PREFIX=/usr # Install location
```

#### 2. Zig Build System

```bash
# Build examples
zig build                          # Build examples
zig build run                      # Run basic example
zig build run-extended             # Run extended example

# Build libraries
zig build lib                      # Build both static and shared
zig build static                   # Build static library only
zig build shared                   # Build shared library only

# Tests
zig build test                     # Run Zig tests
zig build test-c                   # Run C tests (requires CUnit)
zig build test-all                 # Run all tests

# Benchmarks
zig build bench                    # Build benchmark executables
zig build run-bench-seq            # Run sequential benchmark
zig build run-bench-omp            # Run OpenMP benchmark
```

Libraries and headers are installed to `zig-out/lib/` and `zig-out/include/`.

#### 3. Traditional Makefile

```bash
# Build example
make

# Build libraries
make lib                           # Build both static and shared
make static                        # Build static library only
make shared                        # Build shared library only

# Install
sudo make install                  # Installs to /usr/local by default
sudo make install PREFIX=/usr      # Custom install location

# Tests and benchmarks
make test                          # Run test suite
make benchmark                     # Run benchmarks
```

### Running Benchmarks

The benchmark script automatically detects and uses the available build system:

```bash
cd benchmarks
./run_benchmark.sh
```

This will:
- Build sequential and OpenMP versions
- Run both benchmarks
- Generate a performance comparison report
- Show speedup for each operation

Run tests:

### Installing Dependencies

**Windows (vcpkg - Recommended):**

Install vcpkg if not already installed:
```powershell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
```

Install GCC via MSYS2 or MinGW-w64:
```powershell
# Option 1: MSYS2 (recommended)
# Download from https://www.msys2.org/ and install
# Then in MSYS2 terminal:
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja

# Option 2: Chocolatey
choco install mingw

# Option 3: Standalone MinGW-w64 from https://winlibs.com/
```

Dependencies will be automatically installed via vcpkg.json when building with CMake.

**macOS (Homebrew):**
```bash
brew install gcc libomp openblas cunit
```

**Ubuntu/Debian:**
```bash
sudo apt-get install gcc libomp-dev libopenblas-dev libcunit1-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install gcc libomp-devel openblas-devel CUnit-devel
```

### Windows Build with vcpkg

The project includes a PowerShell build script for easy Windows compilation:

```powershell
# Quick build (uses Ninja generator with GCC)
.\build.ps1

# Clean build
.\build.ps1 -Clean

# Debug build
.\build.ps1 -BuildType Debug

# With MinGW Makefiles instead of Ninja
.\build.ps1 -Generator "MinGW Makefiles"

# Build and install
.\build.ps1 -Install -InstallPrefix "C:\Program Files\ndarray"
```

Manual CMake build on Windows:
```powershell
# Configure with vcpkg toolchain
cmake -S . -B build -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="C:\vcpkg\scripts\buildsystems\vcpkg.cmake" `
  -DCMAKE_C_COMPILER=gcc

# Build
cmake --build build --config Release

# Run example
.\build\example.exe
```
