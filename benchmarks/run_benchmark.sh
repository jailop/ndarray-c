#!/bin/bash
# Script to run benchmarks and compare OpenMP vs sequential performance
# Works with Makefile, CMake, or Zig build systems

echo "Building benchmark binaries..."
echo ""

BUILD_METHOD=""
BENCHMARK_SEQ=""
BENCHMARK_OMP=""

# Detect which build system is available and build benchmarks
if [ -f "CMakeLists.txt" ] && [ -d "build" ]; then
    echo "Using CMake build system..."
    BUILD_METHOD="cmake"
    cd build
    cmake --build . --target benchmark_seq --target benchmark_omp > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        BENCHMARK_SEQ="./benchmark_seq"
        BENCHMARK_OMP="./benchmark_omp"
        echo "   ✓ CMake benchmarks ready"
    else
        echo "   ✗ CMake build failed"
        cd ..
        BUILD_METHOD=""
    fi
elif [ -f "build.zig" ]; then
    echo "Using Zig build system..."
    BUILD_METHOD="zig"
    zig build bench > /dev/null 2>&1
    if [ $? -eq 0 ] && [ -f "zig-out/bin/benchmark_seq" ] && [ -f "zig-out/bin/benchmark_omp" ]; then
        BENCHMARK_SEQ="./zig-out/bin/benchmark_seq"
        BENCHMARK_OMP="./zig-out/bin/benchmark_omp"
        echo "   ✓ Zig benchmarks ready"
    else
        echo "   ✗ Zig build failed (trying manual compilation)"
        BUILD_METHOD=""
    fi
fi

# Fallback to manual compilation with GCC
if [ -z "$BUILD_METHOD" ]; then
    echo "Using manual GCC compilation..."
    BUILD_METHOD="manual"
    
    # Find GCC
    GCC_CMD=""
    for gcc_ver in gcc-15 gcc-14 gcc-13 gcc-12 gcc-11 gcc; do
        if command -v $gcc_ver &> /dev/null; then
            GCC_CMD=$gcc_ver
            break
        fi
    done
    
    if [ -z "$GCC_CMD" ]; then
        echo "Error: GCC not found"
        exit 1
    fi
    
    echo "   Using compiler: $GCC_CMD"
    
    # Detect library paths (macOS Homebrew)
    INCLUDE_FLAGS=""
    LINK_FLAGS="-lm"
    
    if [ "$(uname)" == "Darwin" ]; then
        if command -v brew &> /dev/null; then
            LIBOMP_PREFIX=$(brew --prefix libomp 2>/dev/null)
            OPENBLAS_PREFIX=$(brew --prefix openblas 2>/dev/null)
            
            if [ -n "$LIBOMP_PREFIX" ]; then
                INCLUDE_FLAGS="$INCLUDE_FLAGS -I$LIBOMP_PREFIX/include"
                LINK_FLAGS="$LINK_FLAGS -L$LIBOMP_PREFIX/lib"
            fi
            
            if [ -n "$OPENBLAS_PREFIX" ]; then
                INCLUDE_FLAGS="$INCLUDE_FLAGS -I$OPENBLAS_PREFIX/include"
                LINK_FLAGS="$LINK_FLAGS -L$OPENBLAS_PREFIX/lib -lopenblas"
            else
                LINK_FLAGS="$LINK_FLAGS -lopenblas"
            fi
        fi
    else
        LINK_FLAGS="$LINK_FLAGS -lopenblas"
    fi
    
    # Get source directory
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    SRC_DIR="$(dirname "$SCRIPT_DIR")/src"
    BENCH_DIR="$SCRIPT_DIR"
    
    # Build sequential version
    echo "   Building sequential version..."
    $GCC_CMD -O3 -Wall -std=c99 -march=native \
        -I"$SRC_DIR" $INCLUDE_FLAGS \
        -o benchmark_seq \
        "$BENCH_DIR/benchmark.c" \
        "$SRC_DIR"/*.c \
        $LINK_FLAGS > /dev/null 2>&1
    
    if [ $? -ne 0 ]; then
        echo "   ✗ Sequential build failed"
        exit 1
    fi
    echo "   ✓ Sequential binary ready"
    
    # Build OpenMP version
    echo "   Building parallel version..."
    $GCC_CMD -O3 -Wall -std=c99 -march=native -fopenmp \
        -I"$SRC_DIR" $INCLUDE_FLAGS \
        -o benchmark_omp \
        "$BENCH_DIR/benchmark.c" \
        "$SRC_DIR"/*.c \
        $LINK_FLAGS -lgomp > /dev/null 2>&1
    
    if [ $? -ne 0 ]; then
        echo "   ✗ OpenMP build failed"
        exit 1
    fi
    echo "   ✓ Parallel binary ready"
    
    BENCHMARK_SEQ="./benchmark_seq"
    BENCHMARK_OMP="./benchmark_omp"
fi

echo ""

# Run sequential benchmark
echo "=================================================================================="
echo "Running SEQUENTIAL benchmark..."
echo "=================================================================================="
$BENCHMARK_SEQ > benchmark_seq.txt
cat benchmark_seq.txt

echo ""
echo ""

# Run OpenMP benchmark
echo "=================================================================================="
echo "Running OPENMP benchmark..."
echo "=================================================================================="
$BENCHMARK_OMP > benchmark_omp.txt
cat benchmark_omp.txt

echo ""
echo ""

# Generate comparison report
echo "=================================================================================="
echo "PERFORMANCE COMPARISON REPORT"
echo "=================================================================================="
echo ""

# Parse and compare results
python3 << 'PYTHON'
import re

def parse_benchmark(filename):
    results = {}
    with open(filename, 'r') as f:
        content = f.read()
        # Extract benchmark results
        pattern = r'  ([\w\s\(\)/]+?)\s+(\d+\.\d+)\s+sec'
        matches = re.findall(pattern, content)
        for name, time in matches:
            results[name.strip()] = float(time)
    return results

seq = parse_benchmark('benchmark_seq.txt')
omp = parse_benchmark('benchmark_omp.txt')

print(f"{'Operation':<45} {'Sequential':>12} {'OpenMP':>12} {'Speedup':>10}")
print("=" * 85)

total_seq = 0
total_omp = 0

for name in sorted(seq.keys()):
    if name in omp:
        seq_time = seq[name]
        omp_time = omp[name]
        speedup = seq_time / omp_time
        total_seq += seq_time
        total_omp += omp_time
        
        # Color code speedup
        if speedup > 1.2:
            indicator = "⚡"
        elif speedup > 1.0:
            indicator = "✓"
        elif speedup > 0.8:
            indicator = "≈"
        else:
            indicator = "⚠"
        
        print(f"{name:<45} {seq_time:>10.4f}s {omp_time:>10.4f}s {speedup:>9.2f}x {indicator}")

print("=" * 85)
overall_speedup = total_seq / total_omp
print(f"{'OVERALL':<45} {total_seq:>10.4f}s {total_omp:>10.4f}s {overall_speedup:>9.2f}x")
print("")
print(f"Total time saved: {total_seq - total_omp:.4f}s ({(1 - total_omp/total_seq) * 100:.1f}% faster)")
print("")
print("Legend: ⚡ = >20% faster  ✓ = faster  ≈ = similar  ⚠ = slower")
PYTHON

echo ""
echo "Benchmark files saved:"
echo "  - benchmark_seq.txt (sequential results)"
echo "  - benchmark_omp.txt (OpenMP results)"
echo ""

# Cleanup binaries if manually built
if [ "$BUILD_METHOD" == "manual" ]; then
    rm -f benchmark_seq benchmark_omp
fi

# Return to original directory if using CMake
if [ "$BUILD_METHOD" == "cmake" ]; then
    cd ..
fi
