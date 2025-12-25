#!/bin/bash
# Script to run benchmarks and compare OpenMP vs sequential performance

echo "Building benchmark binaries..."
echo ""

# Build without OpenMP
echo "1. Building sequential version (no OpenMP)..."
make clean > /dev/null 2>&1
gcc -O3 -Wall -g -std=c99 -pedantic -march=native -o benchmark_seq benchmark.c ndarray.c  -lm
if [ $? -ne 0 ]; then
    echo "Error building sequential version"
    exit 1
fi
echo "   ✓ Sequential binary ready"

# Build with OpenMP
echo "2. Building parallel version (with OpenMP)..."
gcc -O3 -Wall -g -std=c99 -pedantic -march=native -fopenmp -DUSE_OPENMP -o benchmark_omp benchmark.c ndarray.c  -lm -fopenmp
if [ $? -ne 0 ]; then
    echo "Error building OpenMP version"
    exit 1
fi
echo "   ✓ Parallel binary ready"
echo ""

# Run sequential benchmark
echo "=================================================================================="
echo "Running SEQUENTIAL benchmark..."
echo "=================================================================================="
./benchmark_seq > benchmark_seq.txt
cat benchmark_seq.txt

echo ""
echo ""

# Run OpenMP benchmark
echo "=================================================================================="
echo "Running OPENMP benchmark..."
echo "=================================================================================="
./benchmark_omp > benchmark_omp.txt
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
