ifdef USE_OPENMP
    OMP_FLAGS = -fopenmp -DUSE_OPENMP
else
    OMP_FLAGS =
endif
CFLAGS = -O3 -Wall -g -std=c99 -pedantic -march=native $(OMP_FLAGS)
LDFLAGS = -lm $(OMP_FLAGS)
TEST_LDFLAGS = -lm $(OMP_FLAGS) -lcunit
BIN = example
TEST_BIN = ndarray_test
BENCH_SEQ = benchmark_seq
BENCH_OMP = benchmark_omp
OBJ = ndarray.o
TEST_OBJ = ndarray.o
BENCH_OBJ = ndarray.o

.PHONY: all clean test benchmark

all: $(BIN)

test: $(TEST_BIN)
	./$(TEST_BIN)

benchmark:
	@echo "Running benchmark comparison..."
	@./run_benchmark.sh

clean:
	rm -f $(BIN) $(TEST_BIN) $(OBJ) ndarray_test.o
	rm -f $(BENCH_SEQ) $(BENCH_OMP) benchmark_seq.txt benchmark_omp.txt

$(BIN): $(OBJ)

$(TEST_BIN): ndarray_test.o $(TEST_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(TEST_LDFLAGS)

ndarray_test.o: ndarray_test.c ndarray.h
	$(CC) $(CFLAGS) -c ndarray_test.c

