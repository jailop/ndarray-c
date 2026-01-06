LIB_NAME = ndarray
VERSION = 1.0.0
LIB_STATIC = lib$(LIB_NAME).a
LIB_SHARED = lib$(LIB_NAME).so.$(VERSION)
LIB_SHARED_MAJOR = lib$(LIB_NAME).so.1
LIB_SHARED_BASE = lib$(LIB_NAME).so


PREFIX ?= /usr/local
INCLUDEDIR = $(PREFIX)/include
LIBDIR = $(PREFIX)/lib


CFLAGS = -O3 -Wall -g -std=c99 -pedantic -march=native -fopenmp
CFLAGS_SHARED = $(CFLAGS) -fPIC
LDFLAGS = -lm -fopenmp -lopenblas
TEST_LDFLAGS = -lm -fopenmp -lopenblas -lcunit


BIN = example
TEST_BIN = ndarray_test
BENCH_SEQ = benchmark_seq
BENCH_OMP = benchmark_omp


SRCDIR = src
TESTDIR = tests
BENCHDIR = benchmarks
EXAMPLEDIR = examples


SRCS = $(SRCDIR)/ndarray_core.c $(SRCDIR)/ndarray_creation.c $(SRCDIR)/ndarray_arithmetic.c \
       $(SRCDIR)/ndarray_linalg.c $(SRCDIR)/ndarray_manipulation.c $(SRCDIR)/ndarray_aggregation.c \
       $(SRCDIR)/ndarray_print.c $(SRCDIR)/ndarray_io.c $(SRCDIR)/ndarray_comparison.c


TEST_SOURCES = $(TESTDIR)/test_common.c \
               $(TESTDIR)/test_creation.c \
               $(TESTDIR)/test_operations.c \
               $(TESTDIR)/test_arithmetic.c \
               $(TESTDIR)/test_matmul.c \
               $(TESTDIR)/test_tensordot.c \
               $(TESTDIR)/test_stack.c \
               $(TESTDIR)/test_concat.c \
               $(TESTDIR)/test_take.c \
               $(TESTDIR)/test_transpose.c \
               $(TESTDIR)/test_reshape.c \
               $(TESTDIR)/test_aggregation.c \
               $(TESTDIR)/test_conditional.c \
               $(TESTDIR)/test_slice.c \
               $(TESTDIR)/test_comparison.c \
               $(TESTDIR)/test_main.c

TEST_OBJECTS = $(TEST_SOURCES:.c=.o)

OBJ = $(SRCS:.c=.o)
OBJ_SHARED = $(SRCS:.c=_shared.o)
TEST_OBJ = $(OBJ)
BENCH_OBJ = $(OBJ)

.PHONY: all clean test benchmark install uninstall lib static shared docs test-modular

all: $(BIN)

lib: static shared

static: $(LIB_STATIC)

shared: $(LIB_SHARED)

# Old monolithic test (for backward compatibility)
test: $(TEST_BIN)
	./$(TEST_BIN)

# New modular test suite
test-modular: ndarray_test_modular
	./ndarray_test_modular

benchmark:
	@$(BENCHDIR)/run_benchmark.sh

docs:
	@doxygen Doxyfile

install: lib
	install -d $(DESTDIR)$(LIBDIR)
	install -d $(DESTDIR)$(INCLUDEDIR)
	install -m 644 $(LIB_STATIC) $(DESTDIR)$(LIBDIR)/
	install -m 755 $(LIB_SHARED) $(DESTDIR)$(LIBDIR)/
	ln -sf $(LIB_SHARED) $(DESTDIR)$(LIBDIR)/$(LIB_SHARED_MAJOR)
	ln -sf $(LIB_SHARED_MAJOR) $(DESTDIR)$(LIBDIR)/$(LIB_SHARED_BASE)
	install -m 644 $(SRCDIR)/ndarray.h $(DESTDIR)$(INCLUDEDIR)/
	ldconfig -n $(DESTDIR)$(LIBDIR) 2>/dev/null || true

uninstall:
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_STATIC)
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_SHARED)
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_SHARED_MAJOR)
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_SHARED_BASE)
	rm -f $(DESTDIR)$(INCLUDEDIR)/ndarray.h

clean:
	rm -f $(BIN) $(TEST_BIN) ndarray_test_modular $(OBJ) $(OBJ_SHARED)
	rm -f $(TESTDIR)/*.o
	rm -f $(LIB_STATIC) $(LIB_SHARED) $(LIB_SHARED_MAJOR) $(LIB_SHARED_BASE)
	rm -f $(BENCH_SEQ) $(BENCH_OMP) benchmark_seq.txt benchmark_omp.txt
	rm -rf docs


$(LIB_STATIC): $(OBJ)
	ar rcs $@ $^

$(LIB_SHARED): $(OBJ_SHARED)
	$(CC) -shared -Wl,-soname,$(LIB_SHARED_MAJOR) -o $@ $^ $(LDFLAGS)


$(SRCDIR)/%.o: $(SRCDIR)/%.c $(SRCDIR)/ndarray.h $(SRCDIR)/ndarray_internal.h
	$(CC) $(CFLAGS) -c $< -o $@


$(SRCDIR)/%_shared.o: $(SRCDIR)/%.c $(SRCDIR)/ndarray.h $(SRCDIR)/ndarray_internal.h
	$(CC) $(CFLAGS_SHARED) -c $< -o $@

$(BIN): $(OBJ) $(EXAMPLEDIR)/example.c
	$(CC) $(CFLAGS) -o $@ $(EXAMPLEDIR)/example.c $(OBJ) $(LDFLAGS)

$(TEST_BIN): $(TESTDIR)/test_ndarray.o $(TEST_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(TEST_LDFLAGS)

$(TESTDIR)/test_ndarray.o: $(TESTDIR)/test_ndarray.c $(SRCDIR)/ndarray.h
	$(CC) $(CFLAGS) -c $(TESTDIR)/test_ndarray.c -o $@ -I$(SRCDIR)

# Modular test suite
ndarray_test_modular: $(TEST_OBJECTS) $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(TEST_LDFLAGS)

$(TESTDIR)/%.o: $(TESTDIR)/%.c $(SRCDIR)/ndarray.h $(TESTDIR)/test_common.h
	$(CC) $(CFLAGS) -c $< -o $@ -I$(SRCDIR)

